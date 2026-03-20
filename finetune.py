"""
Finetune a pretrained model (e.g. Qwen 3.5) on custom text data using QLoRA.
Self-contained script — does not import from train.py.

Usage:
    uv run finetune.py --data-dir /path/to/txt_files
    uv run finetune.py --data-dir /path/to/txt_files --model-name Qwen/Qwen3.5-4B --time-budget 60
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Finetune a pretrained model on custom text data with QLoRA")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-9B",
                    help="HuggingFace model ID")
    p.add_argument("--data-dir", type=str, required=True,
                    help="Directory containing .txt files for training")
    p.add_argument("--time-budget", type=int, default=300,
                    help="Training wall-clock time budget in seconds")
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=4, help="Micro batch size")
    p.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    p.add_argument("--grad-accum-steps", type=int, default=8,
                    help="Gradient accumulation steps")
    p.add_argument("--val-ratio", type=float, default=0.05,
                    help="Fraction of data to hold out for validation")
    p.add_argument("--output-dir", type=str, default="./finetune_output",
                    help="Directory to save LoRA adapter")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_texts(data_dir):
    """Read all .txt files from data_dir, split into documents on double-newlines."""
    data_path = Path(data_dir)
    txt_files = sorted(data_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")
    print(f"Data: found {len(txt_files)} .txt files in {data_dir}")

    documents = []
    for f in txt_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        # Split on double-newlines to get logical sections
        sections = text.split("\n\n")
        for section in sections:
            section = section.strip()
            if len(section) >= 50:  # Skip very short fragments
                documents.append(section)

    print(f"Data: {len(documents)} documents after splitting and filtering")
    return documents


class ChunkedTextDataset(Dataset):
    """Dataset of fixed-length token chunks for causal LM training."""

    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        # Drop last incomplete chunk
        n_chunks = len(token_ids) // (seq_len + 1)
        self.data = torch.tensor(token_ids[:n_chunks * (seq_len + 1)], dtype=torch.long)
        self.data = self.data.view(n_chunks, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def prepare_datasets(documents, tokenizer, max_seq_len, val_ratio, seed):
    """Tokenize documents, concatenate with EOS, chunk, and split into train/val."""
    random.seed(seed)
    random.shuffle(documents)

    eos_id = tokenizer.eos_token_id

    # Tokenize all documents and concatenate with EOS separator
    all_ids = []
    for doc in documents:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(eos_id)

    total_tokens = len(all_ids)
    print(f"Data: {total_tokens:,} tokens total")

    if total_tokens < (max_seq_len + 1) * 2:
        raise ValueError(
            f"Not enough data: {total_tokens} tokens, need at least {(max_seq_len + 1) * 2}. "
            f"Add more text files or reduce --max-seq-len."
        )

    # Split into train/val
    n_val = max(1, int(total_tokens / (max_seq_len + 1) * val_ratio))
    n_val_tokens = n_val * (max_seq_len + 1)
    val_ids = all_ids[:n_val_tokens]
    train_ids = all_ids[n_val_tokens:]

    train_ds = ChunkedTextDataset(train_ids, max_seq_len)
    val_ds = ChunkedTextDataset(val_ids, max_seq_len)

    print(f"Data: {len(train_ds)} train chunks, {len(val_ds)} val chunks "
          f"(seq_len={max_seq_len})")

    if len(train_ds) == 0:
        raise ValueError("No training chunks. Add more data or reduce --max-seq-len.")

    return train_ds, val_ds

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, lora_rank, lora_alpha):
    """Load quantized model with LoRA adapters."""
    print(f"Model: loading {model_name} with 4-bit quantization...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    t1 = time.time()
    print(f"Model: loaded in {t1 - t0:.1f}s")
    print(f"Model: {total / 1e6:.1f}M total params, {trainable / 1e6:.1f}M trainable (LoRA r={lora_rank})")

    return model, tokenizer, trainable

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, tokenizer):
    """Compute validation loss, perplexity, and bits-per-byte."""
    model.eval()
    total_loss = 0.0
    total_nats = 0.0
    total_bytes = 0
    n_batches = 0

    # Precompute token-to-byte-length lookup
    vocab_size = len(tokenizer)
    token_byte_lengths = torch.zeros(vocab_size, dtype=torch.int32)
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
            token_byte_lengths[token_id] = len(token_str.encode("utf-8"))
        except Exception:
            token_byte_lengths[token_id] = 0
    token_byte_lengths = token_byte_lengths.cuda()

    for batch in val_loader:
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)

        total_loss += outputs.loss.item()
        n_batches += 1

        # BPB calculation
        logits = outputs.logits
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)

        per_token_loss = F.cross_entropy(
            shift_logits, shift_labels, reduction="none"
        )

        nbytes = token_byte_lengths[shift_labels]
        mask = nbytes > 0
        total_nats += (per_token_loss * mask.float()).sum().item()
        total_bytes += nbytes.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    bpb = total_nats / (math.log(2) * max(total_bytes, 1))

    model.train()
    return avg_loss, perplexity, bpb

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step, total_steps, base_lr, warmup_frac=0.05):
    """Cosine LR with linear warmup."""
    warmup_steps = int(total_steps * warmup_frac)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t_start = time.time()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Load model and tokenizer
    model, tokenizer, trainable_params = load_model_and_tokenizer(
        args.model_name, args.lora_rank, args.lora_alpha
    )

    # Load and prepare data
    documents = load_texts(args.data_dir)
    train_ds, val_ds = prepare_datasets(
        documents, tokenizer, args.max_seq_len, args.val_ratio, args.seed
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Warn about small datasets
    n_epochs_estimate = (args.time_budget * args.batch_size * args.grad_accum_steps) / max(len(train_ds), 1)
    if n_epochs_estimate > 3:
        print(f"Warning: estimated ~{n_epochs_estimate:.0f} epochs over data. "
              f"Consider adding more data to avoid overfitting.")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # Training loop with time budget
    print(f"\nTraining: time budget = {args.time_budget}s, "
          f"batch_size = {args.batch_size}, grad_accum = {args.grad_accum_steps}")
    print()

    model.train()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    step = 0
    micro_step = 0
    total_training_time = 0.0
    smooth_loss = 0.0
    epoch = 0
    total_tokens = 0

    # Estimate total steps for LR schedule (rough, based on time budget)
    # We'll refine this after the first few steps
    estimated_total_steps = 1000  # Will be updated

    gc.collect()
    torch.cuda.empty_cache()

    t_train_start = time.time()
    train_iter = iter(train_loader)

    while True:
        t0 = time.time()
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)

            with autocast_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.grad_accum_steps

            loss.backward()
            accum_loss += loss.item()
            micro_step += 1

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update LR
        lr = get_lr(step, estimated_total_steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        step += 1
        total_tokens += args.batch_size * args.max_seq_len * args.grad_accum_steps

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        # Only count training time after warmup (first 3 steps may include compilation)
        if step > 3:
            total_training_time += dt

        # Update total step estimate after first few steps
        if step == 5 and dt > 0:
            estimated_total_steps = max(step, int(args.time_budget / dt))

        # Smoothed loss logging
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * accum_loss
        debiased_loss = smooth_loss / (1 - ema_beta ** step)

        progress = min(total_training_time / args.time_budget, 1.0) if args.time_budget > 0 else 0
        remaining = max(0, args.time_budget - total_training_time)
        tok_per_sec = int(args.batch_size * args.max_seq_len * args.grad_accum_steps / dt) if dt > 0 else 0

        print(f"\rstep {step:04d} ({100*progress:.1f}%) | loss: {debiased_loss:.4f} | "
              f"lr: {lr:.2e} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | "
              f"epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # Fast fail
        if math.isnan(accum_loss) or accum_loss > 100:
            print("\nFAIL: loss diverged")
            exit(1)

        # GC management
        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()

        # Time's up
        if step > 3 and total_training_time >= args.time_budget:
            break

    print()  # newline after \r log

    # Final evaluation
    print("\nEvaluating on validation set...")
    val_loss, val_perplexity, val_bpb = evaluate(model, val_loader, tokenizer)

    # Save adapter
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")

    # Summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_bpb:            {val_bpb:.6f}")
    print(f"val_perplexity:     {val_perplexity:.2f}")
    print(f"val_loss:           {val_loss:.6f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"total_tokens_M:     {total_tokens / 1e6:.1f}")
    print(f"num_steps:          {step}")
    print(f"trainable_params_M: {trainable_params / 1e6:.1f}")
    print(f"base_model:         {args.model_name}")
    print(f"lora_rank:          {args.lora_rank}")
    print(f"epochs:             {epoch}")


if __name__ == "__main__":
    main()
