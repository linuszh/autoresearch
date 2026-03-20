# autoresearch — finetuning mode

This is the finetuning variant of autoresearch. Instead of training a small GPT from scratch, you finetune a pretrained model (e.g. Qwen 3.5) on custom domain data using QLoRA.

## Setup

To set up a new finetuning experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `ft-mar20`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `finetune.py` — the file you modify. QLoRA finetuning: model loading, data pipeline, training loop, evaluation.
4. **Verify data exists**: Check that the user's `--data-dir` path contains `.txt` files.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Run baseline**: Run the finetuning script as-is to establish a baseline metric.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The finetuning script runs for a **fixed time budget** (default 5 minutes wall clock). You launch it as:

```
uv run finetune.py --data-dir /path/to/data > run.log 2>&1
```

**What you CAN do:**
- Modify `finetune.py` — this is the only file you edit. Fair game includes:
  - LoRA rank, alpha, target modules, dropout
  - Learning rate, schedule, optimizer settings
  - Batch size, gradient accumulation, sequence length
  - Data preprocessing (chunking strategy, document filtering)
  - Evaluation approach
  - Base model selection (different Qwen 3.5 sizes)

**What you CANNOT do:**
- Modify `prepare.py`, `train.py`, or `program.md`. These are for the pretraining system.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

**The goal is simple: get the lowest val_bpb on the held-out validation split.**

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**The first run**: Your first run should always establish the baseline with default settings.

## Output format

Once the script finishes it prints a summary:

```
---
val_bpb:            0.997900
val_perplexity:     12.34
val_loss:           2.345678
training_seconds:   300.1
total_seconds:      325.9
peak_vram_mb:       12345.6
total_tokens_M:     10.5
num_steps:          150
trainable_params_M: 25.3
base_model:         Qwen/Qwen3.5-9B
lora_rank:          16
epochs:             2
```

Extract key metrics:
```
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## Logging results

Log experiments to `results.tsv` (tab-separated). Header and 6 columns:

```
commit	val_bpb	memory_gb	status	base_model	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved — use 0.000000 for crashes
3. peak memory in GB (.1f) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. base model name (e.g. Qwen/Qwen3.5-9B)
6. short text description of what this experiment tried

Do NOT commit `results.tsv` — leave it untracked.

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `finetune.py` with an experimental idea
3. git commit
4. Run: `uv run finetune.py --data-dir /path/to/data > run.log 2>&1`
5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace.
7. Record results in `results.tsv`
8. If val_bpb improved (lower), keep the commit
9. If val_bpb is equal or worse, `git reset --hard` to previous commit

**Timeout**: Each experiment should take ~5 minutes + startup overhead. Kill runs exceeding 10 minutes.

**Crashes**: Fix trivial bugs and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Continue autonomously until manually interrupted.
