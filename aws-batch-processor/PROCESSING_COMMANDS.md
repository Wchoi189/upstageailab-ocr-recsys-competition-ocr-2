# KIE Dataset Processing Commands

## Dataset: baseline_val_canonical (404 images, split into 3 parts)

**Split Distribution**:
- Part 1: 135 images → UPSTAGE_API_KEY (3 RPS, concurrency=3)
- Part 2: 135 images → UPSTAGE_API_KEY2 (1 RPS, concurrency=1)
- Part 3: 134 images → UPSTAGE_API_KEY (3 RPS, concurrency=3)

---

## Terminal 1: Process Part 1 (UPSTAGE_API_KEY)

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor

# Set API key (if not in .env.local)
export UPSTAGE_API_KEY="your-key-here"

# Process part 1 with concurrency=3 (3 RPS rate limit)
uv run python scripts/process_split.py \
    --dataset baseline_val_canonical \
    --part 1 \
    --api-key-env UPSTAGE_API_KEY \
    --concurrency 3
```

**To resume if interrupted**:
```bash
uv run python scripts/process_split.py \
    --dataset baseline_val_canonical \
    --part 1 \
    --api-key-env UPSTAGE_API_KEY \
    --concurrency 3 \
    --resume
```

---

## Terminal 2: Process Part 2 (UPSTAGE_API_KEY2)

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor

# Set API key (if not in .env.local)
export UPSTAGE_API_KEY2="your-key-here"

# Process part 2 with concurrency=1 (1 RPS rate limit)
uv run python scripts/process_split.py \
    --dataset baseline_val_canonical \
    --part 2 \
    --api-key-env UPSTAGE_API_KEY2 \
    --concurrency 1
```

**To resume if interrupted**:
```bash
uv run python scripts/process_split.py \
    --dataset baseline_val_canonical \
    --part 2 \
    --api-key-env UPSTAGE_API_KEY2 \
    --concurrency 1 \
    --resume
```

---

## Terminal 3: Process Part 3 (UPSTAGE_API_KEY)

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor

# Set API key (if not in .env.local)
export UPSTAGE_API_KEY="your-key-here"

# Process part 3 with concurrency=3 (3 RPS rate limit)
uv run python scripts/process_split.py \
    --dataset baseline_val_canonical \
    --part 3 \
    --api-key-env UPSTAGE_API_KEY \
    --concurrency 3
```

**To resume if interrupted**:
```bash
uv run python scripts/process_split.py \
    --dataset baseline_val_canonical \
    --part 3 \
    --api-key-env UPSTAGE_API_KEY \
    --concurrency 3 \
    --resume
```

---

## Monitoring Progress

**In a separate terminal, run**:
```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor

# Monitor progress (run repeatedly)
watch -n 10 "python3 scripts/monitor_progress.py --dataset baseline_val_canonical"
```

**Or manually check**:
```bash
python3 scripts/monitor_progress.py --dataset baseline_val_canonical
```

---

## After All Parts Complete: Merge Splits

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor

# Merge all 3 parts into final output
python3 scripts/merge_splits.py --dataset baseline_val_canonical
```

**Output**: `data/output/baseline_val_canonical_pseudo_labels.parquet`

---

## Checkpoint Locations

- Part 1: `data/checkpoints/baseline_val_canonical_part1_prebuilt/`
- Part 2: `data/checkpoints/baseline_val_canonical_part2_prebuilt/`
- Part 3: `data/checkpoints/baseline_val_canonical_part3_prebuilt/`

**To check checkpoint status**:
```bash
ls -lh data/checkpoints/baseline_val_canonical_part*_prebuilt/*.parquet
```

---

## Estimated Processing Times

- **Part 1** (135 images, concurrency=3): ~58 seconds (~0.43s per image)
- **Part 2** (135 images, concurrency=1): ~175 seconds (~1.3s per image)
- **Part 3** (134 images, concurrency=3): ~58 seconds (~0.43s per image)

**Total**: ~5 minutes (all parts run in parallel)

---

## Next Dataset: baseline_test

After baseline_val_canonical completes:

1. Split: `python3 scripts/split_dataset.py --dataset baseline_test`
2. Process parts 1, 2, 3 in parallel (same commands, change dataset name)
3. Merge: `python3 scripts/merge_splits.py --dataset baseline_test`

---

## Next Dataset: baseline_train

After baseline_test completes:

1. Split: `python3 scripts/split_dataset.py --dataset baseline_train`
2. Process parts 1, 2, 3 in parallel (same commands, change dataset name)
3. Merge: `python3 scripts/merge_splits.py --dataset baseline_train`

**Note**: baseline_train has 3,595 images (~1,198 per part). Estimated time:
- Part 1: ~8.5 minutes
- Part 2: ~26 minutes
- Part 3: ~8.5 minutes
- Total: ~26 minutes (limited by part 2)
