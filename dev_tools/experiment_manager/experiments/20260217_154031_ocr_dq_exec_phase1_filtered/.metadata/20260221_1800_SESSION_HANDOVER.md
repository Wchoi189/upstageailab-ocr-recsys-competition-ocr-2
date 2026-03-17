# SESSION_CONTEXT: 2026-02-21T18:00
**Status:** CONCLUDED
**Current Gate:** Gate 4 — SCRIPT COMPLETE (pending pilot run)

## 1. STATE_INVENTORY (Artifacts — changes from 20260221_1600)

- `[A38]` `scripts/data/quality/trdg_to_jsonl.py` : NEW — Gate 4 bridge script. TRDG `GeneratorFromStrings` → JPEG images + JSONL manifest + metadata JSON.

All previous artifacts (A01–A37) unchanged.

## 2. COMPUTE_FINDINGS (Metrics & Telemetry)

### Gate 4 Script Validation (live test, 5 samples)

| Metric | Value |
|---|---|
| Throughput | **131 samples/s** (CPU, TRDG GeneratorFromStrings) |
| Errors | 0 |
| Font round-robin | NanumGothic → NanumMyeongjo → UnBatang → (cycle) |
| Mixed Korean/ASCII | verified (`ABC전화`, `가나다123` both generated correctly) |
| JPEG sizes | 1.7–2.6 KB per sample @ height=32px |

### Script Interface

```
python scripts/data/quality/trdg_to_jsonl.py \
  --strings-file <path> \
  --output-dir data/generated/recognition \
  --version 1 \
  --seed 42 \
  [--count N] \
  [--fonts FONT [FONT ...]] \
  [--size 32] \
  [--background-type 0] \
  [--skewing-angle 5] \
  [--blur 1] \
  [--dry-run]
```

### Output Artifacts (per invocation)

| Artifact | Path pattern |
|---|---|
| JSONL manifest | `{output_dir}/synthetic_v{N}.jsonl` |
| Images | `{output_dir}/synthetic_v{N}_images/{idx:09d}.jpg` |
| Metadata | `{output_dir}/synthetic_metadata_v{N}.json` |

### JSONL Record Schema

```json
{
  "sample_id": "synthetic_v1_000000001",
  "image_path": "<abs_path>/synthetic_v1_images/000000001.jpg",
  "gt_text": "한국가스안전",
  "source": "trdg",
  "font": "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
  "background_type": 0,
  "skewing_angle": 5,
  "blur": 1,
  "version": "v1",
  "seed": 42
}
```

## 3. LOGIC_DECISIONS (Architectural Lock-in)

- **DECISION_30**: `trdg_to_jsonl.py` uses Python API (`GeneratorFromStrings`) — NOT CLI — per spec "Python API preferred for pipeline integration". CLI would require subprocess + parsing labels.txt; Python API yields `(PIL.Image, label)` tuples directly.
- **DECISION_31**: Font round-robin tracked by mirroring TRDG's internal formula `(idx-1) % len(fonts)`. No TRDG internals patched.
- **DECISION_32**: JSONL `image_path` stored as absolute resolved path (consistent with `path_roots` RETIRED convention, T040). Downstream consumers use absolute paths.
- **DECISION_33**: `--count -1` = one pass through strings file (len(strings) total). Pass `--count N` to repeat/cycle strings for larger batches.
- **DECISION_34**: `random_skew=True` and `random_blur=True` always enabled (hardcoded) per spec anti-discriminator requirement; `--skewing-angle` and `--blur` control the maximum magnitude only.

## 4. EXECUTION_QUEUE (Pending Tasks)

### Immediate Next Steps
1. **Gate 4 pilot run**: Generate ~1000 synthetic samples from holdout label corpus
   - Extract labels from `holdout_clean_v2.jsonl` → `strings_v1.txt` (422 unique labels, pre-filtered ≤25 chars)
   - Run: `python scripts/data/quality/trdg_to_jsonl.py --strings-file strings_v1.txt --output-dir data/generated/recognition --version 1 --seed 42 --count 1000`
   - Output: `data/generated/recognition/synthetic_v1.jsonl` + images
2. **Gate 4 discriminator check**: Run PaddleOCR on synthetic_v1 → compute CER; verify ≤10%
3. **RISK-02**: Manual review of 50 `script_mismatch` examples — calibrate 0.30 threshold
4. **RISK-03**: len=299 outlier (idx 578326) — source identification + exclusion rec

### Gate 4 Exit Criteria (from spec)
- Synthetic-only validation CER: `<= 10%`
- Synthetic-vs-real discriminator accuracy: `<= 70%`
- `70%:30%` blend confirmed viable for pilot

## 5. HANDOVER_TOKEN (Next Session Start Point)

> **Gate 4 bridge script COMPLETE and verified.**
>
> `scripts/data/quality/trdg_to_jsonl.py` operational (131 samples/s, 3-font round-robin, mixed Korean/ASCII verified).
>
> **Next priority: Gate 4 pilot run.**
>
> Step 1: Extract unique labels (≤25 chars) from `data/processed/recognition/holdout_clean_v2.jsonl`:
> ```bash
> python3 -c "
> import json, pathlib
> labels = [json.loads(l)['gt_text'] for l in pathlib.Path('data/processed/recognition/holdout_clean_v2.jsonl').read_text().splitlines()]
> filtered = [t for t in labels if len(t) <= 25]
> pathlib.Path('data/generated/recognition/strings_v1.txt').parent.mkdir(parents=True, exist_ok=True)
> pathlib.Path('data/generated/recognition/strings_v1.txt').write_text('\n'.join(filtered))
> print(len(filtered), 'strings written')
> "
> ```
> Step 2: Run generation:
> ```bash
> python scripts/data/quality/trdg_to_jsonl.py \
>   --strings-file data/generated/recognition/strings_v1.txt \
>   --output-dir data/generated/recognition \
>   --version 1 --seed 42 --count 1000
> ```
> Step 3: Run Gate 4 discriminator check (CER via PaddleOCR on synthetic_v1.jsonl).
