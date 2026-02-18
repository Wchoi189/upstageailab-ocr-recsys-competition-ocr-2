# OCR Synthetic Data Specification

## Purpose

Define initial synthetic data targets and quality constraints for robust OCR remediation experiments.

## Initial Composition Target

- Pilot blend: `70% synthetic : 30% verified real`
- Intended use: robustness phase only (post correction-queue readiness)

## Coverage Targets

Synthetic set should cover:
- Korean-only strings
- mixed Korean/ASCII strings
- punctuation and symbol-heavy cases
- long-tail sequence lengths near tokenizer boundary
- clipping-like visual distortions and spacing variance

## Generation Requirements

- Deterministic generation seed captured per batch
- Font/background/noise parameters logged
- Per-sample metadata includes generator profile and label source

## Generation Tool: TRDG (TextRecognitionDataGenerator v1.8.0)

**Location**: `../parent/DATA_SYNTHETIC/TextRecognitionDataGenerator/`
**License**: MIT
**Python API** (preferred over CLI for pipeline integration):
- `GeneratorFromStrings` — primary interface; accepts pre-filtered string list
- `GeneratorFromDict` — fallback for syllable-level generation only

### Reference Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `language` | `ko` | Korean font/dict loading |
| Image height (`size`) | `32` | Matches training crop height |
| Output format | `name_format=2` (labels.txt) | Separates image from label cleanly |
| `background_type` | `0` (Gaussian noise) | Paper simulation; swap for `-b 3` with domain images if available |
| `skewing_angle` | `5`, `random_skew=True` | Anti-discriminator visual variance |
| `blur` | `1–2`, `random_blur=True` | Print degradation simulation |
| Thread count | `8` | Parallel generation |

### Implementation Constraints

- **Tokenizer boundary**: Pre-filter all strings to `len(text) <= 25` before passing to generator
- **Mixed Korean/ASCII**: Construct strings manually in wrapper (e.g., `가나다123`, `ABC전화`); TRDG's `ko` dict is single-script only
- **Font diversity**: Minimum ≥ 3 distinct Korean font families supplied via `fonts=` parameter. Resolved (RQ-01): use `/usr/share/fonts/truetype/nanum/NanumGothic.ttf` (sans), `/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf` (serif), `/usr/share/fonts/truetype/unfonts-core/UnBatang.ttf` (traditional). Recommended blend: 40% NanumGothic, 30% NanumMyeongjo, 20% NEXONLv1Gothic, 10% UnBatang.
- **Reproducibility**: Call `random.seed(SEED)` + `numpy.seed(SEED)` before generator instantiation; capture seed in metadata
- **Metadata wrapper**: TRDG does not natively log per-sample params; a thin wrapper must emit `synthetic_metadata_v{N}.json` with seed, font list, background type, distortion flags

### Integration Gap (Action Required Before Gate 4)

TRDG outputs `{id}.jpg` + `labels.txt`. Training pipeline expects LMDB or JSONL manifest.
A conversion script (`scripts/data/quality/trdg_to_jsonl.py`) must be implemented to bridge this gap before robustness training begins.

### Research Questions — Resolved (2026-02-18)

| ID | Question | Status | Finding |
|---|---|---|---|
| RQ-01 | Additional Korean `.ttf` fonts in `../parent/`? | **RESOLVED** | Installed via `apt`: `fonts-nanum`, `fonts-noto-cjk`, `fonts-unfonts-core` (Ubuntu 22.04). 54 Korean-capable fonts available; 12 Nanum TTFs at `/usr/share/fonts/truetype/nanum/`. Recommended TRDG set: `NanumGothic.ttf` (sans), `NanumMyeongjo.ttf` (serif), `UnBatang.ttf` (traditional serif). Font diversity constraint MET (≥ 3 families). |
| RQ-02 | LMDB key/value schema in training pipeline? | **RESOLVED** | `image-{idx:09d}` (bytes), `label-{idx:09d}` (UTF-8 str), `num-samples` (int). Pipeline entry: `ocr/domains/recognition/data/lmdb_dataset.py`. Conversion script target: `scripts/data/quality/trdg_to_jsonl.py` (JSONL manifest → LMDB via existing ingestion path). |
| RQ-03 | TRDG installed in active `uv` env? | **BLOCKED** | NOT installed. `uv run python -c "from trdg.generators import GeneratorFromStrings"` raises `ModuleNotFoundError`. Action: `uv add trdg` or install from `../parent/DATA_SYNTHETIC/TextRecognitionDataGenerator/` before Gate 4. |
| RQ-04 | Character/word frequency distribution of real training data? | **OPEN** | Not computed. Requires LMDB label scan. Defer until Gate 3 (correction queue complete). Use uniform syllable sampling from TRDG bundled dict as interim fallback. |
| RQ-05 | Domain-specific background images available? | **RESOLVED** | No Korean document paper backgrounds found. `synthtiger/resources/image/` contains only generic stock photos (bedroom, coffee, farm, etc.). Decision: use `background_type=0` (Gaussian noise) as default; `background_type=3` (custom) only if corpus backgrounds are sourced post-Gate-3. |
| RQ-06 | CTC loss distribution as discriminator vs. separate classifier? | **DEFERRED** | Requires trained model in eval mode (RISK-01). Default recommendation: use CTC loss distribution as discriminator proxy (lower implementation cost); escalate to binary classifier only if discriminator accuracy > 70% threshold is exceeded in Gate 4 pilot. |

---

## Quality Constraints

- Synthetic-vs-real discriminator accuracy must remain `<= 70%`
- Synthetic-only validation CER must remain `<= 10%`
- Script distribution should not collapse any critical class

## Required Artifacts

- `data/generated/recognition/synthetic_v{N}.jsonl`
- `data/generated/recognition/synthetic_metadata_v{N}.json`
- `data/audit/synthetic_quality_report_v{N}.json`

## Validation Workflow

1. Run schema checks on generated manifest
2. Compare distribution against verified real subset
3. Run discriminator and synthetic-only CER checks
4. Approve/reject batch for augmentation

## Rejection Conditions

- Discriminator `> 70%`
- Synthetic-only CER `> 10%`
- Missing metadata lineage
- Severe script distribution skew

## Versioning

- Every synthetic batch must be versioned and immutable after approval
- Any regeneration increments version and records reason code

## Governance Notes

- Synthetic data supports remediation but never replaces clean holdout evaluation.
- Gate decisions are always anchored to clean-holdout metrics first.