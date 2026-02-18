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
- **Font diversity**: Minimum ≥ 3 distinct Korean font families supplied via `fonts=` parameter; bundled `NEXONLv1GothicRegular.ttf` alone is insufficient for discriminator variance
- **Reproducibility**: Call `random.seed(SEED)` + `numpy.seed(SEED)` before generator instantiation; capture seed in metadata
- **Metadata wrapper**: TRDG does not natively log per-sample params; a thin wrapper must emit `synthetic_metadata_v{N}.json` with seed, font list, background type, distortion flags

### Integration Gap (Action Required Before Gate 4)

TRDG outputs `{id}.jpg` + `labels.txt`. Training pipeline expects LMDB or JSONL manifest.
A conversion script (`scripts/data/quality/trdg_to_jsonl.py`) must be implemented to bridge this gap before robustness training begins.

### Open Research Questions (Pre-Implementation)

| ID | Question | Blocks |
|---|---|---|
| RQ-01 | Are additional Korean `.ttf` fonts available in `../parent/` or project corpus? | Font diversity constraint |
| RQ-02 | What is the LMDB key/value schema in the training pipeline? | `trdg_to_jsonl.py` or `trdg_to_lmdb.py` |
| RQ-03 | Is TRDG installed in the active `uv` environment? (`uv run python -c "from trdg.generators import GeneratorFromStrings"`) | All wrapper scripting |
| RQ-04 | Character/word frequency distribution of real training data (from LMDB labels) | Custom `strings` list for `GeneratorFromStrings` |
| RQ-05 | Are domain-specific background images (scanned Korean document paper) available? | Background type decision |
| RQ-06 | Will the existing trained OCR model's CTC loss distribution serve as the discriminator, or is a separate binary classifier needed? | Gate 4 discriminator check |

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