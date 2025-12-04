# OCR Experiment Quickstart

**For AI Agents**: Rapid-start workflow.

## Prerequisites

- Read `AgentQMS/knowledge/agent/ocr_experiment_agent.md`
- VLM configured (see `AgentQMS/vlm/README.md`)
- `cd experiment-tracker/`

## 5-Minute Workflow

### 1. Start Experiment (30 seconds)

```bash
./scripts/start-experiment.py \
  --type perspective_correction \
  --intention "Investigate corner detection failures in low-contrast urban scenes"
```

**Returns**: Experiment ID (e.g., `20241204_143000_perspective_correction`)

### 2. Record Baseline Artifacts (1 minute)

```bash
# Record baseline image
./scripts/record-artifact.py \
  --path outputs/baseline/urban_scene_001.jpg \
  --type baseline \
  --metadata '{"scene": "urban_low_contrast", "technique": "baseline_v1"}'

# Record failure case
./scripts/record-artifact.py \
  --path outputs/failures/urban_scene_001_failed.jpg \
  --type poor_performance \
  --metadata '{"scene": "urban_low_contrast", "failure_mode": "corner_overshoot"}'
```

### 3. Run VLM Defect Analysis (2 minutes)

```bash
# Analyze failure with VLM
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image outputs/failures/urban_scene_001_failed.jpg \
  --mode defect \
  --output-format markdown \
  --output $(pwd)/artifacts/vlm_defect_analysis_001.md

# Auto-populate incident report
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image outputs/failures/urban_scene_001_failed.jpg \
  --mode defect \
  --auto-populate \
  --incident-report $(pwd)/incident_reports/corner_overshoot.md
```

### 4. Document & Assess

```bash
./scripts/record-decision.py --decision "DECISION" --rationale "RATIONALE"
./scripts/log-insight.py --insight "INSIGHT"
./scripts/generate-assessment.py --template visual-evidence-cluster --verbose minimal
```

## Common Patterns

### Pattern 1: Quick Failure Investigation (5 min)

```bash
# 1. Start
./scripts/start-experiment.py --type ocr_training --intention "Debug batch 42 failures"

# 2. Record failures
./scripts/record-artifact.py --path outputs/batch42_failure_*.jpg --type poor_performance

# 3. VLM analysis
for img in outputs/batch42_failure_*.jpg; do
  uv run python -m AgentQMS.vlm.cli.analyze_image_defects --image "$img" --mode defect
done

# 4. Cluster assessment
./scripts/generate-assessment.py --template visual-evidence-cluster --verbose minimal
```

### Pattern 2: Parameter Tuning (10 min)

```bash
# 1. Start
./scripts/start-experiment.py --type preprocessing --intention "Tune shadow removal parameters"

# 2. Record baseline
./scripts/record-artifact.py --path outputs/baseline.jpg --type baseline

# 3. Test variations (record each with metadata)
./scripts/record-artifact.py \
  --path outputs/alpha_0.5.jpg \
  --type improved \
  --metadata '{"alpha": 0.5, "beta": 0.2}'

# 4. Compare
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image outputs/baseline.jpg \
  --mode comparison \
  --comparison-image outputs/alpha_0.5.jpg

# 5. Document optimal params
./scripts/record-decision.py --decision "Use alpha=0.5" --rationale "Best contrast preservation"
```

### Pattern 3: Negative Results (8 min)

```bash
# 1. Start
./scripts/start-experiment.py --type synthetic_data --intention "Test GAN-based augmentation"

# 2. Record all attempts
for approach in gan_v1 gan_v2 gan_v3; do
  ./scripts/record-artifact.py \
    --path outputs/${approach}_result.jpg \
    --type poor_performance \
    --metadata "{\"approach\": \"${approach}\", \"success\": false}"
done

# 3. VLM analysis WHY each failed
for img in outputs/gan_*_result.jpg; do
  uv run python -m AgentQMS.vlm.cli.analyze_image_defects --image "$img" --mode defect
done

# 4. Negative result log
./scripts/generate-assessment.py --template run-log-negative-result --verbose minimal

# 5. Log insights
./scripts/log-insight.py --insight "GAN approaches produce unrealistic text distortions"
```

## Experiment Types Reference

| Type | Use Case | Common Metadata |
|------|----------|-----------------|
| `perspective_correction` | Geometric transforms, corner detection | `technique`, `failure_mode`, `geometry` |
| `ocr_training` | Model training runs, batch processing | `model`, `batch`, `epoch`, `metrics` |
| `synthetic_data` | Data augmentation, generation | `generator`, `parameters`, `quality` |
| `preprocessing` | Image preprocessing pipelines | `method`, `parameters`, `scene_type` |
| `evaluation` | Model evaluation, benchmarking | `model`, `dataset`, `metrics` |

## Assessment Templates Reference

| Template | Purpose | When to Use |
|----------|---------|-------------|
| `visual-evidence-cluster` | Group failures by visual pattern | Multiple similar failures |
| `triad-deep-dive` | Input/output discrepancy analysis | Single complex failure |
| `ab-regression` | Baseline vs. experiment comparison | Before/after comparison |
| `run-log-negative-result` | Document unsuccessful approaches | Failed experiments |

## VLM Analysis Modes

| Mode | Output | Use Case |
|------|--------|----------|
| `defect` | Defect description | Analyze failure artifacts |
| `input` | Input characteristics | Document input properties |
| `comparison` | Before/after comparison | Compare baseline vs. result |
| `comprehensive` | All modes combined | Complete analysis |

## After Experiment

```bash
./scripts/generate-feedback.py
./scripts/export-experiment.py --format archive --destination ./exports
git commit -m "feat(exp): [YYYYMMDD_HHMMSS]"
```

## Troubleshooting

### VLM Analysis Fails

```bash
# Check backend status
echo $OPENROUTER_API_KEY
echo $SOLAR_PRO2_API_KEY

# Verify image resolution
identify -format "%wx%h" path/to/image.jpg  # Should be ≤ 2048px

# Check VLM config
cat AgentQMS/vlm/config.yaml
```

### Script Errors

```bash
# Verify Python environment
uv run python --version

# Check experiment state
cat state.json

# Validate experiment structure
python scripts/validate-experiment.py
```

### Missing Context

```bash
# Check experiment intention
cat .metadata/experiment.yaml

# List all artifacts
ls -la artifacts/

# Review decisions
cat .metadata/decisions.yaml
```

---

**Version**: 1.1
**Last Updated**: 2024-12-04 12:00 (KST)
