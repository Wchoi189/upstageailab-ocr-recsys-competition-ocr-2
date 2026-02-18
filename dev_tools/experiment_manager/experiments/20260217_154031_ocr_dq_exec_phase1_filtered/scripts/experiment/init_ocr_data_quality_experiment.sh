#!/usr/bin/env bash
# init_ocr_data_quality_experiment.sh
# Non-mutating workspace readiness checker for OCR DQ remediation sessions.
# Validates active experiment state, gate artifacts, and locked configuration.
# Usage: bash dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/experiment/init_ocr_data_quality_experiment.sh

set -euo pipefail

# ── Constants ─────────────────────────────────────────────────────────────────

EXPERIMENT_ID="20260217_154031_ocr_dq_exec_phase1_filtered"
EXPERIMENT_DIR="dev_tools/experiment_manager/experiments/${EXPERIMENT_ID}"
METADATA_DIR="${EXPERIMENT_DIR}/.metadata"
CONFIG_FILE="configs/data/quality/remediation.yaml"

GATE0_ARTIFACTS=(
  "data/audit/defect_prevalence.json"
  "data/audit/loss_percentiles.json"
  "data/audit/truncation_analysis.json"
)

GATE1_ARTIFACTS=(
  "specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-phase-gates.md"
  "specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-metric-criteria.md"
  "specs/003-ocr-data-quality-remediation/planning/ocr-clean-holdout-protocol.md"
  "specs/003-ocr-data-quality-remediation/planning/ocr-annotation-qa-protocol.md"
  "specs/003-ocr-data-quality-remediation/EXECUTION_RUNBOOK.md"
)

SHARED_MODULES=(
  "scripts/data/quality/manifest_io.py"
  "scripts/data/quality/defect_rules.py"
  "scripts/data/quality/quality_scoring.py"
  "scripts/data/quality/gate_metrics.py"
  "scripts/audit/analyze_defect_distribution.py"
  "scripts/audit/compute_loss_distribution.py"
  "scripts/audit/analyze_sequence_lengths.py"
)

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "  ${GRN}[PASS]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; FAIL_COUNT=$((FAIL_COUNT+1)); }
warn() { echo -e "  ${YLW}[WARN]${NC} $1"; }

FAIL_COUNT=0

# ── 1. Experiment directory ────────────────────────────────────────────────────

echo ""
echo "=== [1/5] EXPERIMENT WORKSPACE ==="
if [[ -d "${EXPERIMENT_DIR}" ]]; then
  pass "Experiment dir: ${EXPERIMENT_DIR}"
else
  fail "Experiment dir missing: ${EXPERIMENT_DIR}"
fi
if [[ -f "${EXPERIMENT_DIR}/manifest.json" ]]; then
  pass "manifest.json present"
else
  fail "manifest.json missing"
fi
if [[ -d "${METADATA_DIR}" ]]; then
  pass ".metadata dir present"
else
  fail ".metadata dir missing"
fi

# ── 2. Gate 0 artifacts ────────────────────────────────────────────────────────

echo ""
echo "=== [2/5] GATE 0 — BASELINE DIAGNOSTIC ARTIFACTS ==="
for artifact in "${GATE0_ARTIFACTS[@]}"; do
  if [[ -f "${artifact}" ]]; then
    pass "${artifact}"
  else
    fail "${artifact} — MISSING"
  fi
done

# ── 3. Gate 1 artifacts ────────────────────────────────────────────────────────

echo ""
echo "=== [3/5] GATE 1 — POLICY DESIGN ARTIFACTS ==="
for artifact in "${GATE1_ARTIFACTS[@]}"; do
  if [[ -f "${artifact}" ]]; then
    pass "${artifact}"
  else
    fail "${artifact} — MISSING"
  fi
done

# ── 4. Shared infrastructure modules ──────────────────────────────────────────

echo ""
echo "=== [4/5] SHARED INFRASTRUCTURE MODULES ==="
for module in "${SHARED_MODULES[@]}"; do
  if [[ -f "${module}" ]]; then
    pass "${module}"
  else
    fail "${module} — MISSING"
  fi
done

# ── 5. Locked configuration ───────────────────────────────────────────────────

echo ""
echo "=== [5/5] LOCKED CONFIGURATION ==="
if [[ -f "${CONFIG_FILE}" ]]; then
  UNREADABLE_MIN_LEN=$(grep 'unreadable_min_len' "${CONFIG_FILE}" | grep -o '[0-9]*' | head -1 || echo "NOT_FOUND")
  if [[ "${UNREADABLE_MIN_LEN}" == "0" ]]; then
    pass "${CONFIG_FILE}: unreadable_min_len=0 (LOCKED)"
  else
    fail "${CONFIG_FILE}: unreadable_min_len=${UNREADABLE_MIN_LEN} — expected 0"
  fi
else
  fail "${CONFIG_FILE} — MISSING"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== WORKSPACE STATUS ==="
if [[ "${FAIL_COUNT}" -eq 0 ]]; then
  echo -e "${GRN}READY${NC} — All checks passed. Gate 0 and Gate 1: PASS."
  echo ""
  echo "Next execution block: Phase 6 (US4), T033+T034 in parallel"
  echo "  T033: scripts/data/quality/upstage_validator.py"
  echo "  T034: scripts/data/quality/paddle_validator.py"
  echo ""
  echo "Latest handover:"
  LATEST=$(ls -1t "${METADATA_DIR}"/*_SESSION_HANDOVER.md 2>/dev/null | head -1 || echo "none")
  echo "  ${LATEST}"
else
  echo -e "${RED}NOT READY${NC} — ${FAIL_COUNT} check(s) failed. Resolve before proceeding."
fi
echo ""
