#!/bin/bash
set -e

echo "=== Refactor Verification Script ==="

echo "[1/3] Verifying test collection (checking for import errors)..."
pytest --collect-only

echo "[2/3] Verifying Hydra config loading..."
python runners/train.py --help > /dev/null || { echo "Failed to load config!"; exit 1; }

echo "[3/3] Running fast dev run (dry run)..."
python runners/train.py debug=default || { echo "Fast dev run failed!"; exit 1; }

echo "=== Verification Successful! ==="
