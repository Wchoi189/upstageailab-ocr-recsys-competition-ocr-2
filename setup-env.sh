#!/bin/bash
# Temporary environment setup until uv sync is complete
set -a
[ -f .env ] && . .env
set +a

export PATH="$PWD/.venv/bin:$PATH"
