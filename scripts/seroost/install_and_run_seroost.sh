#!/bin/bash

# Script to install Seroost and run indexing

set -e  # Exit immediately if a command exits with a non-zero status

echo "Installing Seroost and setting up indexing..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH"
    exit 1
fi

# Run the setup script with ephemeral dependency
echo "Running seroost indexing setup..."
uv run --with seroost python setup_seroost_indexing.py

echo "Seroost installation and indexing setup completed!"
