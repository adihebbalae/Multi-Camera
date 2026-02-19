#!/usr/bin/env bash
# Install dependencies so nuscens_build.py is runnable.
# Run from the Multi-Camera repo root.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
echo "Installing multicamera package (editable)..."
pip install -e .

echo "Installing nuscenes dataset builder dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Done. Run: python -m nuscenes.datasetbuilder.nuscens_build --help"
