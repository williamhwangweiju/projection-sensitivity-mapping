#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install -q --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install -q \
  'transformers>=4.30,<5' \
  'datasets>=2.14,<4' \
  'pandas>=1.5' \
  'PyYAML>=6' \
  'tqdm>=4.65' \
  'pytest>=7'

echo "Install AIHWKit separately using IBM's wheel for the active Colab Python/GPU."
