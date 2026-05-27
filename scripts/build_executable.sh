#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv pip install pyinstaller
uv run pyinstaller hoops.spec --clean --noconfirm

echo ""
echo "Built: dist/hoops"
echo "Size: $(du -sh dist/hoops | cut -f1)"
