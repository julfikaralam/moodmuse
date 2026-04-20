#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  MoodMuse — Start script
#  Run: chmod +x run.sh && ./run.sh
# ═══════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
STAMP="$ROOT/.install_stamp"
REQ="$ROOT/requirements.txt"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

echo ""
echo -e "  ${BOLD}🧠 MoodMuse${NC}"
echo    "  ──────────────────────────────────────"

# Find Python 3.10+
if python3 -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null; then
  PY=python3
elif python -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null; then
  PY=python
else
  echo -e "  ${R}✗ Python 3.10+ required.${NC}"
  exit 1
fi

PY_VER=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  ${G}✓ Python $PY_VER${NC}"

# Create venv if missing (skip if conda or venv already active)
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  if [ ! -d "$VENV" ]; then
    echo -e "  ${Y}  Creating virtual environment…${NC}"
    $PY -m venv "$VENV"
  fi
  source "$VENV/bin/activate"
  PY=python
  echo -e "  ${G}✓ Virtual environment active${NC}"
else
  PY=python
  echo -e "  ${G}✓ Using active environment${NC}"
fi

# Install deps only when requirements.txt changes
REQ_HASH=$(md5 -q "$REQ" 2>/dev/null || md5sum "$REQ" | awk '{print $1}')
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$REQ_HASH" ]; then
  echo -e "  ${Y}  Installing packages (first run)…${NC}"
  $PY -m pip install --quiet --upgrade pip
  $PY -m pip install --quiet -r "$REQ"
  echo "$REQ_HASH" > "$STAMP"
  echo -e "  ${G}✓ Packages installed${NC}"
else
  echo -e "  ${G}✓ Packages up to date${NC}"
fi

echo ""
echo -e "  ${BOLD}${G}→ http://127.0.0.1:5000${NC}"
echo    "  Press Ctrl+C to stop."
echo    "  ──────────────────────────────────────"
cd "$ROOT"
$PY app.py
