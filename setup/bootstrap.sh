#!/bin/bash
# One-shot setup for a fresh machine. Run from anywhere after clone; idempotent.
#   bash setup/bootstrap.sh
# Installs the pinned Python deps, then smoke-tests the engine and torch.
# The competition SDK (data/cg/) and card sheet (data/EN_Card_Data.csv) must
# already be in place: see data/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."

# PTCG_REQUIREMENTS overrides the pinned file, e.g. a copy with the torch
# pin relaxed for a CUDA 12.x host (see the note at the top of
# setup/requirements.txt).
REQS="${PTCG_REQUIREMENTS:-setup/requirements.txt}"
PIP="pip install"
[ -n "${VIRTUAL_ENV:-}" ] && command -v uv >/dev/null 2>&1 && PIP="uv pip install"
$PIP -r "$REQS"

[ -d data/cg ] || { echo "data/cg missing: copy the competition SDK there (data/README.md)"; exit 1; }
[ -f data/EN_Card_Data.csv ] || { echo "data/EN_Card_Data.csv missing (data/README.md)"; exit 1; }

python3 - <<'EOF'
import sys
sys.path.insert(0, "data")
from cg.api import all_card_data
print("engine ok, cards:", len(all_card_data()))
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
EOF

[ -f data/carddb.json ] || PYTHONPATH=data python3 tools/build_carddb.py
python3 tools/pool_names.py --pool decks/pool --write data/pool_names.txt
echo "bootstrap ok"
