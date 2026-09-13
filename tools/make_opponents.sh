#!/bin/bash
# Build Kaggle-format agent bundles from a torch checkpoint, one per deck.
#
#   tools/make_opponents.sh <model.pt> <tag> [heads] [short=deck_name ...]
#   e.g. tools/make_opponents.sh experiments/train-native_run1/model_005000.pt e5000 8 \
#            marnie=marnie_s_grimmsnarl_ex_spikemuth_gym__05 \
#            garchomp=cynthia_s_garchomp_ex_cynthia_s_roserade_cynth__01
#
# Produces opponents/<tag>_<short>/ with the layout ptcg.rl.arena.load_kaggle_agent
# expects (main.py + model.pt + deck.csv + vendored ptcg/, cg/, data/). The same
# layout is a valid competition submission (tar the directory contents).
#
# HEADS must match the checkpoint's attention head count: a wrong value loads
# silently and computes different attention.
set -e
cd "$(dirname "$0")/.."
CKPT="${1:?usage: make_opponents.sh <model.pt> <tag> [heads] [short=deck ...]}"
TAG="${2:?need a tag, e.g. e5000}"
HEADS="${3:-8}"
shift 3 2>/dev/null || shift $#
POOL="${POOL:-decks/pool}"
PAIRS=("$@")
[ ${#PAIRS[@]} -gt 0 ] || { echo "need at least one short=deck_name pair"; exit 1; }
[ -f "$CKPT" ] || { echo "NO CKPT $CKPT"; exit 1; }
[ -d data/cg ] || { echo "data/cg missing (competition SDK, see data/README.md)"; exit 1; }
[ -f data/EN_Card_Data.csv ] || { echo "data/EN_Card_Data.csv missing (see data/README.md)"; exit 1; }

build() {   # $1 = short name, $2 = deck basename
  local out="opponents/${TAG}_$1"
  [ -f "$POOL/$2.csv" ] || { echo "NO DECK $POOL/$2.csv"; exit 1; }
  rm -rf "$out"; mkdir -p "$out/data" "$out/ptcg/rl"
  cp agent/main.py "$out/main.py"
  cp "$CKPT" "$out/model.pt"
  cp "$POOL/$2.csv" "$out/deck.csv"
  cp -r data/cg "$out/cg"
  cp data/embed_pca.npz data/lingering_effects.json data/EN_Card_Data.csv "$out/data/"
  cp ptcg/__init__.py ptcg/tracker.py ptcg/effects.py "$out/ptcg/"
  for m in __init__ buffers cards encoder model search_agent winline; do
    cp "ptcg/rl/$m.py" "$out/ptcg/rl/"
  done
  find "$out" -name __pycache__ -type d -prune -exec rm -rf {} +
  sed -i.bak "s/^HEADS = .*/HEADS = $HEADS/" "$out/main.py" && rm -f "$out/main.py.bak"
  grep -q "^HEADS = $HEADS\$" "$out/main.py" || { echo "HEADS stamp failed"; exit 1; }
  echo "  $out  deck=$2  heads=$HEADS"
}

echo "[make_opponents] $TAG from $CKPT  (pool $POOL, heads $HEADS)"
DIRS=""
for pair in "${PAIRS[@]}"; do
  build "${pair%%=*}" "${pair#*=}"
  DIRS="${DIRS:+$DIRS,}opponents/${TAG}_${pair%%=*}"
done
echo "[make_opponents] add to eval with:"
echo "  --bench-opponents $DIRS"
