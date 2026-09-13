#!/bin/bash
# Arena-backed battery: one checkpoint plays every candidate deck against every
# opponent bundle, policy-only, seats alternating, as parallel ptcg.rl.arena
# processes. cg.game is a per-process singleton, so one process holds one game
# at a time and parallelism is processes, not threads.
#
#   battery/run_battery.sh <model.pt> <heads> <outdir> [options]
#     --decks a.csv,b.csv   candidate decks      (default: battery/candidate_decks/*.csv)
#     --opponents d1,d2     opponent bundle dirs (default: opponents/e4235_*)
#     --games N             games per chunk      (default 100)
#     --chunks K            seed-independent chunks per cell, seeds k*7919 (default 2)
#     --jobs J              concurrent arena processes (default: nproc)
#     --device cpu|cuda     forward device       (default cpu)
#     --temp T              learner temperature  (default 0 = greedy)
#
# Every cell writes <outdir>/<deck>--vs--<opponent>--c<k>.jsonl (one game per
# line, a {"summary": ...} record last) plus a .log; battery/aggregate.py then
# prints the per-cell, per-opponent, per-deck and macro win rates. The
# opponents' own temperature is whatever their bundle ships (agent/main.py is
# greedy), so with --temp 0 both sides are greedy: the configuration a
# submission actually plays.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:?usage: run_battery.sh <model.pt> <heads> <outdir> [--decks ..] [--opponents ..] [--games N] [--chunks K] [--jobs J] [--device D] [--temp T]}"
HEADS="${2:?heads}"
OUT="${3:?outdir}"
shift 3
DECKS=""; OPPS=""; GAMES=100; CHUNKS=2; JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"; DEVICE=cpu; TEMP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --decks) DECKS="$2"; shift 2;;
    --opponents) OPPS="$2"; shift 2;;
    --games) GAMES="$2"; shift 2;;
    --chunks) CHUNKS="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --temp) TEMP="$2"; shift 2;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
[ -n "$DECKS" ] || DECKS=$(ls battery/candidate_decks/*.csv | paste -sd, -)
[ -n "$OPPS" ] || OPPS=$(ls -d opponents/e4235_* 2>/dev/null | paste -sd, - || true)
[ -n "$OPPS" ] || { echo "no opponent bundles: pass --opponents or build opponents/e4235_* (battery/README.md)"; exit 1; }
[ -f "$CKPT" ] || { echo "no checkpoint $CKPT"; exit 1; }
PY="${PY:-python3}"
mkdir -p "$OUT"
n=0
for deck in ${DECKS//,/ }; do
  [ -f "$deck" ] || { echo "no deck $deck"; exit 1; }
  dname=$(basename "$deck" .csv)
  for opp in ${OPPS//,/ }; do
    [ -f "$opp/main.py" ] || { echo "no bundle $opp"; exit 1; }
    oname=$(basename "$opp")
    for ((c = 0; c < CHUNKS; c++)); do
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
      tag="${dname}--vs--${oname}--c${c}"
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="data:." \
      $PY -m ptcg.rl.arena --ckpt "$CKPT" --deck "$deck" --opponent "$opp" \
          --games "$GAMES" --seed $((c * 7919)) --heads "$HEADS" --device "$DEVICE" \
          --temp "$TEMP" --tag "$tag" --out "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1 &
      n=$((n + 1))
    done
  done
done
echo "[run_battery] $n arena processes, $GAMES games each, up to $JOBS at a time -> $OUT"
wait
fails=$(grep -l -E "Traceback|Error" "$OUT"/*.log 2>/dev/null | wc -l)
[ "$fails" -eq 0 ] || { echo "[run_battery] $fails process logs contain errors (see $OUT/*.log)"; }
$PY battery/aggregate.py "$OUT"
