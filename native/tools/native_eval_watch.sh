#!/bin/bash
# Linux only (flock, pgrep, /proc).
# Eval for a native run: converts the trainer's flat fp32 weight blobs
# (experiments/native-<RUN>/<step16>.bin) into torch checkpoints
# (experiments/train-native_<RUN>/model_<epoch6>.pt) and runs the stock torch
# eval_watch over them on a separate GPU. Torch is fine here — only the
# training loop is native.
#
#   bash native/tools/native_eval_watch.sh <RUN> [GPU]
#
# STEPS_PER_EPOCH must match the trainer (total_agents * horizon); epoch
# numbering and eval's x-axis both derive from it.
set -u
cd "$(dirname "$0")/../.." || exit 1
RUN=${1:?usage: native_eval_watch.sh <RUN> [GPU]}
GPU=${2:-1}
PY="${PY:-python3}"
# STEPS_PER_EPOCH: eval x-axis scale (AGGREGATE steps per epoch).
# RANK_STEPS_PER_EPOCH: divisor for naming epochs from the trainer's
# step-stamped .bin files, which count PER-RANK steps — under DDP these
# differ by the world size (65536/rank vs 131072 aggregate for 2 GPUs).
# Single-instance lock. Stacking watchers is easy to do by accident (a retry
# loop around a backgrounded launch will happily fire twice) and expensive:
# each instance runs its own converter AND its own full bench fan-out, so two
# instances double the GPU load on the training ranks and log duplicate evals
# to the same wandb run. fd 9 stays open for the life of the script.
mkdir -p runs
exec 9>"runs/native_eval_watch_$RUN.lock"
if ! flock -n 9; then
    echo "native_eval_watch: an instance for '$RUN' already holds the lock;" \
         "refusing to stack. Running now:" >&2
    pgrep -fa "native_eval_watch.sh $RUN" | grep -v $$ >&2
    exit 1
fi
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-65536}"
RANK_STEPS_PER_EPOCH="${RANK_STEPS_PER_EPOCH:-$STEPS_PER_EPOCH}"
EVAL_THREADS="${EVAL_THREADS:-2}"
# Attention heads for the eval model. A hardcoded `--heads 4` below would be
# WRONG the moment the trunk size changes, and wrong SILENTLY:
# the qkv projections are d x d whatever the head count, so a mismatched value
# loads without error and computes different attention — the battery would then
# be measuring a model the trainer never trained. Read it from native_weights,
# the same single source the converter and the parity gate use.
EVAL_HEADS="${EVAL_HEADS:-$($PY -c 'import sys; sys.path.insert(0, "native/tools"); import native_weights; print(native_weights.HEADS)' 2>/dev/null || echo 4)}"
# EVAL_BENCH=0 runs the CONVERTER ONLY and skips the battery — for when the
# battery is being run somewhere else (it costs ~28% of trainer SPS, because
# the bench workers are CPU-side). Conversion stays
# here because it is cheap and whoever runs the battery needs the .pt files.
EVAL_BENCH="${EVAL_BENCH:-1}"
HF_REPO="${HF_REPO:-}"
SRC="experiments/native-$RUN"
DST="experiments/train-native_$RUN"
mkdir -p "$DST"

# Converter loop runs alongside eval_watch and dies with this script (no exec:
# the EXIT trap must survive so a relaunch never stacks a second converter).
trap 'kill $(jobs -p) 2>/dev/null' EXIT
(
    while true; do
        for f in "$SRC"/0*.bin; do
            [ -e "$f" ] || continue
            step=$((10#$(basename "$f" .bin)))
            epoch=$((step / RANK_STEPS_PER_EPOCH))
            # EVAL_EPOCH_MOD thins the eval cadence without touching the
            # trainer: only epochs divisible by it are converted, and eval_watch
            # only ever sees converted .pt files. Set it to a multiple of the
            # checkpoint interval (2x that interval = half the eval rate).
            # Skipped .bin blobs stay on disk and on HF — convert them later
            # with native/tools/native_weights.py if a gap needs filling.
            if [ "${EVAL_EPOCH_MOD:-0}" -gt 0 ] \
               && [ $((epoch % EVAL_EPOCH_MOD)) -ne 0 ]; then
                continue
            fi
            out=$(printf '%s/model_%06d.pt' "$DST" "$epoch")
            [ -f "$out" ] && continue
            # write via tmp+rename so eval_watch never reads a half-written .pt
            PYTHONPATH=data:.:native $PY native/tools/native_weights.py \
                import "$f" "$out.tmp" >> runs/native_convert.log 2>&1 \
                && mv "$out.tmp" "$out" \
                || echo "$(date -u +%FT%T) convert FAILED for $f" >> runs/native_convert.log
        done
        sleep 60
    done
) &

if [ "$EVAL_BENCH" = "0" ]; then
    echo "native_eval_watch: EVAL_BENCH=0 — converter only, no battery." \
         "Converted .pt land in $DST for whoever is running the battery." >&2
    wait          # keep the converter subshell (and the flock) alive
    exit 0
fi

env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=$EVAL_THREADS \
    MKL_NUM_THREADS=$EVAL_THREADS PYTHONPATH="$(pwd)/data:$(pwd)" \
    $PY -m ptcg.rl.eval_watch --wandb --run-name "native_$RUN" \
    --ckpt-dir "$DST" \
    ${HF_REPO:+--hf-repo "$HF_REPO"} \
    ${EVAL_WANDB_NAME:+--wandb-name "$EVAL_WANDB_NAME"} \
    ${EVAL_BENCH_GPUS:+--bench-gpus "$EVAL_BENCH_GPUS"} \
    --heads "$EVAL_HEADS" --steps-per-epoch "$STEPS_PER_EPOCH" --device cuda \
    ${EVAL_EXTRA:-}
