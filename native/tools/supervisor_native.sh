#!/bin/bash
# Linux only (flock, pgrep, /proc).
# Supervisor for the NATIVE trainer (train_native.py). Committed on purpose:
# a supervisor that lives only on the training host dies with it.
#
#   cd <repo root>
#   setsid nohup bash native/tools/supervisor_native.sh <RUN> \
#       > runs/native_supervisor.log 2>&1 < /dev/null &
#
# Launch flags live in runs/native_<RUN>.args -- everything after
# `python native/train_native.py`, newlines/comments ok. Per-host settings in
# runs/native_<RUN>.env (TRAIN_GPUS, EVAL_GPU, PY, STALE_MIN, ...), sourced
# from a file so an automatic restart after a reboot gets identical values.
#
# Behaviour:
#   * relaunches the trainer on death; resume is BUILT IN (train_native picks
#     up experiments/native-<RUN>/state_latest.bin: weights + muon momentum +
#     step/epoch/LR-anneal position)
#   * kills + relaunches on a stale log (the CUDA-fault hang: processes alive
#     in S state, GPU idle holding VRAM)
#   * crashloop backoff: 5 rapid deaths => 30 min pause. kill -9 storms on
#     CUDA procs can wedge a whole container's driver; never hammer
#     relaunches into a faulting GPU
#   * keeps native_eval_watch.sh (converter + torch eval on EVAL_GPU) alive
#   * optionally pushes state snapshots, deck matrices and converted models
#     to a Hugging Face dataset repo (HF_REPO in the .env; empty = off)
#
# To STOP: kill this first (match "supervisor_native.sh" WITHOUT the run name
# so your own ssh command line doesn't match), then the trainer. From a
# SEPARATE ssh session than any launch -- pkill -f eats sessions whose command
# line mentions the pattern.
set -u
cd "$(dirname "$0")/../.." || exit 1

RUN="${1:?usage: supervisor_native.sh <RUN>}"
# export everything from the .env so child launches (eval script) see it too
[ -f "runs/native_$RUN.env" ] && { set -a; . "runs/native_$RUN.env"; set +a; }

PY="${PY:-python3}"
ARGS_FILE="runs/native_$RUN.args"
LOG="runs/native_$RUN.log"
STALE_MIN="${STALE_MIN:-15}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"        # CVD for the trainer ("0,1" for --gpus 2)
EVAL_GPU="${EVAL_GPU:-1}"
HF_REPO="${HF_REPO:-}"                # empty = no Hugging Face uploads
HF_STATE_EVERY_S="${HF_STATE_EVERY_S:-1800}"
# Per-host HF path prefix — two hosts running the same RUN name must not
# clobber each other's artifacts. Set HF_PREFIX in the .env.
HF_PREFIX="${HF_PREFIX:-train-native_$RUN}"

[ -f "$ARGS_FILE" ] || { echo "missing $ARGS_FILE" >&2; exit 1; }
mkdir -p runs experiments
TRAIN_ARGS=$(grep -vE '^\s*(#|$)' "$ARGS_FILE" | tr '\n' ' ')

log() { echo "$(date -u +%FT%T) $*" >> runs/native_supervisor.log; }

proc_running() {          # $1 = pgrep -f pattern, $2 = expected comm prefix
    local p
    for p in $(pgrep -f "$1" 2>/dev/null); do
        case "$(cat "/proc/$p/comm" 2>/dev/null)" in "$2"*) return 0 ;; esac
    done
    return 1
}

# Args from the .args file sit between the script name and --run-name in the
# real cmdline — the pattern must bridge them or proc_running never matches
# and the supervisor relaunches a healthy trainer.
TRAIN_PAT="[t]rain_native.py .*--run-name $RUN( |\$)"

kill_train() {
    pkill -f "$TRAIN_PAT"; sleep 15
    pkill -9 -f "$TRAIN_PAT"; sleep 15
    # DDP rank-1+ workers are mp.spawn children: their cmdline is
    # "python -c from multiprocessing.spawn import spawn_main ..." — it does
    # NOT contain train_native.py, and they survive their parent holding
    # ~15GB VRAM each. Nothing else here uses mp.spawn (eval shards are plain
    # subprocesses of the eval script).
    pkill -9 -f "[m]ultiprocessing.spawn import spawn_main"
}

LAUNCHES_FILE="runs/native_$RUN.launches"
launch_train() {
    log "launch train (gpus=$TRAIN_GPUS): $TRAIN_ARGS"
    date +%s >> "$LAUNCHES_FILE"
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
    PTCG_TABLES="${PTCG_TABLES:-native/ptcg_tables.bin}" \
    PYTHONPATH=data:.:native PYTHONFAULTHANDLER=1 \
    setsid nohup $PY native/train_native.py $TRAIN_ARGS --run-name "$RUN" \
        >> "$LOG" 2>&1 < /dev/null &
}

crashloop_backoff() {
    # 5 launches within the last 25 min => something is broken beyond a
    # relaunch; back off so we don't kill-storm the driver.
    local now n
    now=$(date +%s)
    n=$(awk -v t=$((now - 1500)) '$1 > t' "$LAUNCHES_FILE" 2>/dev/null | wc -l)
    if [ "$n" -ge 5 ]; then
        log "CRASHLOOP ($n launches in 25min) -> backing off 30 min"
        sleep 1800
    fi
}

# One pass of the HF pushes (state_latest if due, rotated states once each,
# converted models once each). Shared by the steady-state loop and the
# divergence-stop path so a stopped run still lands its last artifacts.
push_hf() {
    [ -n "$HF_REPO" ] || return 0
    ST="experiments/native-$RUN/state_latest.bin"
    NOW=$(date +%s)
    if [ -f "$ST" ] && [ $((NOW - LAST_HF_STATE)) -ge "$HF_STATE_EVERY_S" ]; then
        $PY -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$ST', path_in_repo='checkpoints/$HF_PREFIX/state_latest.bin', repo_id='$HF_REPO', repo_type='dataset')" \
            >> runs/native_supervisor.log 2>&1 \
            && { LAST_HF_STATE=$NOW; log "state_latest.bin -> HF ($HF_PREFIX)"; }
    fi

    # Push each ROTATED optimizer state once, keyed by step. state_latest.bin
    # above is a moving target -- by the time a divergence is noticed both the
    # disk and HF copies can already hold the post-collapse state, leaving only
    # a weights-only restart and losing the Muon momentum.
    # These are immutable per-step snapshots (trainer keeps the last 10 on
    # disk; HF keeps every one), so there is always a pre-collapse state to
    # resume from. ~15MB each.
    SPUSHED="runs/native_$RUN.hf_states_pushed"
    touch "$SPUSHED"
    for s in experiments/native-$RUN/state_0*.bin; do
        [ -e "$s" ] || continue
        sb=$(basename "$s")
        grep -qx "$sb" "$SPUSHED" && continue
        if $PY -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$s', path_in_repo='checkpoints/$HF_PREFIX/states/$sb', repo_id='$HF_REPO', repo_type='dataset')" \
                >> runs/native_supervisor.log 2>&1; then
            echo "$sb" >> "$SPUSHED"
            log "$sb -> HF ($HF_PREFIX/states)"
        fi
    done

    # Push each deck-vs-deck matrix once. These are the ONLY record of which
    # deck beat whom -- the trainer resets each interval after writing, and the
    # files die with the host, so they must leave the machine. ~330KB each,
    # one per rank per flush interval. Read with tools/deck_matrix.py.
    DMPUSHED="runs/native_$RUN.hf_deckmat_pushed"
    touch "$DMPUSHED"
    for d in experiments/native-$RUN/deckmat/*.bin; do
        [ -e "$d" ] || continue
        db=$(basename "$d")
        grep -qx "$db" "$DMPUSHED" && continue
        if $PY -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$d', path_in_repo='checkpoints/$HF_PREFIX/deckmat/$db', repo_id='$HF_REPO', repo_type='dataset')" \
                >> runs/native_supervisor.log 2>&1; then
            echo "$db" >> "$DMPUSHED"
            log "$db -> HF ($HF_PREFIX/deckmat)"
        fi
    done

    # Push each converted model checkpoint once (also backfills anything
    # already on disk the first time this runs with a valid token).
    PUSHED="runs/native_$RUN.hf_pushed"
    touch "$PUSHED"
    for m in experiments/train-native_$RUN/model_*.pt; do
        [ -e "$m" ] || continue
        b=$(basename "$m")
        grep -qx "$b" "$PUSHED" && continue
        if $PY -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$m', path_in_repo='checkpoints/$HF_PREFIX/$b', repo_id='$HF_REPO', repo_type='dataset')" \
                >> runs/native_supervisor.log 2>&1; then
            echo "$b" >> "$PUSHED"
            log "$b -> HF ($HF_PREFIX)"
        fi
    done
}

log "supervisor up for native_$RUN"

LAST_HF_STATE=0
while true; do
    if ! proc_running "$TRAIN_PAT" python; then
        if [ -f "experiments/native-$RUN/final.bin" ]; then
            log "final.bin present -> run complete, supervisor exiting"
            exit 0
        fi
        # Divergence handling (docs/training.md): the trainer
        # prints FATAL DIVERGENCE and SIGKILLs its own process group when KL
        # blows past the abort threshold or losses go non-finite. Two causes
        # look identical at this point:
        #   * a corrupt START (a flaky launch: insane KL from epoch 1) --
        #     the right response is to relaunch;
        #   * true late-run divergence -- relaunching resumes from a possibly
        #     poisoned state_latest.bin, aborts again within minutes, and
        #     would crashloop the host.
        # The 3-strikes rule separates them: a bad start relaunches clean; a
        # real divergence racks up 3 markers within the hour and hard-stops.
        # The log is rotated either way so each abort keeps its evidence and
        # the marker can never re-trip a later pass.
        if tail -n 400 "$LOG" 2>/dev/null | grep -q "FATAL DIVERGENCE"; then
            kill_train      # sweep any surviving rank workers
            DIVFILE="runs/native_$RUN.divergence_times"
            date +%s >> "$DIVFILE"
            NDIV=$(awk -v t=$(($(date +%s) - 3600)) '$1 > t' "$DIVFILE" | wc -l)
            mv "$LOG" "$LOG.divergence.$(date -u +%Y%m%dT%H%M%S)"
            if [ "$NDIV" -ge 3 ]; then
                log "DIVERGENCE STOP ($NDIV aborts in 60min) -- pushing artifacts, NOT relaunching"
                push_hf
                log "supervisor exiting after divergence stop (eval_watch left up to drain conversions)"
                exit 0
            fi
            log "DIVERGENCE abort $NDIV/3 in 60min -- treating as bad start, relaunching"
        fi
        crashloop_backoff
        launch_train
        sleep 240          # past import + create_pufferl before judging
        continue
    fi

    if ! find "$LOG" -mmin -"$STALE_MIN" 2>/dev/null | grep -q .; then
        log "HANG detected (log stale ${STALE_MIN}m) -> kill"
        kill_train
        continue
    fi

    # NaN backstop independent of the trainer's own abort -- covers a
    # trainer binary older than the guard. Epoch lines read "ent nan kl nan"
    # when losses go non-finite. Append the marker ourselves so the single
    # divergence-stop path above handles push/rotate/exit next iteration.
    if tail -n 30 "$LOG" 2>/dev/null | grep -qE "^epoch [0-9]+ .*(ent|kl) (nan|-nan|inf)"; then
        log "NAN in epoch lines -> kill + divergence stop"
        kill_train
        echo "FATAL DIVERGENCE: supervisor nan backstop" >> "$LOG"
        continue
    fi

    if ! ps -eo args= | grep -q "^bash native/tools/native_eval_watch.sh $RUN"; then
        log "launch native_eval_watch (gpu $EVAL_GPU)"
        setsid nohup bash native/tools/native_eval_watch.sh "$RUN" "$EVAL_GPU" \
            >> runs/native_eval_watch.log 2>&1 < /dev/null &
    fi

    push_hf

    sleep 60
done
