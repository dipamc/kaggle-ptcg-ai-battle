"""Native-backend training driver (no torch in the loop; torch used once at
launch to generate/convert weights). Mirrors pufferl.py's train() shape.

Run from repo root:
  source /venv/main/bin/activate
  PYTHONPATH=data:.:native python native/train_native.py --timesteps 20000000
Multi-GPU:
  ... train_native.py --gpus 2 --timesteps 40000000
"""
import argparse
import glob
import math
import multiprocessing as mp
import os
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "native"))

# Default anchor decks: labelled in the per-deck win-rate telemetry and
# usable as the oversample set (all p_anchor_* default to 0 = uniform draw).
ANCHOR_DECKS = ",".join([
    "marnie_s_grimmsnarl_ex_spikemuth_gym__04",
    "alakazam_enhanced_hammer__03",
    "dragapult_ex_crushing_hammer__05",
    "cynthia_s_garchomp_ex_cynthia_s_roserade_cynth__01",
    "rillaboom_dipplin_festival_grounds__03",
    "crustle_jumbo_ice_cream__01",
])


def blob_deck_table(path):
    """Read the (n_decks, 60) deck table out of ptcg_tables.bin."""
    with open(path, "rb") as f:
        assert f.read(8) == b"PTCGTAB1", f"{path}: bad magic"
        nsec = struct.unpack("<15i", f.read(60))[14]
        for _ in range(nsec):
            name = f.read(24).rstrip(b"\0").decode()
            dt, nd = struct.unpack("<2i", f.read(8))
            shape = struct.unpack("<4q", f.read(32))[:nd]
            esz = {0: 4, 1: 4, 2: 1, 3: 1}[dt]
            n = 1
            for s in shape:
                n *= s
            if name == "decks":
                rows = struct.unpack(f"<{n}i", f.read(n * esz))
                return [list(rows[i * 60:(i + 1) * 60]) for i in range(shape[0])]
            f.seek(n * esz, 1)
    raise AssertionError(f"{path}: no decks section")


def blob_pool_shape(path):
    """(n_decks, n_decks_alt) from the blob, without loading either table."""
    main = alt = 0
    with open(path, "rb") as f:
        assert f.read(8) == b"PTCGTAB1", f"{path}: bad magic"
        nsec = struct.unpack("<15i", f.read(60))[14]
        for _ in range(nsec):
            name = f.read(24).rstrip(b"\0").decode()
            dt, nd = struct.unpack("<2i", f.read(8))
            shape = struct.unpack("<4q", f.read(32))[:nd]
            esz = {0: 4, 1: 4, 2: 1, 3: 1}[dt]
            n = 1
            for sdim in shape:
                n *= sdim
            if name == "decks":
                main = shape[0]
            elif name == "decks_alt":
                alt = shape[0]
            f.seek(n * esz, 1)
    return main, alt


def log_pool_composition(path):
    """State the deck-pool split in the run log, every run.

    A COVERAGE POOL WIDENS WHO THE RANDOM OPPONENT CAN BE: with one loaded,
    that seat draws from the training pool PLUS the coverage decks, while
    every other seat stays on the training pool. That is invisible from the
    args alone (the split lives in the blob), so it gets printed rather than
    left to be inferred.
    """
    main, alt = blob_pool_shape(path)
    if alt:
        print(f"deck pool: {main} training + {alt} coverage. The random "
              f"opponent's seat draws from all {main + alt} (anchors "
              f"included); every other seat draws only the {main} training "
              f"decks. Games vs the random opponent are EXCLUDED from the "
              f"deck matrix, so it stays {main}x{main} and no coverage deck "
              f"appears in it.", flush=True)
    else:
        print(f"deck pool: {main} training decks, no coverage pool — every "
              f"seat including the random opponent draws from the same "
              f"{main}. Games vs the random opponent are excluded from the "
              f"deck matrix.", flush=True)


def _deck_metric_name(key, deck_short):
    """deck_wr_3 -> deck_wr_fezandipiti_ex, using the --anchor-decks order.

    Falls back to the raw key when the index is out of range, so a metric can
    never be dropped or mislabelled just because the anchor list changed."""
    for prefix in ("deck_wr_", "deck_n_"):
        if key.startswith(prefix):
            try:
                i = int(key[len(prefix):])
            except ValueError:
                return key
            if 0 <= i < len(deck_short):
                return prefix + deck_short[i]
            return key
    return key


def pool_deck_names(pool):
    """(names, rows) in DECK ID ORDER: the sorted-glob 60-line filter of the
    pool dir, which is export_tables.py's construction of the deck table.

    Deck id = position in this list. Everything that maps a name to an id --
    anchors, sampling weights -- must go through this one function, or two
    slightly different globs eventually disagree and mislabel every deck after
    the first difference."""
    names, rows = [], []
    for fn in sorted(glob.glob(os.path.join(pool, "*.csv"))):
        try:
            ids = [int(line) for line in open(fn) if line.strip()]
        except ValueError:
            continue          # non-deck file (e.g. a manifest .csv)
        if len(ids) == 60:
            names.append(os.path.basename(fn)[:-len(".csv")])
            rows.append(ids)
    return names, rows


def read_weight_request(path):
    """Deck-weights request file -> (version, default, {name: weight}).

    Never raises. A half-written or malformed request must be SKIPPED, not
    take an 8-rank run down: version 0 means "no usable request", which the
    caller treats as nothing to do. Format:

        version 3
        default 1.0
        <deck name> <weight>          # '#' comments and blank lines ignored
    """
    if not path or not os.path.exists(path):
        return 0, 1.0, {}
    try:
        ver, dflt, per = 0, 1.0, {}
        for ln in open(path):
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            key, val = ln.split()
            if key == "version":
                ver = int(val)
            elif key == "default":
                dflt = float(val)
            else:
                per[key] = float(val)
        return ver, dflt, per
    except (OSError, ValueError):
        return 0, 1.0, {}


def weight_vector(names, dflt, per):
    """(weights in deck-id order, unknown names). A single unknown name fails
    the WHOLE request: it usually means the file was written against a
    different pool, and applying the rest would silently weight the wrong
    decks."""
    idx = {n: i for i, n in enumerate(names)}
    unknown = sorted(n for n in per if n not in idx)
    if unknown:
        return None, unknown
    w = [float(dflt)] * len(names)
    for n, x in per.items():
        w[idx[n]] = float(x)
    return w, []


def resolve_anchor_indices(names, tables_path, pool="decks/pool"):
    """Anchor deck names -> deck-table indices, verified against the ACTUAL
    blob the env will load. Deck ids are positions in the sorted-glob 60-line
    filter of decks/pool (export_tables.py's construction); a re-exported or
    stale blob would silently shift indices, so the pool glob is checked
    row-by-row against the blob before any index is trusted."""
    pool_names, pool_rows = pool_deck_names(pool)
    blob_rows = blob_deck_table(tables_path)
    assert pool_rows == blob_rows, (
        f"deck table mismatch: {pool} glob ({len(pool_rows)} decks) != "
        f"{tables_path} ({len(blob_rows)} decks) — re-run export_tables.py "
        "or fix the pool; anchor indices would be wrong")
    idx = []
    for n in names:
        assert n in pool_names, f"anchor deck not in pool: {n}"
        idx.append(pool_names.index(n))
    print(f"anchor decks resolved: "
          f"{dict(zip(names, idx))} (pool {len(pool_names)} decks)", flush=True)
    return idx

# Keep the last N optimizer states (weights + Muon momentum + step/epoch), not
# just state_latest.bin. A divergence between two checkpoints can overwrite
# both the on-disk and the uploaded state_latest.bin with the post-collapse
# state before anyone notices, leaving only a weights-only blob to restart
# from and losing the optimizer state entirely.
STATE_HISTORY = 10


def snapshot_state(run_dir, state_path, step):
    """Hard-link state_latest.bin to state_<step16>.bin and prune to the last
    STATE_HISTORY snapshots.

    save_state writes tmp + fsync + rename, so the next save REPLACES the
    directory entry rather than rewriting the file -- which makes a hard link a
    zero-copy point-in-time snapshot (15MB each, ~150MB for the full history).
    Never fatal: a failed snapshot must not kill a training run.
    """
    snap = os.path.join(run_dir, f"state_{step:016d}.bin")
    try:
        if os.path.exists(snap):
            os.remove(snap)
        os.link(state_path, snap)
    except OSError as e:
        print(f"state snapshot failed for step {step}: {e!r}", flush=True)
        return
    # state_latest.bin does not match state_0*.bin, so it is never pruned
    old = sorted(glob.glob(os.path.join(run_dir, "state_0*.bin")))
    for p in old[:-STATE_HISTORY]:
        try:
            os.remove(p)
        except OSError:
            pass


def build_args(a, rank, world, gpu_id, nccl_id):
    return {
        "env_name": "ptcg",
        "reset_state": True,
        "cudagraphs": a.cudagraphs,
        "profile": False,
        "rank": rank,
        "world_size": world,
        "gpu_id": gpu_id,
        "nccl_id": nccl_id,
        "seed": a.seed + rank,
        "train": {
            "horizon": 64,
            "total_timesteps": a.timesteps // world,
            "learning_rate": a.lr,
            "min_lr_ratio": a.min_lr_ratio,
            "anneal_lr": 0 if a.no_anneal_lr else 1,
            "lr_warmup_epochs": a.lr_warmup,
            "start_epoch": a.start_step // (a.total_agents * 64),
            "lr_anneal_from_epoch": a.lr_anneal_from_step // (a.total_agents * 64),
            "beta1": 0.95, "beta2": 0.999, "eps": 1e-12,
            "minibatch_size": a.minibatch,
            "accum_minibatches": getattr(a, "accum_minibatches", 1),
            "replay_ratio": 1.0,
            "max_grad_norm": 1.5,
            "clip_coef": 0.2, "vf_clip_coef": 0.2,
            "vf_coef": a.vf_coef, "ent_coef": a.ent_coef,
            "min_ent_coef_ratio": 0.1, "anneal_ent_coef": 0,
            "gamma": a.gamma, "gae_lambda": a.gae_lambda,
            "vtrace_rho_clip": 1.0, "vtrace_c_clip": 1.0,
            "prio_alpha": 0.8, "prio_beta0": a.prio_beta0,
            # KL/value guard + Muon 1-D weight decay (docs/training.md)
            "kl_skip_threshold": a.kl_skip,
            "vf_skip_threshold": a.vf_skip,
            "muon_wd_1d": a.muon_wd_1d,
            "league_routing": 1 if a.league_bins else 0,
            "gpus": world,
        },
        "vec": {
            "total_agents": a.total_agents,
            "num_buffers": a.num_buffers,
            "num_threads": a.num_threads,
            "num_frozen_banks": len(a.league_bins),
            "frozen_bank_pct": a.league_pct,
        },
        "env": {
            "mix_self": a.mix_self,
            "reward_win": 1,
            "max_engine_steps": 3000,
            "seed": a.seed + 1000 * rank,
            "n_anchors": len(a.anchor_idx),
            **{f"anchor_{i}": v for i, v in enumerate(a.anchor_idx)},
            "p_anchor_learner": a.p_anchor_learner,
            "p_anchor_self_opp": a.p_anchor_self_opp,
            "p_anchor_league_opp": a.p_anchor_league_opp,
        },
        "policy": {"hidden_size": 65, "num_layers": 1},
    }


def _episodes_written(a):
    """Episodes on disk, measured off the deliverable itself.

    Counted in COMPLETED event-log files: a completed file is exactly
    PTCG_EVENT_LOG_EVERY episodes (the writer rotates on that boundary with
    tmp+rename). Returns -1 when there is no event log to count.
    """
    prefix = os.environ.get("PTCG_EVENT_LOG")
    if not prefix:
        return -1
    every = int(os.environ.get("PTCG_EVENT_LOG_EVERY", "20000") or 20000)
    return len(glob.glob(f"{prefix}_e*_*.bin")) * every


def _eta_str(done, target, rate):
    """'12345/1000000 eps  38.2 eps/s  ETA 7h04m (done 18:41Z)'"""
    if target <= 0:
        return f"{done} eps"
    pct = 100.0 * done / target
    if rate <= 0:
        return f"{done}/{target} eps ({pct:.1f}%)  ETA --"
    left = max(0, target - done) / rate
    h, m = int(left // 3600), int((left % 3600) // 60)
    eta = time.strftime("%H:%MZ", time.gmtime(time.time() + left))
    return (f"{done}/{target} eps ({pct:.1f}%)  {rate:.1f} eps/s  "
            f"ETA {h}h{m:02d}m (done {eta})")


def worker(a, rank, world, gpu_id, nccl_id, run_dir):
    # Pin the rank to one physical GPU so every thread (incl. the per-buffer
    # rollout pthreads, which never call cudaSetDevice) defaults to it.
    # Respect an outer CUDA_VISIBLE_DEVICES: remap THROUGH it, don't clobber
    # it (a bare str(gpu_id) sent every "CVD=1" single-GPU run to physical 0).
    outer = os.environ.get("CUDA_VISIBLE_DEVICES")
    if outer:
        vis = outer.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = vis[gpu_id % len(vis)]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PTCG_TABLES", "native/ptcg_tables.bin")
    # OMP workers sleeping at barriers instead of spinning matters on
    # cgroup-quota'd boxes (see ptcg_wait_relax in vecenv.h for the full story).
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ["PTCG_INIT_WEIGHTS"] = os.path.join(run_dir, "init.bin")
    # Full deck-vs-deck result matrix (offline analysis). The C env reads these
    # lazily on the first finished episode, so they must be set before the env
    # is constructed. One prefix PER RANK: every rank is its own process and
    # they would otherwise write the same filenames on top of each other.
    if getattr(a, "deck_matrix_every", 0) > 0:
        dm_dir = os.path.join(run_dir, "deckmat")
        os.makedirs(dm_dir, exist_ok=True)
        os.environ["PTCG_DECK_MATRIX"] = os.path.join(dm_dir, f"r{rank}")
        os.environ["PTCG_DECK_MATRIX_EVERY"] = str(a.deck_matrix_every)
    from puffer_ptcg import _C
    args = build_args(a, rank, world, 0, nccl_id)
    pl = _C.create_pufferl(args)
    steps_per_epoch = a.total_agents * 64
    total = args["train"]["total_timesteps"]
    state_path = os.path.join(run_dir, "state_latest.bin")
    if not a.no_resume and os.path.exists(state_path):
        # Every rank loads the same state (rank0 wrote it; DDP keeps all ranks'
        # weights+momentum identical, so one file resumes them all). epoch and
        # the LR/beta anneal positions come back with it.
        _C.load_state(pl, state_path)
        print(f"rank {rank}: resumed {state_path} at step {pl.global_step}",
              flush=True)
    elif a.start_step > 0:
        # Weights-only restart: stamp the source checkpoint's per-rank step
        # so telemetry, checkpoint names, eval x-axes and the epoch-driven
        # schedules continue from the parent run instead of restarting at 0.
        _C.set_start(pl, a.start_step)
        print(f"rank {rank}: weights-only start stamped at step "
              f"{pl.global_step} (epoch {pl.global_step // steps_per_epoch})",
              flush=True)
    # League banks were created (and their pointers baked into the cudagraphs)
    # inside create_pufferl; weights load in-place, so this must run on EVERY
    # start — resumed or fresh — and on every rank.
    for bi, bp in enumerate(a.league_bins):
        _C.load_frozen_bank(pl, bi, bp)
    if a.league_bins:
        print(f"rank {rank}: loaded {len(a.league_bins)} league banks "
              f"@ {a.league_pct:.4f} rows each", flush=True)
    epoch = pl.global_step // steps_per_epoch
    if rank == 0:
        log_pool_composition(os.environ.get("PTCG_TABLES",
                                            "native/ptcg_tables.bin"))
    # total_timesteps is an ABSOLUTE global_step target, so on a resumed
    # checkpoint it must exceed the step we resumed AT -- otherwise the loop
    # never runs and the process exits looking like a silent success. In
    # rollout-only the episode target is the real stop condition, so derive a
    # generous step ceiling from it rather than making the caller do the sum.
    if a.rollout_only and a.episodes:
        total = max(total, pl.global_step + a.episodes * 400)
    elif pl.global_step >= total:
        print(f"WARNING: resumed at step {pl.global_step} but --timesteps is "
              f"{total} — nothing to do. --timesteps is absolute, not a "
              f"delta.", flush=True)
    roll_start_step = pl.global_step
    roll_t0 = time.time()
    t_start = time.time()   # post-create: wall deltas isolate train loop time
    last_log = time.time()
    log_reads = 0           # divergence-guard grace: ignore the first reads
    kl_bad_streak = 0
    vf_bad_streak = 0
    wandb = None
    if a.wandb and rank == 0:
        import wandb as _w
        cfg = {k: v for k, v in args.items() if k != "nccl_id"}
        # wandb must never kill the run: retry init, then fall back to
        # training without telemetry (a deleted-run id can leave the backend
        # timing out for a while).
        wid = a.wandb_name or f"native-{a.run_name}"
        for attempt in range(3):
            try:
                _w.init(id=wid, name=wid,
                        resume="allow", project="ptcg", group="native",
                        config={**cfg, "argv": sys.argv},
                        settings=_w.Settings(init_timeout=180))
                wandb = _w
                break
            except Exception as e:
                print(f"wandb.init attempt {attempt + 1} failed: {e}",
                      flush=True)
                time.sleep(30)
        else:
            print("wandb disabled for this run (init kept failing)",
                  flush=True)
    # Short names for the per-deck training win rates, in --anchor-decks order.
    # getattr keeps older arg objects (resumes pickled before this existed)
    # working — an empty list just leaves the raw deck_wr_<i> keys alone.
    _dshort = getattr(a, "deck_short", [])
    # Hand the current epoch to the C env so each deck matrix can stamp itself
    # (the env has no epoch concept). tmp+rename so a flush racing this read
    # never sees a half-written number.
    _dm_epoch_file = (os.environ.get("PTCG_DECK_MATRIX", "") + ".epoch") \
        if os.environ.get("PTCG_DECK_MATRIX") else None

    def _stamp_epoch(ep):
        if not _dm_epoch_file:
            return
        try:
            with open(_dm_epoch_file + ".tmp", "w") as fh:
                fh.write(str(ep))
            os.replace(_dm_epoch_file + ".tmp", _dm_epoch_file)
        except OSError:
            pass                      # telemetry only, never fatal

    # ------------------------------------------------- mid-run deck-pool swap
    # A writer stages a grown blob and publishes "<version> <blob path>" with
    # os.replace. Every rank reads it at the same epoch boundary and swaps
    # together. See docs/deck-pool.md.
    _pool_ver = 0
    _pool_failed = set()          # versions already rejected; do not retry
    _pool_file = getattr(a, "deck_pool_file", None)
    _pool_every = max(1, getattr(a, "pool_check_every", 20))
    _pool_ver_path = os.path.join(run_dir, "pool_version")
    if os.path.exists(_pool_ver_path):        # resume: remember where we were
        try:
            _pool_ver = int(open(_pool_ver_path).read().split()[0])
        except (OSError, ValueError, IndexError):
            _pool_ver = 0

    def _read_pool_request():
        """-> (version, blob_path). Never raises: a half-written or malformed
        request must not take the run down."""
        if not _pool_file or not os.path.exists(_pool_file):
            return 0, None
        try:
            parts = open(_pool_file).read().split()
            return (int(parts[0]), parts[1]) if len(parts) >= 2 else (0, None)
        except (OSError, ValueError):
            return 0, None

    def _maybe_swap_pool(cur_ver):
        """Swap the training deck pool if every rank sees the same request.

        Runs between _C.train() returning and the next _C.rollouts(), which is
        the only window where the env threads are parked.
        """
        want, blob = _read_pool_request()
        # Ranks may read either side of the writer's os.replace. That is benign
        # -- it just means this round is not unanimous, so skip and re-check
        # later. Swapping on a split read is the corruption this guards
        # against.
        if _C.allreduce_i32(pl, want, 0) != _C.allreduce_i32(pl, want, 1):
            if rank == 0:
                print("deck pool: ranks disagree on the requested version "
                      "(writer landed mid-check); retrying next check",
                      flush=True)
            return cur_ver
        if not want or want == cur_ver or want in _pool_failed:
            return cur_ver
        n = _C.reload_decks(blob)
        # A rejected blob is normally common-mode (same file, same checks), but
        # "normally" is not good enough here: if it failed on SOME ranks only,
        # they are now training on different deck spaces and nothing downstream
        # would notice. Disagreement is therefore fatal, not a warning.
        if _C.allreduce_i32(pl, n, 0) != _C.allreduce_i32(pl, n, 1):
            raise SystemExit(
                f"FATAL: deck pool v{want} loaded on some ranks and not others "
                f"(this rank got {n}). Ranks now disagree about what deck id "
                f"means, so training data and the deck matrix would both "
                f"corrupt silently. Stopping.")
        if n < 0:
            _pool_failed.add(want)
            if rank == 0:
                print(f"deck pool: v{want} REJECTED from {blob} (see the "
                      f"pt_reload_decks message above); staying on the current "
                      f"pool and not retrying this version", flush=True)
            return cur_ver
        if rank == 0:
            print(f"deck pool -> v{want}, n_decks {n}, at epoch {epoch}",
                  flush=True)
            try:
                with open(_pool_ver_path + ".tmp", "w") as fh:
                    fh.write(f"{want} {n} {blob}\n")
                os.replace(_pool_ver_path + ".tmp", _pool_ver_path)
            except OSError:
                pass
            # `wandb` is the module only after a successful init, else None
            if wandb:
                # x world: the main loop logs at logs["agent_steps"], which is
                # the AGGREGATE across ranks, while pl.global_step is this
                # rank's own count. Logging the swap at the per-rank number
                # puts it behind wandb's monotonic step counter, so wandb
                # DROPS the row silently -- training would carry on with a
                # grown pool and nothing in the dashboard to show it.
                wandb.log({"env/n_decks": n, "env/pool_version": want},
                          step=pl.global_step * world)
        return want

    # ---------------------------------------------------- deck sampling weights
    # Same protocol as the pool swap above (unanimous version, fatal on a split
    # apply, no retry of a rejected version), with one deliberate difference:
    # this is ALSO applied at startup, so a supervisor restart re-applies the
    # live distribution without operator action. The request file is the single
    # source of truth; there is no separate version file to go stale, which is
    # the trap where a stale PTCG_TABLES silently reverts the deck pool.
    _w_failed = set()
    _w_file = getattr(a, "deck_weights_file", None)
    _w_ver = 0
    _w_warned = [None]            # mtime of the last unparseable file we warned about
    _w_stats = {}                 # last applied weight stats, re-logged every row

    def _apply_weights(want, dflt, per):
        """-> (n_active, note). n_active < 0 means rejected, note says why."""
        names, _ = pool_deck_names(a.deck_pool)
        w, unknown = weight_vector(names, dflt, per)
        if unknown:
            return -1, (f"{len(unknown)} name(s) not in {a.deck_pool}, e.g. "
                        f"{unknown[0]}")
        # An anchor is drawn BEFORE the pool draw (env.c draw_deck), so a
        # zeroed anchor would keep being played while every report said it was
        # off. Harmless while the probabilities are 0, silent if they are not.
        if max(a.p_anchor_learner, a.p_anchor_self_opp,
               a.p_anchor_league_opp) > 0:
            zeroed = [names[i] for i in a.anchor_idx
                      if 0 <= i < len(w) and w[i] == 0]
            if zeroed:
                return -1, (f"would zero anchor deck(s) {zeroed} while "
                            f"p_anchor > 0; the anchor branch bypasses the "
                            f"weighted draw so they would still be played")
        return _C.set_deck_weights(w), ""

    def _maybe_apply_weights(cur_ver, at_start=False):
        want, dflt, per = read_weight_request(_w_file)
        # Ranks may read either side of the writer's os.replace; that is benign
        # mid-run (skip, recheck later) but NOT at startup, where skipping
        # would silently resume on the wrong distribution.
        if _C.allreduce_i32(pl, want, 0) != _C.allreduce_i32(pl, want, 1):
            if at_start:
                raise SystemExit(
                    "FATAL: ranks disagree on the deck-weights version at "
                    "startup (the request file changed mid-launch). Re-run "
                    "once the file is stable.")
            if rank == 0:
                print("deck weights: ranks disagree on the requested version "
                      "(writer landed mid-check); retrying next check",
                      flush=True)
            return cur_ver
        if not want:
            # The file exists but yielded no version: a malformed line, or no
            # `version` line at all. Skipping is right, but doing it silently
            # is not -- an operator who typed one bad weight would otherwise
            # see nothing at all and assume the change was live. Warn once per
            # (rank 0, file state), not every check, or it floods the log.
            if _w_file and os.path.exists(_w_file) and rank == 0:
                stamp = os.path.getmtime(_w_file)
                if stamp != _w_warned[0]:
                    _w_warned[0] = stamp
                    print(f"deck weights: {_w_file} exists but yields no "
                          f"version -- a malformed line, or the 'version' line "
                          f"is missing. NOTHING was applied.", flush=True)
            return cur_ver
        if want in _w_failed or (want == cur_ver and not at_start):
            return cur_ver
        n_active, note = _apply_weights(want, dflt, per)
        # A split apply means ranks are sampling different deck distributions,
        # which corrupts both the training data and the deck matrix with
        # nothing downstream to notice. Same reasoning as the pool swap.
        if _C.allreduce_i32(pl, n_active, 0) != _C.allreduce_i32(pl, n_active, 1):
            raise SystemExit(
                f"FATAL: deck weights v{want} applied on some ranks and not "
                f"others (this rank got {n_active}). Ranks now disagree about "
                f"the sampling distribution. Stopping.")
        if n_active < 0:
            _w_failed.add(want)
            if rank == 0:
                why = note or "see the pt_set_deck_weights message above"
                print(f"deck weights: v{want} REJECTED ({why}); staying on the "
                      f"current distribution and not retrying this version",
                      flush=True)
            return cur_ver
        if rank == 0:
            names, _ = pool_deck_names(a.deck_pool)
            w, _unknown = weight_vector(names, dflt, per)   # already validated
            s1, s2 = sum(w), sum(x * x for x in w)
            # Effective sample size. This is the number that catches a
            # fat-fingered file: a typo putting most of the mass on one deck
            # collapses ESS to ~1 while min/max alone can look reasonable.
            ess = (s1 * s1 / s2) if s2 > 0 else 0.0
            print(f"deck weights -> v{want}, {n_active}/{len(w)} active, "
                  f"ESS {ess:.1f}, min {min(w):.3g} max {max(w):.3g}, "
                  f"at epoch {epoch}", flush=True)
            # Kept and re-emitted with every metrics row; see the note there for
            # why a one-shot row at apply time does not survive a resume.
            _w_stats.clear()
            _w_stats.update({"env/deck_w_version": want,
                             "env/n_active_decks": n_active,
                             "env/deck_ess": ess,
                             "env/deck_w_min": min(w),
                             "env/deck_w_max": max(w)})
            if wandb:
                # Aggregate step, not per-rank: logging at pl.global_step puts
                # the pool-swap row behind wandb's monotonic counter and it is
                # dropped silently (see _maybe_swap_pool).
                wandb.log(dict(_w_stats), step=pl.global_step * world)
        return want

    if _w_file and not a.rollout_only:
        _w_ver = _maybe_apply_weights(_w_ver, at_start=True)

    _stamp_epoch(epoch)               # before the first flush can happen
    while pl.global_step < total:
        _C.rollouts(pl)
        if not a.rollout_only:
            _C.train(pl)
        elif a.episodes and rank == 0 \
                and _episodes_written(a) >= a.episodes:
            print(f"rollout-only: reached {a.episodes} episodes, stopping",
                  flush=True)
            break
        epoch += 1
        _stamp_epoch(epoch)
        # Between _C.train() and the next _C.rollouts(): env threads parked.
        if _pool_file and not a.rollout_only and epoch % _pool_every == 0:
            _pool_ver = _maybe_swap_pool(_pool_ver)
        # After the pool check, never before: a batch that appends decks and
        # then reweights must resolve its names against the GROWN pool.
        if _w_file and not a.rollout_only and epoch % _pool_every == 0:
            _w_ver = _maybe_apply_weights(_w_ver)
        if rank == 0 and (time.time() - last_log > 10 or pl.global_step >= total):
            logs = _C.log(pl)
            last_log = time.time()
            env = logs.get("env", {})
            loss = logs.get("loss", {})
            league = ""
            if a.league_bins:
                per_bank = " ".join(
                    f"{b}:{env.get(f'league_w_{b}', -1):.2f}"
                    f"/{env.get(f'league_n_{b}', 0):.2f}"
                    for b in range(len(a.league_bins)))
                league = (f"lgfrac {env.get('league_frac', 0):.3f} "
                          f"lglen {env.get('league_len', 0):.0f} "
                          f"lgw {env.get('win_vs_league', -1):.3f} "
                          f"[{per_bank}] ")
            print(f"epoch {epoch} step {pl.global_step} SPS {logs['SPS']} "
                  f"wall {time.time() - t_start:.0f}s "
                  f"lr {logs.get('lr', a.lr):.5f} "
                  f"vram {logs['util']['vram_used_gb']:.1f}G "
                  f"win_vs_rand {env.get('win_vs_rand', -1):.3f} "
                  f"ep_len {env.get('ep_len', 0):.0f} "
                  f"{league}"
                  f"ent {loss.get('entropy', 0):.3f} "
                  f"kl {loss.get('kl', 0):.4f} "
                  f"vf {loss.get('value', 0):.4f} "
                  f"klskip {loss.get('kl_skipped', 0):.0f} "
                  f"vfskip {loss.get('vf_skipped', 0):.0f}", flush=True)
            if a.rollout_only:
                # Episode rate from agent-steps/ep_len rather than the file
                # count: it updates every log instead of once per rotation, so
                # the ETA is live from the first minute.
                el = env.get("ep_len", 0) or 0
                dt = max(1e-6, time.time() - roll_t0)
                rate = ((pl.global_step - roll_start_step) / el / dt) if el else 0
                on_disk = _episodes_written(a)
                est = int((pl.global_step - roll_start_step) / el) if el else 0
                print(f"   rollout {_eta_str(max(on_disk, est), a.episodes, rate)}"
                      f"  [on disk {on_disk}, in flight {max(0, est - on_disk)}]",
                      flush=True)
            # Divergence hard-abort (docs/training.md): a diverging run can
            # burn tens of epochs of unambiguous KL warning and then hours of
            # NaN with nobody home. Sustained KL over the abort threshold, or
            # any non-finite loss, ends the run NOW. SIGKILL the whole process
            # group, not just ourselves: rank-1+ mp.spawn workers survive
            # their parent and sit in NCCL holding VRAM.
            # The supervisor sees the marker and does not relaunch.
            kl = None if a.rollout_only else loss.get("kl")
            vfl = None if a.rollout_only else loss.get("value")
            bad = [] if a.rollout_only else [
                   f"{k}={v}" for k in ("kl", "entropy", "total", "value")
                   if (v := loss.get(k)) is not None and not math.isfinite(v)]
            if kl is not None and a.kl_abort > 0 and kl > a.kl_abort:
                kl_bad_streak += 1
            else:
                kl_bad_streak = 0
            # The critic ignites BEFORE the policy KL moves (value 0.023 ->
            # 0.5 -> 10 -> 160+ while KL sits calm), so it gets its own abort
            # line.
            if vfl is not None and a.vf_abort > 0 and vfl > a.vf_abort:
                vf_bad_streak += 1
            else:
                vf_bad_streak = 0
            log_reads += 1
            if log_reads > 2 and (bad
                    or (a.kl_abort > 0 and kl_bad_streak >= a.kl_abort_consec)
                    or (a.vf_abort > 0 and vf_bad_streak >= a.kl_abort_consec)):
                reason = (f"non-finite losses ({', '.join(bad)})" if bad else
                          f"kl {kl:.4f} > {a.kl_abort} on {kl_bad_streak} "
                          f"consecutive logs" if kl_bad_streak >= a.kl_abort_consec
                          and a.kl_abort > 0 else
                          f"value loss {vfl:.2f} > {a.vf_abort} on "
                          f"{vf_bad_streak} consecutive logs")
                print(f"FATAL DIVERGENCE: {reason} at epoch {epoch} "
                      f"step {pl.global_step}", flush=True)
                time.sleep(2)   # let the marker reach the supervised log file
                os.kill(0, signal.SIGKILL)
            if wandb:
                # lr is the APPLIED value read back from the device pointer
                # muon uses — no python-side schedule mirror to drift out of
                # sync (docs/training.md)
                flat = {"SPS": logs["SPS"], "agent_steps": logs["agent_steps"],
                        "epoch": epoch, "lr": logs.get("lr", a.lr)}
                # deck_wr_<i>/deck_n_<i> come out of the C env keyed by
                # position in --anchor-decks; relabel with the deck's short
                # name so the wandb series is readable (env/deck_wr_marnie...).
                flat.update({f"env/{_deck_metric_name(k, _dshort)}": v
                             for k, v in env.items()})
                flat.update({f"losses/{k}": v for k, v in loss.items()})
                # Deck-weight state goes out with EVERY metrics row, not just
                # once when it changes. A one-shot row at apply time is dropped
                # whenever the apply lands at a step wandb has already seen --
                # which is exactly what a resume does, since the run restarts
                # from a checkpoint BEHIND the last logged step ("Tried to log
                # to step N that is less than the current step M"), even though
                # the weights are live. Repeating the values makes the series
                # survive any resume.
                if _w_stats:
                    flat.update(_w_stats)
                wandb.log(flat, step=logs["agent_steps"])
        if rank == 0 and not a.rollout_only \
                and epoch % a.checkpoint_interval == 0:
            path = os.path.join(run_dir, f"{pl.global_step:016d}.bin")
            _C.save_weights(pl, path)
            _C.save_state(pl, state_path)
            snapshot_state(run_dir, state_path, pl.global_step)
            print(f"saved {path} (+state)", flush=True)
    if rank == 0 and not a.rollout_only:
        _C.save_weights(pl, os.path.join(run_dir, "final.bin"))
        _C.save_state(pl, state_path)
        snapshot_state(run_dir, state_path, pl.global_step)
    _C.close(pl)
    os._exit(0)  # pufferl-style hard exit (env threads are non-daemon)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="bench")
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--total-agents", type=int, default=2048)
    p.add_argument("--num-buffers", type=int, default=2)
    p.add_argument("--num-threads", type=int, default=6)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--accum-minibatches", type=int, default=1,
                   help="micro-batches per optimizer step: grads of N "
                        "micro-batches of --minibatch rows are AVERAGED, then "
                        "one muon step applies. Effective minibatch = N * "
                        "--minibatch at unchanged VRAM (the update is the "
                        "big-batch update, not N times it). 1 = off.")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--min-lr-ratio", type=float, default=0.0,
                   help="cosine floor as a fraction of --lr (0.2 with --lr "
                        "0.001 => decays to 0.0002). Was hardcoded 0.0, i.e. "
                        "every annealed run decayed to zero.")
    p.add_argument("--lr-anneal-from-step", type=int, default=0,
                   help="PER-RANK step at which the cosine starts (0 = "
                        "upstream epoch-0 anchor). Set this when enabling "
                        "annealing MID-RUN so the curve spans [here, target] "
                        "instead of riding the tail of a curve anchored "
                        "before the run began.")
    p.add_argument("--lr-warmup", type=int, default=0,
                   help="linear LR ramp over the first N epochs (composes "
                        "with the anneal; 0 = off). For weights-only "
                        "restarts: docs/training.md")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=2.0)
    p.add_argument("--mix-self", type=float, default=0.98)
    # Weights-only restarts lose the epoch counter and with it the prio-beta
    # anneal position (beta = beta0 + 0.64*epoch/total). A restart mid-run
    # should pass the previous run's effective beta here or its prioritized
    # updates come back systematically hotter (a mid-run control at beta~0.45
    # vs 0.20 from a fresh counter).
    p.add_argument("--prio-beta0", type=float, default=0.2)
    # Divergence guards (docs/training.md). kl-skip is enforced
    # per-minibatch on the GPU (0 disables); kl-abort is checked on rank 0
    # against the logged epoch-mean KL (0 disables).
    p.add_argument("--kl-skip", type=float, default=0.05,
                   help="skip a minibatch's optimizer step when its mean "
                        "approx-KL exceeds this (0 = off)")
    p.add_argument("--kl-abort", type=float, default=0.3,
                   help="kill the run when logged KL exceeds this on "
                        "--kl-abort-consec consecutive logs (0 = off)")
    p.add_argument("--kl-abort-consec", type=int, default=3)
    p.add_argument("--vf-skip", type=float, default=0.5,
                   help="skip a minibatch's optimizer step when its mean "
                        "value loss exceeds this (0 = off) — the critic "
                        "ignites before the policy KL moves")
    p.add_argument("--vf-abort", type=float, default=20.0,
                   help="kill the run when logged value loss exceeds this "
                        "on --kl-abort-consec consecutive logs (0 = off)")
    p.add_argument("--muon-wd-1d", type=float, default=0.01,
                   help="decoupled weight decay on 1-D params (LN gains, "
                        "biases, readouts) — the un-orthogonalized Muon path")
    # League (frozen historical opponents on GPU). Banks are numbered in list
    # order; per-bank win rates log as env/league_w_<idx>.
    p.add_argument("--league-ckpts", default="",
                   help="comma list of frozen league checkpoints — native "
                        ".bin blobs, or .pt files (exported to .native.bin "
                        "at launch). Empty = no league (self-play only)")
    p.add_argument("--league-pct", type=float, default=0.019,
                   help="fraction of agents each frozen bank owns (10 banks "
                        "at 0.019 -> ~19%% of rows play league games)")
    # Anchor-deck oversampling (per-seat; 0 = plain uniform draw)
    p.add_argument("--anchor-decks", default=ANCHOR_DECKS,
                   help="comma list of deck names to oversample")
    p.add_argument("--p-anchor-learner", type=float, default=0.0,
                   help="P(learner seat draws an anchor deck)")
    p.add_argument("--p-anchor-self-opp", type=float, default=0.0,
                   help="P(opponent seat draws an anchor) in self-play games")
    p.add_argument("--p-anchor-league-opp", type=float, default=0.0,
                   help="P(opponent seat draws an anchor) in league games")
    p.add_argument("--deck-pool", default="decks/pool",
                   help="pool the tables blob was built from. Deck ids are "
                        "positions in this sorted 60-line glob, so anchor "
                        "resolution must read the SAME dir export_tables.py "
                        "used or every index is silently wrong.")
    p.add_argument("--deck-pool-file", default=None,
                   help="path to a deck-pool REQUEST file, enabling mid-run "
                        "pool growth. Contents: '<version> <blob path>', "
                        "published with os.replace. Every rank reads it at the "
                        "same epoch boundary and swaps together, and a version "
                        "the ranks do not unanimously see is skipped rather "
                        "than applied. APPEND ONLY -- the new blob's existing "
                        "deck rows must be bytewise identical or it is "
                        "rejected. See docs/deck-pool.md")
    p.add_argument("--pool-check-every", type=int, default=20,
                   help="epochs between deck-pool request checks")
    p.add_argument("--deck-weights-file", default=None,
                   help="path to a deck-WEIGHTS request file, enabling mid-run "
                        "reshaping of the deck sampling distribution. Format: "
                        "'version <n>' and 'default <w>' lines, then "
                        "'<deck name> <weight>' lines; weight 0 disables a "
                        "deck without removing it (its id survives, so old "
                        "deck matrices stay summable). Weights are relative "
                        "within the training pool and are normalised to mean 1, "
                        "so the coverage share is unaffected. Read at startup "
                        "AND every --pool-check-every epochs, so a restart "
                        "re-applies without operator action. Names, never "
                        "indices: an append renumbers nothing, but an "
                        "index-keyed file would still mis-target as the pool "
                        "grows. See docs/deck-pool.md")
    p.add_argument("--deck-matrix-every", type=int, default=0,
                   help="episodes per deck-vs-deck matrix flush, PER RANK "
                        "(0 = off). Writes <run_dir>/deckmat/r<rank>_e<epoch>_<seq>.bin: "
                        "a 203x203 games+wins matrix over the whole pool, for "
                        "offline archetype analysis. ~330KB per flush; the "
                        "supervisor ships them to HF. Read with "
                        "tools/deck_matrix.py")
    p.add_argument("--cudagraphs", type=int, default=10)
    p.add_argument("--rollout-only", action="store_true",
                   help="generate episodes with FROZEN weights and never call "
                        "_C.train: no optimizer step, no checkpointing, no "
                        "divergence guard. For dataset generation (the event "
                        "log) off a fixed checkpoint. Strictly stronger than "
                        "'--lr 0', which still runs Muon and would have to be "
                        "proven not to move the weights.")
    p.add_argument("--episodes", type=int, default=0,
                   help="rollout-only: stop once this many episodes have been "
                        "written to PTCG_EVENT_LOG (counted in completed "
                        "files, so the granularity is PTCG_EVENT_LOG_EVERY).")
    p.add_argument("--checkpoint-interval", type=int, default=50)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-name", default=None,
                   help="wandb run id/name override (deleted ids can never "
                        "be reused — 'run ID is in use' forever)")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore an existing state_latest.bin and start fresh")
    p.add_argument("--start-step", type=int, default=0,
                   help="weights-only restart: the source checkpoint's "
                        "PER-RANK step (its .bin filename) — continues "
                        "step/epoch counters, eval axes, and epoch-driven "
                        "schedules from the parent run; --lr-warmup still "
                        "ramps relative to this point")
    a = p.parse_args()

    # Gradient accumulation: groups must tile the epoch's minibatches exactly
    # (horizon is fixed at 64 in the train config; replay_ratio is 1.0).
    if a.accum_minibatches < 1:
        p.error("--accum-minibatches must be >= 1")
    if (a.total_agents * 64) % (a.minibatch * a.accum_minibatches) != 0:
        p.error(f"total_agents*horizon ({a.total_agents * 64}) must be "
                f"divisible by minibatch*accum "
                f"({a.minibatch * a.accum_minibatches})")
    if a.accum_minibatches > 1:
        print(f"grad accumulation: {a.accum_minibatches} x {a.minibatch} = "
              f"effective minibatch {a.accum_minibatches * a.minibatch}",
              flush=True)

    # Anchor deck indices (verified against the tables blob the env loads).
    # Resolved whenever --anchor-decks is non-empty, INDEPENDENTLY of the
    # p_anchor_* weights: the indices do double duty as (a) the oversample set
    # and (b) the label set for per-deck training win rates. With all three
    # probabilities at 0 the sampling stays uniform over the whole pool and
    # these decks are merely counted.
    names = [s.strip() for s in a.anchor_decks.split(",") if s.strip()]
    if names:
        tables = os.environ.get("PTCG_TABLES", "native/ptcg_tables.bin")
        a.anchor_idx = resolve_anchor_indices(names, tables, a.deck_pool)
    else:
        a.anchor_idx = []
    a.deck_short = [n.split("__")[0][:24] for n in names]

    # League checkpoints: accept native .bin blobs or .pt (auto-exported once
    # here, pre-spawn — the only place torch runs).
    a.league_bins = []
    if a.league_ckpts:
        for pth in (s.strip() for s in a.league_ckpts.split(",")):
            if not pth:
                continue
            if pth.endswith(".pt"):
                bp = pth[:-len(".pt")] + ".native.bin"
                if not os.path.exists(bp) or \
                        os.path.getmtime(bp) < os.path.getmtime(pth):
                    subprocess.check_call(
                        [sys.executable, "native/tools/native_weights.py",
                         "export", pth, bp],
                        env={**os.environ, "PYTHONPATH": "data:.:native"})
                pth = bp
            assert os.path.exists(pth), f"league checkpoint missing: {pth}"
            a.league_bins.append(pth)
        assert 0 < len(a.league_bins) <= 16, "1..16 league banks (PT_MAX_BANKS)"
        assert a.league_pct * len(a.league_bins) < 0.5, \
            "league banks would own >50% of rows — check --league-pct"

    run_dir = os.path.join("experiments", f"native-{a.run_name}")
    os.makedirs(run_dir, exist_ok=True)
    init = os.path.join(run_dir, "init.bin")
    if not os.path.exists(init):
        subprocess.check_call(
            [sys.executable, "native/tools/native_weights.py", "init", init,
             "--seed", str(a.seed)],
            env={**os.environ, "PYTHONPATH": "data:.:native"})

    if a.gpus == 1:
        worker(a, 0, 1, 0, b"", run_dir)
        return
    from puffer_ptcg import _C
    nccl_id = _C.get_nccl_id()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(a, r, a.gpus, r, nccl_id, run_dir))
             for r in range(1, a.gpus)]
    for pr in procs:
        pr.start()
    worker(a, 0, a.gpus, 0, nccl_id, run_dir)


if __name__ == "__main__":
    main()
