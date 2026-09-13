# Training with the native backend

Build the native trainer, run it, read its telemetry, drive the tools around it. Linux with an
NVIDIA GPU; every command runs from the repo root.

## 1. What the native backend is

The trainer is one CUDA process per GPU. Three pieces compile into a single Python extension,
`native/puffer_ptcg/_C<ext>.so`:

| piece | what it is |
|---|---|
| `native/ptcg/` | the Pokemon TCG environment in C — observation encoder, effect tracker, oracle, deck sampling, self-play driver — linked against the competition engine (`data/cg/libcg.so`) and a static tables blob |
| `native/model/` | the policy/value network written directly in CUDA, forward and backward, mirroring the torch reference in `ptcg/rl/model.py` |
| `native/src/` | PufferLib's native CUDA trainer, vendored at commit `c5d3c63` and patched locally (`native/PATCHES.md`) |

The env is statically linked in, so rollouts never cross a Python boundary: the C env fills a pinned
observation buffer, the CUDA model acts on it, PPO and Muon run on device. One env is one game is
one agent row, and self-play serves both seats through it. An observation is 10944 fp32 values;
there is one action head of 64 options with a mask. PyTorch is not in the training loop — it runs
only through `native/tools/native_weights.py`, at launch to create
`experiments/native-<run>/init.bin` (the packed fp32 initial weights; the CUDA model refuses to
random-init) and offline to convert step-stamped weight blobs into `.pt` checkpoints for eval, or a
`.pt` league opponent into a blob.

## 2. Prerequisites, data, and build

Needed: `nvcc` and `gcc`/`g++` with C++17 and OpenMP; the Python packages in
`setup/requirements.txt` (torch, numpy, tensorboard, pufferlib, wandb) plus `pybind11`; NCCL, linked
even for one GPU, either system-wide or from the `nvidia-nccl-cu12` / `nvidia-nccl-cu13` wheel;
`huggingface_hub` only if the supervisor uploads artifacts. Put the competition SDK at `data/cg/`
and the card sheet at `data/EN_Card_Data.csv` (`data/README.md`), then build the card database and
the tables blob (every static table the C env and the CUDA model read, deck table included):

```bash
PYTHONPATH=data python3 tools/build_carddb.py
PYTHONPATH=data:. python3 native/tools/export_tables.py native/ptcg_tables.bin \
    --pool decks/pool --alt-pool decks/pool_alt
```

Deck ids are positions in the sorted glob of `--pool`, so the blob and the trainer's `--deck-pool`
must be the same directory or every index is silently wrong; `--alt-pool` adds a coverage pool the
random opponent draws from, numbered from a fixed high base so ids never collide
(`docs/deck-pool.md`). `init.bin` is created automatically on the first launch of a run name.

```bash
bash native/build_native.sh          # --debug for -O0 -g
```

This compiles the C env with `-fopenmp` into `native/build/libstatic_ptcg.a`, compiles
`native/src/bindings.cu` with nvcc (`-DPTCG_NATIVE -DPRECISION_FLOAT -DOBS_TENSOR_T=FloatTensor
-DENV_NAME=ptcg`), and links both with CUDA, cuBLAS, cuRAND, NCCL, NVML and `libcg`. The GPU arch
comes from the device, falling back to `sm_86` — override with `PTCG_SM=120` for a card the box does
not have. NCCL is looked for in system paths, then in the `nvidia.nccl` wheel; wheels ship only
`libnccl.so.2`, so that soname is linked directly. NVML links through `$CUDA_HOME/lib64/stubs` when
the `libnvidia-ml.so` development symlink is absent, and `PY=` selects the interpreter.

The tests below build with the host compiler only — no GPU, no CUDA. The deck targets take tables
blobs as variables (`docs/deck-pool.md`); `make -C native build/test_stall` builds the section 14
test, run by hand.

| command | what it checks |
|---|---|
| `make -C native test` | 16 envs x 20000 steps: masks non-empty, reward only at terminals, no illegal actions, no engine errors |
| `make -C native parity` | replays `native/parity_stream.jsonl` through the C tracker/oracle/encoder, comparing rows bit-for-bit against `native/parity_rows.bin` |
| `make -C native reload V1=… GROWN=… INSERTED=…` | the mid-run deck-append contract of `pt_reload_decks` |
| `make -C native decklock V1=… GROWN=…` | that a swapper holding the write lock stalls the env threads |
| `make -C native deckweights V1=… [GROWN=…]` | that per-deck sampling weights shape the draw and fail safe |

## 3. Model size is a compile-time constant

`PT_D`, `PT_H`, `PT_FFN`, `PT_LAYERS` at the top of `native/model/ptcg_model.cu` are the whole size
knob; every other width derives from them. The same four numbers exist again as `D, HEADS, FFN,
LAYERS` in `native/tools/native_weights.py`, which owns the packed-blob layout (`param_order()`
mirrors `PT_PARAMS` name for name). A mismatch does not crash, it produces a silently wrong `.pt`
and therefore wrong evals: a wrong width usually trips the blob length check, but a wrong head count
does not, because the qkv projections are `d x d` whatever the head count — the converted model
loads cleanly and computes different attention. Check both sides, change them together:

```bash
grep -E '^#define PT_(D|H|FFN|LAYERS) ' native/model/ptcg_model.cu
PYTHONPATH=data:.:native python3 -c \
  "import sys; sys.path.insert(0,'native/tools'); import native_weights as w; \
   print(w.D, w.HEADS, w.FFN, w.LAYERS)"
```

Everything downstream reads the Python copy rather than restating it: `model_parity.py`,
`stage_parity.py`, `EVAL_HEADS` in `native_eval_watch.sh`.

## 4. Running

```bash
PYTHONPATH=data:.:native PTCG_TABLES=native/ptcg_tables.bin \
python3 native/train_native.py --run-name demo --timesteps 200000000 \
    --total-agents 2048 --minibatch 1024 --lr 0.001 --wandb
```

Artifacts land in `experiments/native-<run-name>/`. An epoch is `--total-agents * 64` steps per rank
(horizon is fixed at 64), and that product must divide by `--minibatch * --accum-minibatches`.

| scale / loop | default | effect |
|---|---|---|
| `--run-name` | `bench` | names the run directory and the default wandb id |
| `--timesteps` | 20000000 | ABSOLUTE global-step target, split across ranks |
| `--total-agents` | 2048 | parallel games per rank; the main VRAM knob |
| `--num-buffers`, `--num-threads` | 2, 6 | rollout buffering depth; env stepping threads per rank |
| `--minibatch` | 1024 | minibatch rows PER RANK |
| `--accum-minibatches` | 1 | micro-batches per optimizer step; their grads are averaged, so the update is the big-batch update at unchanged VRAM |
| `--cudagraphs`, `--checkpoint-interval` | 10, 50 | epoch at which rollout and train graphs are captured (negative disables capture); epochs between weight + state saves |
| `--gpus`, `--seed` | 1, 42 | ranks (section 5); per-rank seeds derive from the seed |
| `--wandb`, `--wandb-name` | off | log to wandb (project `ptcg`, group `native`); the name overrides the run id |
| `--no-resume`, `--start-step` | off, 0 | ignore an existing `state_latest.bin`; stamp this per-rank step for a weights-only restart (section 7) |

| optimization | default | effect |
|---|---|---|
| `--lr`, `--no-anneal-lr`, `--min-lr-ratio` | 0.001, off, 0.0 | Muon learning rate; flat LR instead of the cosine; cosine floor as a fraction of `--lr` |
| `--lr-anneal-from-step` | 0 | PER-RANK step the cosine is anchored at; set it when enabling annealing mid-run so the curve spans "here to target" instead of riding the tail of a curve anchored before the run began |
| `--lr-warmup` | 0 | linear ramp over the first N epochs, on top of the anneal, counted relative to `--start-step` |
| `--gamma`, `--gae-lambda` | 1.0, 0.95 | discount (per decision) and GAE lambda |
| `--ent-coef`, `--vf-coef` | 0.01, 2.0 | entropy bonus and value-loss weight |
| `--mix-self` | 0.98 | fraction of untagged rows playing mirror self-play; the rest play a random-legal opponent, which is what `win_vs_rand` measures |
| `--prio-beta0` | 0.2 | prioritized-replay beta at epoch 0; it anneals with the epoch counter |

| guards / league / decks | default | effect |
|---|---|---|
| `--kl-skip`, `--vf-skip` | 0.05, 0.5 | veto a minibatch's optimizer step when its mean approx-KL, or its mean value loss, exceeds this (0 = off) |
| `--kl-abort`, `--vf-abort`, `--kl-abort-consec` | 0.3, 20.0, 3 | kill the run when logged KL, or logged value loss, exceeds this (0 = off) on that many consecutive logs |
| `--muon-wd-1d` | 0.01 | decoupled weight decay on 1-D parameters (section 9) |
| `--league-ckpts`, `--league-pct` | empty, 0.019 | comma list of frozen opponents (`.bin` blobs or `.pt` files; empty = no league) and the fraction of rows each bank owns |
| `--anchor-decks` | six names | decks with their own win-rate series, oversampled when a `p_anchor_*` is non-zero |
| `--p-anchor-learner`, `--p-anchor-self-opp`, `--p-anchor-league-opp` | 0.0 | probability that the learner seat, the self-play opponent seat, or the league opponent seat draws an anchor deck |
| `--deck-pool` | `decks/pool` | the pool the tables blob was built from; anchor and weight names resolve against it |
| `--deck-pool-file`, `--deck-weights-file` | none | request files enabling mid-run pool growth and mid-run reweighting; both re-read every `--pool-check-every` epochs (default 20), weights also at startup |
| `--deck-matrix-every` | 0 | episodes per deck-vs-deck matrix flush, per rank (section 11) |
| `--rollout-only`, `--episodes` | off, 0 | frozen weights with no optimizer step, checkpointing or guard, stopping once that many episodes are on disk |

Deck-pool and deck-weight mechanics are in `docs/deck-pool.md`. Fixed in `build_args()` rather than
exposed: horizon 64, replay ratio 1.0, clip 0.2, value clip 0.2, max grad norm 1.5, Muon betas
0.95/0.999, prio alpha 0.8, v-trace clips 1.0, terminal reward +/-1, 3000 engine steps to
truncation.

**Reading the log while a run is live.** The `epoch` line and the `PTCG_DEBUG_ROLLOUT` heartbeat
are written by the extension through C stdio, which is fully buffered when stdout is a file and is
not affected by `stdbuf` or `PYTHONUNBUFFERED`; they reach the file when the process exits. While
the run is up, the checkpoint files in `experiments/native-<run>/` (every `--checkpoint-interval`
epochs) and `nvidia-smi` are the live signals.

**Open issue on 12 GB cards.** On an RTX 3060 (12 GB) at `--total-agents 512 --minibatch 256`
(11.4 GB in use, about 790 SPS), single-GPU runs with `--checkpoint-interval 5
--deck-matrix-every 2000` printed four healthy epochs and then stopped advancing at the start of
the fifth rollout (the heartbeat stays at `t=0/64`), with the GPU at 100% and no error, for two
different seeds. The same code trains for billions of steps on 24 GB cards at the documented
settings. Not yet reduced; if you hit it, try fewer agents first and report the settings.

## 5. Multi-GPU

`--gpus N` runs one process per GPU: rank 0 in the launching process, the rest as `spawn` children
sharing an NCCL id. Each rank sets `CUDA_VISIBLE_DEVICES` before importing the extension — remapping
through an outer value rather than clobbering it — so the rollout pthreads, which never call
`cudaSetDevice`, land on the right device.

- Gradients are averaged with `ncclAllReduce` inside the Muon step, on the same stream, so the
  collective captures into the train graph.
- `--minibatch` is PER RANK (two ranks at 512 is one rank at 1024); `--timesteps` is aggregate and
  divided by the world size.
- Rank 0 owns logging, checkpointing and the request files; request versions and apply results are
  allreduced, so ranks cannot diverge on the deck pool or sampling distribution they train against.
- Export `NCCL_P2P_DISABLE=1` where PCIe peer-to-peer is unavailable; the eager warmup allreduce
  surfaces that as a clean error (`NCCL warmup allreduce failed`, and under `NCCL_DEBUG=WARN`
  `Cuda failure 217 'peer access is not supported between these two devices'`) rather than an
  illegal memory access later. `nvidia-smi topo -p2p r` prints `CNS` for such GPU pairs, the normal
  case for consumer cards on a PCIe chipset.
- A failure earlier than that, in `ncclCommInitRank`, with `CUDA driver version is insufficient
  for CUDA runtime version`, means the NCCL that `_C` links was built for a newer CUDA than the
  driver: rebuild with `NCCL_HOME` pointing at the `nvidia-nccl-cu12` wheel (`native/README.md`).
- Kill a run by pid, or by matching `train_native.py` and then `multiprocessing.spawn`: rank 1+ are
  spawn children whose command line omits `train_native.py` and outlive their parent holding VRAM.

## 6. Environment variables read by the C/CUDA side

Paths in the second column are relative to `native/`.

| variable | read in | effect |
|---|---|---|
| `PTCG_TABLES` | `ptcg/tables.c` | path to the tables blob (default `native/ptcg_tables.bin`); missing or bad magic aborts |
| `PTCG_INIT_WEIGHTS` | `model/ptcg_model4.cu` | packed fp32 initial weights; unset aborts, there is no native random init. The trainer points it at the run's `init.bin` |
| `PTCG_DECK_MATRIX` | `ptcg/env.c` | file prefix for deck-vs-deck matrices; unset = off at zero cost |
| `PTCG_DECK_MATRIX_EVERY` | `ptcg/env.c` | episodes per matrix flush (default 20000) |
| `PTCG_DECK_MATRIX_MAX` | `ptcg/env.c` | matrix width override; the default is the fixed coverage base, clamped up to the live deck count and never above that base, so files stay summable as the pool grows |
| `PTCG_EVENT_LOG` | `ptcg/env.c`, `src/vecenv.h` | file prefix for the game event log; also what allocates the decoder mirror the per-decision logits come from |
| `PTCG_EVENT_LOG_EVERY` | `ptcg/env.c` | episodes per event-log file (default 20000) |
| `PTCG_EVENT_DEBUG` | `ptcg/env.c` | dump each event capture window to stderr |
| `PTCG_DUMP_DIR` | `model/ptcg_model3.cu` | dump every named forward stage as raw f32 for `stage_parity.py` |
| `PTCG_SYNC_CHECK` | `model/ptcg_model3.cu` | `=1` syncs and error-checks after each forward stage, aborting with the stage name |
| `PTCG_DEBUG_ROLLOUT` | `src/vecenv.h` | `=1` prints a rollout heartbeat every 16 steps — a stalled rollout versus a slow one |
| `PTCG_BLOCKING_SYNC` | `src/pufferlib.cu` | sleeping CUDA syncs, on by default so idle syncs do not spin-burn cores; `=0` opts out |
| `PTCG_TF32` | `model/ptcg_kernels.cuh` | `=0` forces true fp32 cuBLAS compute; anything else keeps TF32 |
| `OMP_WAIT_POLICY` | OpenMP runtime | the trainer defaults it to `PASSIVE` so OMP workers sleep at barriers |
| `PUFFER_HEAD_GATING` | `src/pufferlib.cu` | opt-in consumed-head gating; it needs an env exporting `env_head_consume_map`, which this env does not, so it is inert here |

## 7. Checkpoints and resume

Rank 0 writes into `experiments/native-<run>/`. The live arena 16-byte-aligns every tensor while the
on-disk blob is packed, so all saves and loads go through the packed/arena helpers.

| file | contents |
|---|---|
| `<step16>.bin` | weights only, packed fp32 in `PT_PARAMS` order, named by zero-padded per-rank step. This is what `native_weights.py` and the eval converter read |
| `state_latest.bin` | full state: `PTCGSTA1` header with step/epoch, then packed weights, then packed Muon momentum. Written tmp + fsync + rename |
| `state_<step16>.bin` | rotated snapshots, hard-linked from `state_latest.bin` each checkpoint; the last 10 are kept |
| `final.bin` | weights at the end of the run |

Unless `--no-resume` is given, every rank loads `state_latest.bin` if present: weights, momentum,
`global_step` and `epoch`, and with the epoch the cosine LR and prio-beta positions. Since
`--timesteps` is absolute, a resume with a target below the resumed step does nothing and exits
looking like success — the trainer warns in exactly those words.

To fork from a bare `<step16>.bin` without optimizer state, drop it in as the new run's `init.bin`
and pass `--start-step <that step>`: `global_step` and `epoch` are stamped, so checkpoint names,
wandb axes, eval epoch numbering and epoch-driven schedules continue from the parent run. Two things
do not follow by themselves — `--prio-beta0`, whose anneal restarts cold from a fresh counter, and
the cosine, otherwise anchored at epoch 0 of a curve that began before this run existed
(`--lr-anneal-from-step`). `--lr-warmup` already ramps relative to `--start-step`.

## 8. Telemetry

Rank 0 prints one line roughly every ten seconds:

```
epoch 1200 step 157286400 SPS 10850 wall 43120s lr 0.00100 vram 22.6G win_vs_rand 1.000 ep_len 118 ent 0.531 kl 0.0261 vf 0.0152 klskip 0 vfskip 0
```

`SPS` is aggregate; `lr` is the applied rate read back from the device pointer Muon uses, not a
Python-side mirror of the schedule; `win_vs_rand` is the win rate over the random-legal-opponent
games `--mix-self` leaves; `ent`/`kl`/`vf` are epoch means; `klskip`/`vfskip` count minibatches the
guard vetoed since the last log; a league adds `lgfrac`/`lglen`/`lgw` and a per-bank
`idx:winrate/games` list. Healthy for a converged run: `win_vs_rand` 1.000, `ent` around 0.5, `kl`
0.02 to 0.03, `vf` around 0.015, `klskip` occasionally in the single digits, `vfskip` 0 — a `vfskip`
that starts firing, or `kl` past roughly 0.05, is the critic-divergence mode in section 9.

With `--wandb` the same data goes out keyed by aggregate agent steps: `SPS`, `agent_steps`, `epoch`,
`lr`, everything the C env logs under `env/` (`win_vs_rand`, `ep_len`, end-cause histograms,
`league_*`, per-anchor `deck_wr_<name>`/`deck_n_<name>`) and the losses under `losses/` (`policy`,
`value`, `entropy`, `total`, `kl`, `old_kl`, `clipfrac`, `kl_skipped`, `vf_skipped`). Deck-pool and
deck-weight state is re-emitted every row, so it survives a resume.

The run also stops itself: any non-finite loss, or KL above `--kl-abort` (or value loss above
`--vf-abort`) on `--kl-abort-consec` consecutive logs, prints `FATAL DIVERGENCE` and SIGKILLs the
whole process group, taking surviving rank workers with it. `supervisor_native.sh` reads that marker
and applies a 3-strikes rule: the first two aborts within an hour are treated as a bad start and
relaunched, the third stops the run, pushes artifacts and exits — so a corrupt start recovers while
a genuinely diverging run does not crashloop the box.

## 9. Divergence guards

**The per-minibatch gate.** `ppo_loss_reduce` writes this minibatch's mean approx-KL and mean value
loss into a small guard buffer. Under DDP the pair is allreduced (`ncclAvg`) before anything reads
it: gradients are averaged but updates apply rank-locally, so a per-rank skip decision would fork
the ranks' weights permanently. A one-thread kernel compares them against `--kl-skip` and
`--vf-skip` and writes a gate float; every Muon kernel, momentum and weight update alike, returns
early when the gate is 0. The veto is a branch, not a multiply — the gate exists precisely for
minibatches whose gradients may be non-finite, and `0 * nan` is `nan` — and it fails closed, since
NaN compares false against any threshold and the explicit finiteness test is what stops a fully
diverged minibatch. With gradient accumulation the gate runs on the group's final micro-batch,
vetoing the whole optimizer step.

**Why the value loss has its own threshold.** The critic ignites first: the value loss leaves its
flat band and climbs by orders of magnitude roughly ten epochs before approx-KL moves at all, so a
KL-only gate stays open while the critic destroys itself, and the policy storm that eventually shows
in KL is downstream damage from rotten advantages. `--vf-skip 0.5` sits far above a healthy value
loss and far below that ignition trajectory.

**The Muon 1-D path.** Upstream applies classic summed momentum to every parameter and
orthogonalizes only tensors of 2 or more dimensions, so a 1-D parameter (LayerNorm gains, biases,
readouts) takes a raw momentum step — roughly `1/(1-mu)` times the mean gradient at steady state,
sized by gradient magnitude rather than bounded by the learning rate, with no weight decay to damp
it — and gains and logit readouts enter a sharpening feedback loop that blows up. The patched
optimizer splits the walk: 2-D and larger tensors keep upstream's rule bit-for-bit, 1-D parameters
get EMA momentum (`m = mu*m + (1-mu)*g`, update `g + m`) and their own decoupled weight decay,
`--muon-wd-1d`. Setting that to 0 removes the damping; there is no flag to revert the EMA.

## 10. League training

`--league-ckpts a.bin,b.pt,…` loads frozen historical opponents as extra weight banks on the GPU.
`.pt` files are converted once at launch (the only place torch runs); `.bin` blobs load as is. At
most 16 banks (`PT_MAX_BANKS`), and the launcher refuses a configuration whose banks would own more
than half the rows.

Each bank owns `--league-pct` of the rows in every rollout buffer. Rows in a bank's slice always
play that bank — there is no per-game draw — and the env tags each with the bank id in one
observation column. On the GPU the learner forwards every row, each frozen bank forwards only its
slice into per-bank scratch, and a merge kernel overwrites the tagged rows' action and logprob with
the frozen bank's. The tag column is pulled into its own rollout tensor and zeroed before any
forward, because every checkpoint was trained with it at 0.

Values are deliberately not merged: the learner's critic prices every row, so a league game is a
mirror game with an external action source and a stale frozen critic cannot inject TD noise into
GAE. In the loss, frozen decisions train the critic but get no policy gradient, no entropy bonus and
no KL or clipfrac vote — their logprobs came from other weights, so the ratio is not an importance
weight. It is pinned to 1, the policy gradient is zeroed explicitly rather than scaled (`0 * inf` is
NaN), and policy-side statistics renormalize by the learner-decision count; otherwise meaningless
learner-versus-frozen logratios would push approx-KL past `--kl-skip` and veto every minibatch. With
no league flags nothing allocates, tags stay 0, and the PPO kernel takes the original path.
Telemetry: `league_frac`, `win_vs_league`, `league_len`, end-cause splits, and per bank
`league_w_<idx>` / `league_n_<idx>`, indexed by position in `--league-ckpts`.

## 11. Deck matrix and event log

`--deck-matrix-every N` accumulates a deck-versus-deck games/wins matrix from the learner seat's
perspective and flushes it every N finished episodes per rank, to
`experiments/native-<run>/deckmat/r<rank>_e<epoch>_<seq>.bin` (tmp + rename, so a reader never sees
a partial file). It is pre-sized to a constant width rather than the current pool size, so files
stay summable across a mid-run append; each is two `width x width` uint32 planes, and the counters
reset after every flush, so summing any set of files gives exactly that window. Read them with
`tools/deck_matrix.py`.

`PTCG_EVENT_LOG=<prefix>` makes the env write a full game record — every engine event plus, per
decision, the policy's logits over the offered options and its value estimate in fp16. Files are
`<prefix>_e<epoch>_<seq>.bin`, rotated every `PTCG_EVENT_LOG_EVERY` episodes: a 16-byte file header
(magic `PTEV`, version, record size, epoch), then per episode a 24-byte header (epoch, both deck
ids, seat, result, opponent kind, first player, event counts, payload size), 16-byte event records,
and the fp16 payload. Nothing is allocated when the variable is unset. Generate datasets off a fixed
checkpoint, not on a live run:

```bash
PYTHONPATH=data:.:native PTCG_TABLES=native/ptcg_tables.bin \
PTCG_EVENT_LOG=data/eventlog/run PTCG_EVENT_LOG_EVERY=20000 \
python3 native/train_native.py --run-name farm --rollout-only --episodes 1000000
```

`--rollout-only` never calls the train entry point: no optimizer step, no checkpoint, no guard —
strictly stronger than `--lr 0`, which would still run Muon. `--episodes` stops once that many
episodes are on disk, counted in completed files. `tools/event_log_deckmat.py` reads episode headers
only, seeking past events and payloads, and reports coverage and win rates straight from a dump.

## 12. Supervisor and eval watch

```bash
setsid nohup bash native/tools/supervisor_native.sh demo \
    > runs/native_supervisor.log 2>&1 < /dev/null &
```

`runs/native_<RUN>.args` holds everything after `python native/train_native.py` (newlines and `#`
comments allowed; the supervisor appends `--run-name`), and `runs/native_<RUN>.env` holds per-box
settings, sourced and exported so a reboot relaunch gets identical values. The supervisor relaunches
on death (resume is built into the trainer), kills and relaunches on a stale log, backs off 30
minutes after 5 launches in 25 minutes, applies the divergence rule from section 8, keeps the eval
watcher alive, and — only if `HF_REPO` is set — pushes rotated states, deck matrices and converted
models to that dataset repo once each.

```
# runs/native_demo.args
--timesteps 2000000000     --total-agents 2048     --minibatch 1024
--lr 0.001                 --deck-pool decks/pool  --deck-matrix-every 20000
--checkpoint-interval 50   --wandb
```

```
# runs/native_demo.env
PY=python3
TRAIN_GPUS=0
EVAL_GPU=1
STALE_MIN=15
EVAL_BENCH=1
STEPS_PER_EPOCH=131072
RANK_STEPS_PER_EPOCH=131072
PTCG_TABLES=native/ptcg_tables.bin
# HF_REPO=<user>/<dataset-repo>
```

For eight GPUs add `--gpus 8` to the args, drop `--total-agents` to 1024 and `--minibatch` to 512
(both per rank), then set `TRAIN_GPUS=0,1,2,3,4,5,6,7`, `STEPS_PER_EPOCH=524288`,
`RANK_STEPS_PER_EPOCH=65536` and `EVAL_BENCH=0` — with every GPU training there is nowhere to run
the battery, and it is CPU-hungry enough to cost the trainer throughput.

`native/tools/native_eval_watch.sh <RUN> [GPU]` converts each
`experiments/native-<RUN>/<step16>.bin` into `experiments/train-native_<RUN>/model_<epoch6>.pt` and,
unless `EVAL_BENCH=0`, runs the torch eval battery on that GPU. `STEPS_PER_EPOCH` is the eval x-axis
scale (aggregate steps per epoch); `RANK_STEPS_PER_EPOCH` turns a per-rank step-stamped filename
into an epoch number, and under DDP the two differ by the world size. `EVAL_HEADS` defaults to
`native_weights.HEADS` rather than a literal, for the reason in section 3; `EVAL_EPOCH_MOD` thins
the conversion cadence; the script takes a lock so watchers cannot stack.

## 13. Throughput

VRAM is the limit, not speed. Cut `--total-agents` first on an out-of-memory error (1536 or 1024
instead of 2048), `--minibatch` second. Model width dominates everything else: d128 runs about 1.5x
the throughput of d256 at otherwise identical settings. `--num-threads` barely matters — the loop is
more kernel-launch-latency-bound than env-throughput-bound — and cudagraph capture has not measured
as a clear win either way. The first two epochs are warmup and understate SPS; startup is several
minutes of NCCL init and graph capture. Rough fp32 numbers:

| setup | steps/s |
|---|---|
| 1x RTX 3090, 1024 agents, d128 | roughly 3K |
| 8x RTX 3090, d256 | roughly 11K aggregate |
| 8x RTX 4090, d256 / d128 | roughly 16K / 25K aggregate |

## 14. The Solar Transfer activation cap

One card's ability ("as often as you like during your turn", a reversible effect, no opponent
priority in between) is a legal, unbounded, state-preserving loop: an agent can repeat it forever
inside one turn. The episode then dies on the engine-step limit, which resets silently with no
terminal flag, so the trainer bootstraps across it and a stall reads as neutral rather than bad. A
terminal penalty cannot fix that: under a discount a loss later beats a loss now, so for a losing
agent stalling is correct play, and a penalty big enough to overturn it is outside the reward scale
the value-loss guard tolerates.

So the env caps it structurally. `PT_AB652_CARD` / `PT_AB652_CAP` in `native/ptcg/env.c` allow 10
activations of that card's ability per seat per turn; over-budget activations are cleared from the
policy mask (covering the learner and every frozen league bank) and from the random opponent's draw
pool. Forced answers are never filtered — forced means no alternative — and the sub-selections an
activation opens are other option types, so an in-flight resolution cannot be starved; ending the
turn is always available, so masking can never empty the option set. The cap is scoped to that one
card, so it is inert for any pool without it. `native/ptcg/test_stall.c` is a stall-greedy driver
asserting the cap binds, is airtight within a turn, and leaves zero truncations.

## 15. Parity tooling

The C env and the CUDA model are re-implementations, so each has a gate.
`native/tools/dump_stream.py` drives the Python env with random legal actions and records every raw
engine JSON string, every oracle payload, and the exact fp32 rows Python produced;
`native/ptcg/parity_replay.c` feeds the same strings through the C tracker/oracle/encoder and
compares rows bit-level. `native/tools/dump_stream_decks.py` writes the same stream for a chosen
deck glob, for mechanics only a few decks can trigger.

```bash
PYTHONPATH=data:. python3 native/tools/dump_stream.py 30
make -C native parity
PYTHONPATH=data:. python3 native/tools/dump_stream_decks.py \
    out_stream.jsonl out_rows.bin 'decks/pool/slowking*.csv' 40
```

`native/tools/model_parity.py` runs the same weights and observation batch through the torch
`PTCGTransformer` and through the native `_C.debug_forward`, comparing option scores where the mask
is 1, the value column, and the KL of the masked softmax — the number the PPO loss actually feels.
When it fails, `native/tools/stage_parity.py` localizes the divergence: the native side dumps every
named forward stage through `PTCG_DUMP_DIR`, the torch side captures the same tensors with module
hooks in a subprocess (both runtimes double-initialize the engine library in one process), and the
first stage over tolerance is where the bug lives. Both need the built extension, a GPU and strict
fp32, and both read their architecture from `native_weights.py`, so a gate cannot disagree with the
model it validates.

```bash
PTCG_TF32=0 PTCG_TABLES=native/ptcg_tables.bin PYTHONPATH=data:.:native \
python3 native/tools/model_parity.py \
    --rows native/parity_rows.bin --weights experiments/native-demo/init.bin
PTCG_TF32=0 PTCG_TABLES=native/ptcg_tables.bin PYTHONPATH=data:.:native \
python3 native/tools/stage_parity.py --weights experiments/native-demo/init.bin
```
