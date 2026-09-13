# Evaluating checkpoints and running games

Training produces weights; this document is about measuring them. Everything here is torch-side —
only the training loop is native.

## From native blobs to torch checkpoints

The native trainer writes flat fp32 weight blobs. Convert one to a torch checkpoint with:

```bash
PYTHONPATH=data:.:native python3 native/tools/native_weights.py import BLOB.bin model.pt
```

`native/tools/native_eval_watch.sh` automates that and optionally runs the battery on top:

```bash
STEPS_PER_EPOCH=65536 bash native/tools/native_eval_watch.sh <RUN> [GPU]
```

It starts a background converter loop that scans the run's `.bin` files every 60 s, names each output
`model_<epoch>.pt` and writes it via tmp-plus-rename so the watcher never reads a half-written file.
Then it launches `ptcg.rl.eval_watch` pointed at the converted directory.

| variable | effect |
|---|---|
| `STEPS_PER_EPOCH` | aggregate steps per epoch (`total_agents * horizon`); becomes the eval x-axis scale |
| `RANK_STEPS_PER_EPOCH` | divisor used to name epochs from step-stamped blobs, which count per-rank steps; under multi-GPU training these differ from `STEPS_PER_EPOCH` by the world size |
| `EVAL_HEADS` | attention heads for the eval model; defaults to `native_weights.HEADS`, the same source the converter and the parity gate use |
| `EVAL_BENCH` | `0` runs the converter only and skips the battery, for when the battery runs on another machine |
| `EVAL_EPOCH_MOD` | convert only epochs divisible by this, to thin the eval cadence; skipped blobs stay on disk for later |
| `EVAL_THREADS`, `EVAL_BENCH_GPUS`, `EVAL_WANDB_NAME`, `EVAL_EXTRA`, and `HF_REPO` with `HF_UPLOAD` set | forwarded to the battery |

The script takes a `flock` so two instances for one run cannot stack: each would run its own converter
*and* its own bench fan-out, doubling load and logging duplicate evals into the same wandb run.

## The eval battery

The exact battery the final run was scored on — the six anchor opponents, the five candidate
decks, the game counts, and the arena-backed driver that produced the final numbers — is in
[`battery/README.md`](../battery/README.md).

`ptcg/rl/eval_watch.py` watches a checkpoint directory and scores each new checkpoint. It measures
three things and writes them to `runs/eval_watch.jsonl` and, optionally, wandb.

**Versus random.** `--episodes` games against the random opponent, giving `eval/win_rate_vs_random`,
`eval/ep_len_random`, `eval/prize_margin_random`, `eval/turns_random`, plus the win-cause split
`eval/win_end_random_<cause>` and `eval/lose_end_random_<cause>` over the causes `prize`, `bench`,
`deck` and `other`. This is the objective progress metric that self-play statistics cannot give.

**Self-play statistics.** `--self-games` mirror games produce `eval/end_self_<cause>`,
`eval/turns_self`, `eval/ep_len_self` (both seats' decisions) and `eval/prize_margin_self`. A few
games from each mode are dumped as viewer-compatible replays.

**The bench matrix.** With `--bench-opponents` set, the candidate plays *every* deck in
`--bench-decks` against *every* opponent directory, policy-only, `--bench-games` games per cell. The
metrics are `<prefix>/win_rate_{deck}_vs_{opp}` per cell, `<prefix>/mean5_vs_{opp}` per opponent and
`<prefix>/macro_mean` over all cells, where `<prefix>` is `--bench-prefix`.

| flag | default | meaning |
|---|---|---|
| `--ckpt-dir` | `experiments/train-<run-name>` | directory to watch, scoped to one run |
| `--interval` | 120 | seconds between scans |
| `--once` | off | score one checkpoint and exit |
| `--episodes` / `--games` | 100 / 64 | vs-random episodes, and concurrent battles per env |
| `--self-games` / `--self-replays` / `--replays` | 24 / 3 / 3 | mirror games and replay dumps |
| `--bench-opponents` | `""` | comma-separated Kaggle-agent dirs; empty disables the matrix |
| `--bench-decks` | six pool decks | the candidate plays each of them |
| `--bench-games` | 16 | games per matrix cell |
| `--bench-prefix` | `eval` | wandb namespace for the matrix |
| `--bench-workers` | 0 | total bench processes |
| `--bench-device` | `cuda` | device for bench workers |
| `--bench-gpus` | `""` | cuda device ids to round-robin workers over |
| `--device` | `cpu` | device for the vs-random and self-play diagnostics |
| `--heads` | 8 | attention heads the checkpoint was trained with |
| `--steps-per-epoch` | 0 | when set, metrics carry a `timesteps` coordinate and wandb plots against it |
| `--pool-dir` | none | copy each scored checkpoint here |
| `--deckmeta-dir` / `--deckmeta-window` | none / 50000 | aggregate training deck logs into archetype movement metrics |
| `--wandb`, `--wandb-project`, `--run-name`, `--wandb-name` | | logging; the watcher resumes a stable run id on restart |
| `--hf-repo` | none | optional rolling backup, one commit per eval cycle |

`--gate-opponent` is accepted but emits no metric of its own; the per-opponent means serve as the
promotion signal.

**Parallelism rules.** `--bench-workers` is the only knob that turns fan-out on; it does not need
`--bench-gpus`. Each game is roughly half engine and half batch-1 forward, and one process can hold
exactly one game, because `cg.game` is a per-process singleton — so more *processes* is the lever, and
batching would need a shared inference server. On a dedicated eval machine use cores minus one; next
to training, stay well under the spare core count. `--bench-gpus` only round-robins
`CUDA_VISIBLE_DEVICES` across the workers; leave it empty on a single-GPU box and workers inherit the
parent's visibility. `--bench-device` is set per machine: `cuda` is faster per forward and frees cores
for engine work on a dedicated box, while `cpu` costs no VRAM next to a training job that already owns
it. Workers are subprocesses of the same module (`--bench-worker`, internal), each capped to one
thread, and results aggregate in the parent so there is a single wandb writer either way.

```bash
python -m ptcg.rl.eval_watch --once --ckpt-dir experiments/train-myrun --heads 8 \
    --bench-opponents opponents/a,opponents/b --bench-workers 12 --bench-device cpu
```

## Kaggle-format opponents

A fixed opponent is a directory in the competition's own agent format:

```
opponents/<name>/main.py     defines agent(obs_dict) -> list[int]
opponents/<name>/deck.csv    60 card ids, one per line
...                          anything else it needs (model, data, cg/)
```

`ptcg.rl.arena.load_kaggle_agent` exec-loads `main.py` the way the competition runner does: no
`__file__`, cwd switched to the directory so relative `deck.csv` reads work, and the last callable in
the module namespace taken as the agent. It also hides the `ptcg` package from `sys.modules` around
the exec, so a bundle that vendors its own copy imports that copy rather than the host process's, then
restores the host's afterwards.

Build opponents from your own checkpoints:

```bash
tools/make_opponents.sh model.pt v1 8 \
    marnie=marnie_s_grimmsnarl_ex_spikemuth_gym__05 \
    crustle=crustle_jumbo_ice_cream__01
```

The arguments are the checkpoint, a tag, the head count, and one or more `short=deck_name` pairs
naming decks in `decks/pool` (override with `POOL=`). Each pair produces `opponents/<tag>_<short>/`
containing `main.py`, `model.pt`, `deck.csv`, the vendored `ptcg/` modules, `cg/` and the `data/` files
the model needs. The script rewrites the `HEADS = <n>` line in the copied `main.py` and verifies the
stamp, then prints the `--bench-opponents` argument to paste into the battery.

That directory layout is also a valid competition submission: `agent/main.py` is the template it
copies. It plays policy-only and greedy (`SearchAgent` with `cfg=None`), guards against answer
rejection by tracking whether `obs["logs"]` advanced since an identical select signature, and switches
to zero-model-cost legal answers when the remaining overage time falls below 30 s. Any internal
failure degrades to a legal random answer, because an invalid answer or a crash forfeits the episode
while a merely bad answer does not.

## One-off arena runs

`ptcg.rl.arena` plays a checkpoint plus a deck against one opponent directory:

```bash
python -m ptcg.rl.arena --ckpt model.pt --deck decks/pool/crustle_jumbo_ice_cream__01.csv \
    --opponent opponents/v1_marnie --games 200 --heads 8 --device cpu --out results.jsonl
```

Seats alternate every game (`seat = i % 2`). Each line of `--out` is one game — result, cause, steps,
decisions, wall time — and the final line is a `{"summary": ...}` record. `--seed` fixes the run so
two invocations are comparable, and `--tag` labels the rows.

Play is policy-only unless you ask otherwise: `--search` builds a `SearchCfg` (for example
`--search "mode=turn,actions=6,dets=3,gate_gap=0.35"`), and omitting it leaves `cfg=None`, the pure
policy. The rest is off by default too — `--winline` and its `--wl-*` options run an engine-exact
prover for a win-this-turn line, `--vrank` enables value-head reranking, `--ens` averages softmax
probabilities across extra checkpoints, `--benchguard` plays a Basic instead of ending the turn at one
Pokemon in play, and `--temp` sets the sampling temperature. There is one game per process, so scale
by launching many processes with different seeds and aggregating the JSONL.

## Self-checkpoint deck evaluations

`tools/selfeval.py` holds the policy exactly constant and varies only the decks — one checkpoint
pilots both seats — which isolates deck strength from policy strength.

```bash
PYTHONPATH=data:. python3 tools/selfeval.py job.json 8
```

The job file is `{ckpt, cells, games, device, heads, temp, seed, out}`, where `cells` is a list of
`[my_deck, opp_deck]` path pairs. Seats alternate every game, so no cell can be biased by the
first-player advantage. The second argument is a worker count: cells are split round-robin across
forked workers with decorrelated seeds, so results are statistically equivalent to a serial run but
not bit-identical.

`tools/selfeval_batched.py` takes the same job format and the same game loop, and moves only the
model forward. W engine workers write encoded rows into shared memory and one server process holds the
single shared checkpoint and stacks every pending request into one forward; batches self-clock, since
whatever arrives during the previous forward becomes the next batch, so no latency is added. It also
shards each cell's games into even-sized chunks across workers, preserving seat alternation inside
every shard, so small probe jobs parallelize instead of serializing per cell.

Two rules the file encodes. Thread caps (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`) must be exported *before* numpy and torch are imported, because the pools are sized
at library load and forked children inherit that configuration; and `cg`/`ptcg` imports stay inside the
worker so each child loads the engine itself and gets independent C-side RNG state. Results are not
bit-identical to the serial runner, since the batched matmul reduction order differs — at `temp` 0 the
trajectories diverge only on float-tie argmax flips. Validate statistically, and keep one runner within
an experiment so the instrument does not change under you.

## Independent cross-check

`ptcg/rl/indep_eval.py` is a from-scratch re-implementation of the evaluation loop that deliberately
does not import `ptcg.rl.env`. It drives `BattleHandle` directly with its own forced-select handling,
multi-pick decomposition, random opponent and win-cause classifier, all computed from the raw engine
JSON. It shares with training only what defines the policy: `encode()`, `InfoTracker` and
`PTCGTransformer`. If the environment's row driver or statistics had a bug, its numbers would diverge
from the battery's.

```bash
python -m ptcg.rl.indep_eval --ckpt model.pt --games-self 100 --games-random 200
```

It builds the model with `PTCGTransformer()` defaults, so it measures 8-head checkpoints only.

## Greedy versus sampled play

`SearchAgent`'s `temp` controls real play: `temp <= 0` is a greedy argmax, anything positive samples
from the tempered softmax. The constructor default is 1.0 (sampling), which is what the arena's
`--temp` default and the battery's bench matrix use; the submission template (`agent/main.py`) and the
decklab runners play greedy with `temp=0.0`. Compare arms at one temperature only. Rollouts inside
search do not read this knob.

## Two traps

**The head count.** `arch_from_state_dict` cannot recover the number of attention heads, because the
qkv projection is `d x d` at any head count. A wrong value loads without error and computes different
attention, so every tool takes it explicitly: `--heads` for the arena and the battery, `heads` in a
selfeval job, `EVAL_HEADS` for the watcher, and the third positional argument to
`tools/make_opponents.sh`. A cheap guard is to include one obviously weak opponent in every
comparison — both arms should crush it, and an arm that does not is usually mis-headed.

**Deck identity.** Deck names are long, similar and easy to transpose, and two files with the same
suffix can be entirely different lists. Identify decks by md5, not by name — `tools/validate_deck_batch.py`
does exactly that when checking a batch — and confirm that the deck in a submission bundle is the deck
you measured. The battery compounds this: its metric keys use `basename.split("__")[0][:24]`, so two
variants of one archetype collide into a single series. Change `--bench-prefix` whenever the deck or
opponent set changes, since series are keyed by deck and opponent name and a new set otherwise starts
new keys silently beside the old ones.
