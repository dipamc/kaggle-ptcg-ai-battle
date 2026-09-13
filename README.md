# Pokémon TCG AI Battle: native RL trainer and deck lab

Code release for the Kaggle competition *The Pokémon Company - PTCG AI
Battle Challenge*, both the
[Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
division (the agent; submission 55559155) and the
[Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)
hackathon, by **Dipam Chakraborty, Team Unown Gradiant**. The write-up:
[Team Unown Gradiant solution: all decks on hand](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/writeups/team-unown-gradiant-solution-all-decks-on-hand).

The agent is a token-transformer policy trained by self-play PPO on a
no-PyTorch training stack: the game environment is written in C against the
competition's simulator, the policy network is implemented in CUDA, and both
are linked into a patched copy of PufferLib's native CUDA trainer. Decks are
sampled from a pool of 616 competitive lists that can grow and be re-weighted
while a run is live, and a separate "deck lab" harness lets agents propose,
review, and A/B-test deck variants against a fixed checkpoint before they are
appended to the training pool.

Included:

- `native/` — the trainer: C env, CUDA model, vendored + patched PufferLib, build scripts, launch/supervisor tooling
- `ptcg/` — the Python runtime shared by training, evaluation and the Kaggle agent: observation encoder, information tracker, torch model, policy agent
- `tools/` — deck-pool tooling, deck-matrix analysis, the eval/arena runners, and the decklab harness
- `decks/pool`, `decks/pool_alt` — the training pool (616 decks) and the coverage pool (87 decks)
- `decklab/` — decklab configuration and field gauntlets
- `agent/` — the Kaggle-format agent template used to build eval opponents and submissions
- `battery/` — the evaluation battery: the anchor and candidate deck lists the final run was scored on, and the arena-backed driver
- `submission/` — the exact final submission (55559155): agent code, weights, deck
- `docs/` — how everything works

Not included: the competition simulator and card sheet (fetch them from
Kaggle, see below).

## Layout

| path | what |
|---|---|
| `native/ptcg/` | C game env: engine driver, observation encoder, tracker, tables, event log, deck pool hot-swap; `test_*.c` contract tests |
| `native/model/` | the policy network in CUDA (forward + backward), compile-time size macros |
| `native/src/` | PufferLib CUDA trainer at commit c5d3c63 with local patches (`native/patches/`, `native/PATCHES.md`) |
| `native/train_native.py` | training driver: flags, resume, league banks, deck pool / weight hot updates, telemetry |
| `native/tools/` | table export, weight conversion, parity checks, supervisor, eval watcher |
| `ptcg/rl/` | `buffers.py` (observation layout), `encoder.py`, `model.py`, `search_agent.py` (the agent), `env.py` (PufferLib 3.0 env), `eval_watch.py` (eval battery), `arena.py` |
| `ptcg/tracker.py`, `ptcg/effects.py` | hidden-information tracker and lingering-effect model |
| `tools/decklab_*.py`, `tools/dlab.py`, `tools/selfeval*.py` | the deck lab |
| `tools/deck_*.py`, `tools/validate_deck_batch.py`, `tools/pool_names.py` | deck pool and deck-matrix tooling |
| `battery/` | `run_battery.sh`, `aggregate.py`, `opponent_decks/` (six anchors), `candidate_decks/` (five candidates); configuration on record in `battery/README.md` |
| `data/` | shipped static tables; where the Kaggle SDK and card sheet go |
| `AGENTS.md` | instructions for coding agents working in this repo |

## Setup

1. Clone, then fetch the competition data from the
   [Data tab](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data):
   the simulator SDK directory goes to `data/cg/` and the card sheet to
   `data/EN_Card_Data.csv`. Details in [data/README.md](data/README.md).
2. Install the pinned dependencies and run the smoke test:

   ```bash
   bash setup/bootstrap.sh
   ```

   This installs `setup/requirements.txt` (torch, numpy<2, pufferlib 3.0,
   pybind11, wandb), checks the engine loads, and builds `data/carddb.json`.
   On a host whose nvcc is CUDA 12.x, install torch from the cu128 index
   before running it (pufferlib builds a torch CUDA extension that must
   match nvcc; the recipe is at the top of `setup/requirements.txt`).
3. Build the tables blob the trainer and the C env read:

   ```bash
   PYTHONPATH=data:. python3 native/tools/export_tables.py native/ptcg_tables.bin \
       --pool decks/pool --alt-pool decks/pool_alt
   ```

4. Build and test the C env (works on Linux and macOS, no GPU needed):

   ```bash
   make -C native test
   ```

5. On a machine with an NVIDIA GPU, nvcc and NCCL, build the trainer:

   ```bash
   bash native/build_native.sh          # -> native/puffer_ptcg/_C*.so
   ```

## Train

Single GPU:

```bash
PYTHONPATH=data:.:native PTCG_TABLES=native/ptcg_tables.bin \
python3 native/train_native.py --run-name run1 --deck-pool decks/pool \
    --gpus 1 --total-agents 1024 --minibatch 512 --timesteps 1000000000 \
    --deck-matrix-every 20000 --checkpoint-interval 50
```

Eight GPUs: `--gpus 8 --total-agents 1024` (agents and minibatch are per
rank). Runs resume automatically from `experiments/native-<run>/state_latest.bin`.
For unattended runs put the flags in `runs/native_<run>.args` and use
`native/tools/supervisor_native.sh <run>`, which relaunches on crash or hang,
aborts on divergence, and keeps the eval watcher alive.

Every flag, the telemetry, the divergence guards, league training, the deck
matrix and the event log are described in [docs/training.md](docs/training.md).
The deck pool, the deck-id rule, mid-run appends and sampling weights are in
[docs/deck-pool.md](docs/deck-pool.md).

## Evaluate

Checkpoints are packed fp32 blobs; convert one to a torch state dict and play
it:

```bash
PYTHONPATH=data:.:native python3 native/tools/native_weights.py import \
    experiments/native-run1/0000000065536000.bin model.pt
PYTHONPATH=data:. python3 -m ptcg.rl.arena --ckpt model.pt \
    --deck decks/pool/dragapult_ex_crushing_hammer__05.csv \
    --opponent opponents/<agent-dir> --games 50 --heads 8 --temp 0 --device cpu --out out.jsonl
```

`native/tools/native_eval_watch.sh` converts every new checkpoint and runs the
battery in `ptcg/rl/eval_watch.py`; `tools/make_opponents.sh` turns any
checkpoint into Kaggle-format opponent directories. See
[docs/eval.md](docs/eval.md).

## Deck lab

```bash
PYTHONPATH=data python3 tools/build_carddb.py                # card database
python3 tools/decklab_pack.py deck --name dragapult_ex_crushing_hammer__05
python3 tools/dlab.py card Judge
python3 tools/decklab_validate.py decklab/proposals/<proposal-id>
python3 tools/decklab_ab.py run decklab/proposals/<proposal-id>
```

The lab reads a deck-vs-deck matrix written by a training run
(`--deck-matrix-every`) and a converged checkpoint configured in
`decklab/config.json`. The mechanics are in [docs/decklab.md](docs/decklab.md);
the agent workflow that drives it is in [AGENTS.md](AGENTS.md).

## Documentation

| doc | contents |
|---|---|
| [docs/training.md](docs/training.md) | native trainer: build, flags, multi-GPU, checkpoints, telemetry, guards, league, event log, supervisor |
| [docs/deck-pool.md](docs/deck-pool.md) | deck files, deck ids, tables blob, hot append, sampling weights, matrix analysis |
| [docs/decklab.md](docs/decklab.md) | the deck lab harness and its methodology |
| [docs/model.md](docs/model.md) | observation layout, tracker, network, weight formats, rewards |
| [docs/eval.md](docs/eval.md) | eval battery, arena, self-eval runners, opponents |
| [docs/engine.md](docs/engine.md) | working with the competition simulator |
| [native/PATCHES.md](native/PATCHES.md) | the local patches on top of PufferLib |

## Acknowledgements

The trainer builds on [PufferLib](https://github.com/PufferAI/PufferLib)
(MIT, see `native/src/PUFFERLIB_LICENSE`); the vendored copy is commit
c5d3c63 plus the patches in `native/patches/`. The C env parses engine
payloads with [cJSON](https://github.com/DaveGamble/cJSON) (MIT, notice in
`native/vendor/cJSON.c`). The game simulator is the competition's `cabt`
engine, which is not redistributed here.

## License

This repository is released under the MIT License (see [LICENSE](LICENSE)),
the license the competition rules require of winning submissions. Vendored
third-party code keeps its own MIT notices (PufferLib, cJSON). The
competition simulator and card sheet are not part of this release and remain
subject to the competition rules.
