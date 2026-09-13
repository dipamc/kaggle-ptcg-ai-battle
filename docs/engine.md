# The competition engine

Every game in this repo is played by the official Pokemon TCG simulator that ships with the
Kaggle competition
([pokemon-tcg-ai-battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)). Nothing
here reimplements the rules: the engine is the rulebook, and where its behaviour differs from
the paper game, the engine is correct. The API reference lives at
<https://matsuoinstitute.github.io/cabt/>.

## Installing and loading the SDK

The SDK is not redistributed here. Download it from the competition Data tab and drop it in, as
`data/README.md` describes:

| path | what |
|---|---|
| `data/cg/` | `api.py`, `game.py`, `sim.py`, `utils.py`, `__init__.py` and the engine shared library |
| `data/EN_Card_Data.csv` | the card sheet (names, expansions, printed text) |

The shared library is picked by platform inside `cg/sim.py`: `libcg.so` on Linux x86_64,
`libcg-arm64.so` on Linux arm64, `libcg.dylib` on macOS.

Python code reaches `cg` because `ptcg/rl/__init__.py` prepends `<repo>/data` to `sys.path`, so
importing anything under `ptcg.rl` makes `from cg.api import ...` work. Scripts that are run
directly still set the path explicitly:

```bash
PYTHONPATH=data:. python3 tools/validate_obs_v4.py
```

The C/CUDA backend links the same library rather than going through ctypes. `native/Makefile`
selects `data/cg/libcg.dylib` or `-L data/cg -lcg`, and on macOS the run targets need
`DYLD_LIBRARY_PATH` set to `data/cg` because the library's install name is a bare `libcg.dylib`.

One derived file must be built before the tooling works:

```bash
PYTHONPATH=data python3 tools/build_carddb.py      # -> data/carddb.json
```

## The C entry points this code uses

`native/ptcg/tables.c` declares the engine symbols directly and wraps them; `ptcg/rl/battle.py`
calls the same symbols through `cg.sim.lib`. Between them that is the whole surface the trainer
needs.

| engine symbol | wrapper | purpose |
|---|---|---|
| `GameInitialize` | `pt_engine_init` | one-time engine init (guarded by `pthread_once`) |
| `BattleStart(int cards[120])` | `pt_battle_start`, `BattleHandle.__init__` | start a battle from `deck0 + deck1`; returns a battle pointer |
| `GetBattleData(ptr)` | `pt_battle_data`, `BattleHandle.obs` | the observation as a JSON document |
| `Select(ptr, int* idx, int n)` | `pt_battle_select`, `BattleHandle.select` | submit option indices; non-zero return is an error |
| `VisualizeData(ptr)` | `pt_battle_visualize`, `BattleHandle.visualize` | omniscient replay frames |
| `BattleFinish(ptr)` | `pt_battle_finish`, `BattleHandle.finish` | release the battle |

The search API (`AgentStart`, `SearchBegin`, `SearchStep`, `SearchEnd`) is used only by
`ptcg/rl/search_agent.py`, through the raw-dict wrappers `search_begin_raw` / `search_step_raw`
/ `search_end`. It runs on its own agent pointer, so a search coexists with a live battle.
`SearchBegin` consumes `obs["search_begin_input"]`, an opaque serialized state that must be
passed exactly as received.

`native/ptcg/obs.c` parses the `GetBattleData` JSON into C structs; its field names mirror the
dataclasses in `cg/api.py`, which is also where every enum lives — `AreaType`, `OptionType`,
`LogType`, `EnergyType`, `CardType` and the select type and context enums. New values may be
appended during a competition, so code here clamps indices rather than matching exhaustively.

## The agent contract

An agent is a module defining `agent(obs_dict) -> list[int]`.

- The first call of an episode has `obs["select"] is None`. Return your 60-card deck as a list
  of card ids. `ptcg/rl/arena.py` sends exactly this shape as `DECK_CALL`.
- Every later call returns indices into `obs["select"]["option"]`, with `minCount <= len(answer)
  <= maxCount`, all indices in range, no duplicates.
- An illegal answer, a crash, or exceeding the time budget loses the episode immediately. A
  legal but bad answer costs nothing but the game.
- The engine only ever offers legal moves, so no rules knowledge is needed to stay legal. There
  is no implicit pass: a `minCount == 0` select accepts `[]`, and the main menu carries an
  explicit END option.
- The budget is 600 s of decision time per agent for the whole game, reported in
  `obs["remainingOverageTime"]`. Inference runs CPU-only on roughly 1.6 vCPU, which is why
  `agent/main.py` calls `torch.set_num_threads(2)` and drops to zero-model-cost legal answers
  below 30 s remaining.
- Many calls are forced: when `minCount == maxCount == len(option)`, or there is one option and
  `minCount >= 1`, there is nothing to decide. `forced_answer` in `ptcg/rl/env.py` answers those
  without spending a decision.

`agent/main.py` is a working template of all of this.

## Rules the encoder and tracker rely on

| fact | consequence in this code |
|---|---|
| Seat 0 is asked whether to go first, before either hand is drawn; there is no coin flip | `GlobF.I_AM_FIRST` and `GlobF.FIRST_DECIDED` both exist, because `current["firstPlayer"]` is `-1` until the choice is made |
| Going first means no attack and no Supporter on turn 1 | the options simply do not appear; nothing special is encoded |
| The copy limit is per card NAME, not per card id, so the same card printed in two sets shares one 4-copy quota | deck builders and `tools/validate_deck_batch.py` treat decklists as fixed artifacts and check them against the engine's card DB |
| Prize liability is 1 for a regular Pokemon, 2 for an ex, 3 for a Mega ex | `ptcg/rl/cards.py` writes it as a static card feature, `(3 if megaEx else 2 if ex else 1) / 3` |
| Both players prizing out at once is a draw, not sudden death | `result == 2`; `ptcg/rl/arena.py` counts it separately from wins and losses |
| The engine advances the turn counter at `TURN_START`, not at turn end | `ptcg/effects.py` increments its replayed turn on log type 2 and then re-seeds from `current["turn"]` at every decision |
| One card can sit in "limbo" mid-resolution: it has left the hand and not yet reached the discard, so public zone totals sum to 59 | `InfoTracker.my_unseen` subtracts the resolving `select["effect"]` card when its serial is not visible anywhere, otherwise prize inference gains a phantom card |
| Cards being looked at are lifted out of `deckCount` but stay in the unseen pool | the encoder gives them their own `Zone.LOOKING` tokens from `select["deck"]` and `current["looking"]` |
| In replay/visualizer payloads, `entry[i].selected` is the answer given in state `entry[i-1]`, `entry[0].selected` is null, and the last action lands in a dummy terminal frame | shift by one when pairing states with actions; `OracleLedger` sidesteps it entirely by reading only the last frame's `current` |
| Replay payloads are omniscient (both hands, both decks, all prizes) while live observations are not | the omniscient view feeds only the critic-side `OracleLedger`, never `InfoTracker` |

## Threading, cost and reproducibility

- **Many battles per process.** Every C function takes the battle pointer explicitly, so
  `ptcg/rl/battle.py` keeps one pointer per `BattleHandle` and a single process can run hundreds
  of concurrent battles. The vectorized environment depends on this.
- **`cg.game` is one game per process.** The convenience module in the SDK stores the pointer in
  a module-level singleton. `ptcg/rl/arena.py` uses it (for `search_begin_input`), so an arena
  process holds exactly one game at a time, which is why arena-based evaluation scales by adding
  processes rather than by batching.
- **`VisualizeData` is expensive**: 26-38 ms per call, because it serializes the whole replay.
  `ptcg/rl/oracle.py` therefore parses it only while `turn == 0`, to pin the initial prize six,
  and maintains prizes from logs afterwards. Never call it per step.
- **An engine step costs roughly 0.15 ms.** Per game there are several hundred decision points,
  most of them forced, so a single core wraps on the order of a thousand decisions per second
  once the Python encoder is included.
- **Shuffles are not seedable from outside.** `BattleStart` takes 120 card ids and nothing else,
  so two runs with identical seeds on our side still deal different hands. Everything here is
  statistically reproducible, not bit-reproducible; compare distributions over many games, never
  single games.
