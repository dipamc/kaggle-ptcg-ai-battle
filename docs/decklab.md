# Decklab

Decklab proposes few-card variants of pool decks from measured evidence, refutes them against card
text, A/Bs them with a **fixed converged checkpoint piloting both seats**, and stages the survivors
for insertion into the training pool (`docs/deck-pool.md` §5). Because one policy plays both sides
and only the decks differ, a measured delta is attributable to the cards rather than to the pilot.

This document is the **mechanical layer**: files, tools, CLIs, and the statistical contract they
enforce. The cognitive layer — how proposals are generated and adversarially reviewed, and how an
orchestrating agent drives a cycle — lives in `AGENTS.md`.

## 1. Prerequisites

* **`data/carddb.json`** — engine card data merged with the card sheet, the ground truth every tool
  reads card text from. Build it with `PYTHONPATH=data python3 tools/build_carddb.py` once
  `data/cg/` and `data/EN_Card_Data.csv` are in place (see `data/README.md`).
* **A checkpoint** (`.pt`) for the evaluator plus its **head count**. The head count is not
  recoverable from a state dict, so it travels with the checkpoint in the config; a wrong value
  loads silently and yields a plausible-looking but meaningless instrument.
* **A deck matrix** from a training run — the `<run_dir>/deckmat/` directory written by
  `--deck-matrix-every`: every pool deck against every other at training volume.
* **`decklab/config.json`**, which names all of the above.

### Configuration keys

| key | meaning |
|---|---|
| `arena` | the confirmatory A/B lane — the frozen instrument |
| `arena_probe` | optional exploratory lane; falls back to `arena` when absent. `<lane>` below means either block |
| `<lane>.host` | `"local"` (games run in this checkout with this interpreter) or an ssh host |
| `<lane>.runner` | `batched` (`tools/selfeval_batched.py`) or `serial` (`tools/selfeval.py`) |
| `<lane>.workers_ab`, `<lane>.workers_probe` | worker processes for that lane's jobs; `<lane>.device` is `cpu` or `cuda` for its model forward |
| `<lane>.python`, `<lane>.repo`, `<lane>.arena_dir` | remote lanes only: interpreter, checkpoint-relative root, mirrored checkout. Local lanes fill these in themselves |
| `arena_probe.ckpt`, `arena_probe.heads` | a lane may pilot a different checkpoint than the A/B evaluator |
| `ckpt_primary`, `heads` | the evaluator checkpoint and its head count; the primary A/B arm |
| `ckpt_replicate` | second checkpoint, used by `--replicate` |
| `device`, `temp` | defaults for both lanes; `temp: 0.0` is greedy play |
| `games_targeted_per_opp`, `games_field_per_opp` | games per opponent per arm, stage 1 and stage 2 |
| `field_gauntlet` | path to the frozen stage-2 gauntlet file |
| `max_swaps`, `replication_required` | maximum card swaps a proposal may make from its base deck; whether acceptance needs both checkpoints to pass |
| `accept.targeted_min_delta`, `accept.targeted_min_z` | stage-1 bars: minimum effect size, and minimum effect in standard errors |
| `accept.field_noninferiority_z` | stage-2 non-inferiority margin, in standard errors |
| `screen` | a faster profile: its own `games_*` and `field_gauntlet`, same accept bars |
| `matrices` | named matrix sources, `{key: {paths, names, epoch_min}}`: repo-relative dirs of `r<rank>_e<epoch>_<seq>.bin` files; the pinned deck listing the matrix was written against (`null` = the live glob of `decks/pool`); and a lower epoch bound applied on load |
| `default_matrix` | which key the tools use when `--matrix` / `--which` is omitted |
| `evaluator_pool_names` | listing of the decks the evaluator trained on; `null` means the whole current pool |

`evaluator_pool_names` defines the **evaluator vocabulary**: the union of card ids in those
decklists. Cards outside it were never piloted by the evaluator, which changes how a proposal using
them is judged (§4).

## 2. Evidence inputs

**Ladder field cache** (optional): `decklab/packs/field_cache.json` says how often each pool deck is
played by the opponents you care about, plus each deck's own record.

```json
{"total_sides": 12000,
 "decks": {"<pool deck name>": {"sides": 430, "lad_n": 210, "lad_w": 119}},
 "built": "<timestamp>", "window": "<description>"}
```

Build it from whatever ladder data you have, joining decklists to `decks/pool` by exact multiset.
Without the file, `decklab_common.field_cache` prints a note and assumes a **uniform field** over
the pool, so every field-weighted number reduces to a uniform-field number.

**Field gauntlets**: `decklab/field30.txt` and `decklab/field15.txt` are frozen stage-2 opponent
sets, one `<share> <deck name>` pair per line with `#` comments; the share is the weight in the
field-weighted delta and the name must be a stem in `decks/pool`. `field30.txt` is the default,
`field15.txt` the `screen` profile's shorter set. They are frozen on purpose — changing the
gauntlet changes the instrument.

**Implied ranking** (optional): `data/deck_implied.csv`, columns `deck`, `rank_implied`,
`implied_wr`, `unrankable_note`, from whatever ladder analysis you run; every consumer tolerates
its absence.

## 3. The tools

### Query and evidence packs

`tools/dlab.py` is the one-shot query CLI. Deck arguments resolve against `decks/pool` stems (a
unique substring is enough), a `.csv` path, or a proposal id; card arguments resolve against card
names.

```
tools/dlab.py deck <name|path|dlab_NNNN> [--full]   list, field/implied standing, OOV flags, notes
tools/dlab.py diff <a> <b>                          multiset swap diff
tools/dlab.py near <deck> [--k 8]                   closest pool decks
tools/dlab.py card <query> [--brief] [--all]        text, vocab flag, pool usage, notes
tools/dlab.py find <query>                          decks and cards matching
tools/dlab.py note deck|card <key> <text> [--by X]  append a lab note (<= 240 chars)
tools/dlab.py notes [query]                         browse notes
tools/dlab.py history <query>                       ledger entries and proposals mentioning it
tools/dlab.py mat cell <a> <b> [--which <key>]      one matchup, both seats
tools/dlab.py mat row <deck> [--best] [--n 12]      worst/best matchups with field share
tools/dlab.py mat rank <deck>                       raw and implied rank
tools/dlab.py mat check [--which <key>]             matrix vs decklists-on-disk sync report
```

`tools/decklab_pack.py` writes heavyweight evidence packs into `decklab/packs/`. `field [--top N]`
ranks the field by share with each deck's ladder record and implied rank. `swapstats` tabulates
card usage across the field, weighted by share. `deck --name <pool deck> [--matrix <key>]
[--cohort-dist N]` is the main dossier: composition, standing, matchup profile by data-driven
cluster with each cluster's field weight, the nearest pool decks and what they run differently, and
the relevant single-swap natural experiments. `cards --deck <csv> [--extra 12,34]` dumps
engine-exact text for every card in a deck plus any extras.

`tools/decklab_pairs.py [--matrix <key>] [--max-swaps 2] [--min-cell 150] [--cut 0.45]` mines the
free experiments already in the matrix: every pool pair differing by at most `--max-swaps` cards,
with the win-rate delta overall and per matchup class. Each side of such a pair already has
thousands of games, so a one-card pair is a natural experiment at training volume. Writes
`decklab/packs/pairs_<matrix>.csv` and prints the significant single-swap findings.

### Proposals and validation

A proposal is a directory `decklab/proposals/<id>/` holding `deck.csv` (60 card ids, one per line,
the `decks/pool` format) and `meta.json`, the **pre-registration**, written before any games run:

```json
{
  "id": "dlab_0001_<short slug>",
  "base_deck": "decks/pool/....csv",
  "rationale": "why these swaps (cites evidence pack lines)",
  "target_opponents": ["decks/pool/a.csv", "..."],
  "predicted": {"targeted": "+", "field": "0"},
  "status": "proposed"
}
```

`tools/decklab_validate.py <proposal dir>...` is the gate. It **fails** on: not 60 cards; unknown
card ids; more than 4 copies of a card name (basic energy exempt); more than 1 ACE SPEC; zero swaps
or more than `max_swaps` swaps from the base; a list identical to a pool deck or to a live prior
proposal; fewer than 3 or more than 6 `target_opponents`, or one that does not exist; a missing
`meta.json` field. It **warns only** on an evolution line without its base, because that rule
miscalls real decks. On success it rewrites `meta.json` with `status: "validated"` and a measured
swap summary. It also stamps the **evidence class**: `measured` when every card is in the evaluator
vocabulary, `reasoning` when any card is outside it, with the offenders in `meta.oov_cards` (§4).

### The A/B harness

```bash
tools/decklab_ab.py setup-arena                       # remote lanes only
tools/decklab_ab.py run <proposal dir> [--replicate] [--stage both|targeted|field]
                       [--games-scale 1.0] [--profile screen]
tools/decklab_ab.py probe --deck <proposal dir|deck.csv> [--opps a.csv,b.csv]
                       [--games 60] [--vs-base] [--pilot lane|frozen]
tools/decklab_ab.py verdict <proposal dir>
```

A **local lane** runs games in this checkout with this interpreter, writing jobs to `decklab/jobs/`
and staged decks to `decklab/staging/`. A **remote lane** is an ssh host whose `arena_dir` mirrors
the checkout: `setup-arena` rsyncs `ptcg/`, `data/`, `decks/pool/` and the two runners onto it
(checkpoints are never synced — place them under the lane's `repo` yourself), and each job is
shipped over, run under `flock`, and rsynced back. Never split one experiment's two arms across
lanes; the lane is part of the instrument.

`run` stages the proposal deck, builds the cell list (`target_opponents` at
`games_targeted_per_opp`, the gauntlet at `games_field_per_opp`), groups cells by game count into
jobs, and runs them. Base-arm cells are cached in `decklab/results/base_cache.jsonl` keyed by
checkpoint, base-deck md5, opponent, games, temperature, heads and lane — so five variants of one
base pay the base cost once, and a base cell from one lane can never pair with a proposal cell from
another. Results land in `decklab/results/<id>/ab_<pri|rep>.json` and `run` calls `verdict` itself;
a partial `--stage` rerun keeps the other stage and replaces its own.

`probe` plays exploratory games on the probe lane, optionally against the base deck too
(`--vs-base`), and records a `probe` ledger event; `--pilot frozen` pilots with `ckpt_primary`
rather than the lane's own checkpoint so a probe is comparable with earlier probes. Probes are
never acceptance evidence (§4).

`verdict` recomputes the decision from the stored results and writes the status into `meta.json`.
Per opponent cell it takes the arm difference `p_prop - p_base` with the binomial variance of both
arms; stage 1 averages cells equally, stage 2 weights them by field share. Stage 1 passes when the
delta, signed by the pre-registered direction, is at least `max(accept.targeted_min_delta,
accept.targeted_min_z * SE)`. Stage 2 passes on non-inferiority: the weighted delta is at least
`-accept.field_noninferiority_z * SE`. Both stages must pass on `ckpt_primary`; with
`replication_required` the `--replicate` arm must pass too, otherwise a primary-only pass is
recorded as `tested_pass`.

### Staging and post-append scoring

`tools/decklab_batch.py [--batch N] [--out <dir>] [--dry-run]` collects every accepted proposal not
yet staged, copies its `deck.csv` as `zz<NN>_<base stem>__<proposal id>.csv` — a name that sorts
after every existing pool name, so ids stay append-only — into `decklab/staged/vNNN/`, records
`staged_as` in `meta.json` plus a `staged` ledger event, runs `tools/validate_deck_batch.py` over
the result, and prints the manual append steps. The append is deliberately not automated: it
touches a live trainer (`docs/deck-pool.md` §5).

`tools/decklab_score.py --since <epoch> [--matrix <key>]` is the measurement afterwards. For every
staged proposal it reads the live deck matrix from that epoch onward and compares the child's
overall win rate with its parent's over the same field, with binomial intervals, flagging any child
below 0.15 over at least 2000 games as an anomaly worth a zero sampling weight. Each scored deck
appends a `scored` ledger event.

### The game runners

Both runners take the same job JSON and produce the same output rows:

```json
{"ckpt": "<abs path>", "cells": [["my_deck.csv", "opp_deck.csv"], "..."],
 "games": 200, "device": "cpu", "heads": 8, "temp": 0.0, "seed": 12345,
 "out": "decklab/jobs/<name>.out.json"}
```

```bash
PYTHONPATH=data:. python3 tools/selfeval.py job.json [n_workers]
PYTHONPATH=data:. python3 tools/selfeval_batched.py job.json [n_workers]
```

One checkpoint pilots both seats and seats alternate every game, so first-player advantage cannot
bias a cell. `selfeval.py` forks one process per worker and splits cells round-robin.
`selfeval_batched.py` keeps the identical game loop and decision semantics and moves only the model
forward: engine workers write encoded rows into shared memory, and one server process holds the
single shared checkpoint and stacks every pending request into one forward. Batches self-clock —
whatever arrives during the previous forward is the next batch — so there is no wait window, and a
cell's games are sharded even-sized across workers so even a small probe parallelizes. It caps
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` **before** importing numpy and
torch, because those libraries size their pools at load time and every forked child inherits the
parent's configuration.

Neither runner is bit-reproducible against the other — worker seeds are decorrelated and the
batched matmul reduction order differs — so results are **statistically equivalent, not identical**.
A dead worker leaves its cells missing from the output, which is why `decklab_ab.py` counts
returned cells and aborts rather than recording a partial measurement.

## 4. The methodology contract

**Two stages, both required.** Stage 1 measures the effect where it is predicted, on pre-committed
opponents; stage 2 checks the deck did not pay for it elsewhere. Either alone misleads in a
different direction: targeted-only sells a card that is a field-wide loss, field-only discards a
card worth a lot in exactly the matchup it was chosen for.

**Pre-registration is the multiple-testing guard.** `target_opponents` and `predicted` are frozen
at validation time, before any confirmatory game runs; editing them after seeing results makes a
new proposal id. Targeted deltas carry a standard error of a few points, so the maximum of many
null tests lands near the size of a real effect — the pre-committed gauntlet is the guard, and
`--replicate` on a second checkpoint is the stronger one.

**Detection floor.** An effect around +0.17 is resolvable at ~2000 games; around +0.05 needs roughly
ten times that. Most swaps return "no measurable effect at this n", which is **not** evidence of
neutrality, and is recorded as null rather than re-rolled for a better seed.

**Probes are hypothesis generation.** A probe may motivate a rationale and is ledger-recorded, but
it is never acceptance evidence and never replaces the confirmatory A/B — fresh games under the
frozen protocol.

**Evaluator-OOV cards invert the epistemics.** The frozen evaluator never piloted cards outside its
vocabulary, so its games — probes included — carry no information about them. The validator stamps
such proposals `evidence_class: "reasoning"` and `decklab_ab.py run` refuses them by design. The
gate is instead an adversarial review of the first-principles case against exact card text, ideally
with engine-level mechanics verification. This is the acquisition route: the deck is appended so
the live run *learns* to pilot the new cards, and `decklab_score.py` a few hundred epochs later is
the measurement that counts.

**Co-evolution.** Every delta is measured against one policy's play. Accepted decks feed the next
training pool and the field shifts, so a card whose value evaporates once the policy trains on it
was never a deck improvement; post-append scoring is what makes that re-check concrete.

## 5. State on disk

| path | what |
|---|---|
| `decklab/config.json` | lanes, checkpoints, game counts, accept bars, matrix sources |
| `decklab/field30.txt`, `decklab/field15.txt` | frozen field gauntlets |
| `decklab/proposals/<id>/` | `deck.csv` + `meta.json` per proposal |
| `decklab/results/<id>/ab_pri.json`, `ab_rep.json` | raw per-cell A/B results, per arm |
| `decklab/results/base_cache.jsonl`, `decklab/packs/` | cached base-arm cells; generated evidence packs and the field cache |
| `decklab/jobs/`, `decklab/staging/` | lane working directories for jobs and staged decks |
| `decklab/staged/vNNN/` | accepted decks renamed and validated for an append batch |
| `decklab/LEDGER.jsonl` | append-only event log |
| `decklab/notes.jsonl` | short lab notes |

The ledger holds one JSON object per line, appended through `decklab_common.ledger_append` and
timestamped automatically. Event kinds are `probe` (exploratory games and results), `verdict` (the
stage statistics and resulting status), `dropped_review` (a proposal rejected on review, with a
reason), `staged` (a proposal copied into a batch under its new name) and `scored` (post-append
child-vs-parent win rates). Being append-only and line-atomic, it lets several cycles run at once.

Notes (at most 240 characters each) are the lab's institutional memory: dose maps, engine-proven
mechanics, "this swap already failed on that base". Every `dlab.py deck` / `card` fetch and every
dossier surfaces the relevant ones. The working state — `packs/`, `jobs/`, `staging/`,
`staged/`, `base_cache.jsonl`, `LEDGER.jsonl`, `notes.jsonl` — is gitignored; the `proposals/`
and `results/` directories are tracked.

## 6. A full cycle on the local lane

```bash
# 0. one-time: card database, plus a config naming a checkpoint and a deckmat dir
PYTHONPATH=data python3 tools/build_carddb.py
# 1. refresh the evidence
python3 tools/decklab_pack.py field
python3 tools/decklab_pack.py swapstats
python3 tools/decklab_pairs.py
python3 tools/decklab_pack.py deck --name crustle_jumbo_ice_cream__01
python3 tools/decklab_pack.py cards --deck decks/pool/crustle_jumbo_ice_cream__01.csv
# 2. pick the targeted gauntlet from the worst matchups that carry real field share
python3 tools/dlab.py mat row crustle_jumbo_ice_cream__01 --n 12
python3 tools/dlab.py history "<card under consideration>"
# 3. write the proposal: deck.csv + a pre-registered meta.json
python3 -c "import sys;sys.path.insert(0,'tools');import decklab_common as C;print(C.next_proposal_id())"
P=decklab/proposals/dlab_0001_crustle_01_draw
mkdir -p $P && $EDITOR $P/deck.csv && $EDITOR $P/meta.json
# 4. validate: legality, swap budget, pre-registration, evidence class
python3 tools/decklab_validate.py $P
# 5. optional exploratory probe - informational only
python3 tools/decklab_ab.py probe --deck $P --games 60 --vs-base
# 6. the confirmatory two-stage A/B; run prints the verdict, verdict recomputes it
python3 tools/decklab_ab.py run $P
python3 tools/decklab_ab.py verdict $P
# 7. record what was learned
python3 tools/dlab.py note deck crustle_jumbo_ice_cream__01 "<one durable fact>" --by dlab_0001
# 8. stage the accepted proposals and validate the batch
python3 tools/decklab_batch.py --batch 3 --dry-run
python3 tools/decklab_batch.py --batch 3
# 9. append to the live pool by hand (docs/deck-pool.md section 5), then measure
python3 tools/decklab_score.py --since <epoch the append landed>
```

Step 6 is skipped for a `reasoning`-class proposal: `run` refuses it, review is the gate, and step 9
is the measurement. Step 8 stages `accepted`, `accepted_reasoning` and `accepted_user` alike.
