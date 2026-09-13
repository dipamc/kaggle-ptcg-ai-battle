# Working in this repository (for coding agents)

Read this file first, then the doc for the area you are touching:
`docs/training.md` (native trainer), `docs/deck-pool.md` (decks, tables blob,
hot updates), `docs/decklab.md` (deck lab), `docs/model.md` (observation and
network), `docs/eval.md` (playing and scoring checkpoints), `docs/engine.md`
(the simulator).

## What is where

| area | code | tests / checks |
|---|---|---|
| C game env | `native/ptcg/*.c`, `native/ptcg/ptcg_env.h` | `make -C native test`, `reload`, `decklock`, `deckweights`, `parity` |
| CUDA model | `native/model/*.cu`, `*.cuh` | `native/tools/model_parity.py`, `stage_parity.py` (need a GPU) |
| trainer core | `native/src/*.cu` (patched PufferLib), `native/train_native.py` | a short `--timesteps` run on a GPU |
| Python runtime | `ptcg/rl/buffers.py`, `encoder.py`, `model.py`, `search_agent.py`, `ptcg/tracker.py`, `ptcg/effects.py` | `tools/validate_obs_v4.py [games]`, `make -C native parity` |
| eval | `ptcg/rl/eval_watch.py`, `ptcg/rl/arena.py`, `tools/selfeval*.py`, `battery/` | run a few games; `battery/run_battery.sh ... --games 4 --chunks 1` |
| deck lab | `tools/decklab_*.py`, `tools/dlab.py`, `decklab/config.json` | the commands in `docs/decklab.md` |
| deck pool | `decks/pool`, `decks/pool_alt`, `tools/validate_deck_batch.py`, `tools/pool_names.py`, `native/tools/export_tables.py` | `validate_deck_batch.py`, `make -C native reload` |

Prerequisites before anything runs: the competition SDK at `data/cg/` and the
card sheet at `data/EN_Card_Data.csv` (see `data/README.md`), then
`bash setup/bootstrap.sh`. Python commands are run from the repo root with
`PYTHONPATH=data:.` (add `:native` for the trainer and the weight tools).

## Invariants that must not be broken

1. **Deck id = position in the sorted glob of 60-line `.csv` files in the pool
   directory.** Every tool derives ids this way. Do not rename, insert or
   delete files in `decks/pool` for a pool that an existing run, checkpoint,
   or deck matrix was built from. Decks added mid-run must have names that sort
   after every existing name (the `zz<nn>_` prefix convention) and go through
   `tools/validate_deck_batch.py`.
2. **The tables blob (`native/ptcg_tables*.bin`) is an artifact of a run, not a
   build product.** Re-exporting it from a changed pool renumbers deck ids.
   Keep the blob a run was started with next to that run; grown blobs are
   produced with `export_tables.py --append-to <pinned listing>` only.
3. **The model size lives in two places**: `PT_D / PT_H / PT_FFN / PT_LAYERS`
   in `native/model/ptcg_model.cu` and `D, HEADS, FFN, LAYERS` in
   `native/tools/native_weights.py`. Change both in the same commit and run
   `model_parity.py` afterwards. A mismatch loads cleanly and computes garbage.
4. **The attention head count cannot be recovered from a state dict.** Every
   loader takes `--heads` (or `HEADS =` in an agent bundle); a wrong value
   loads silently. Always pass it explicitly.
5. **The observation layout is frozen by `ptcg/rl/buffers.py`.** Any change
   to it invalidates every checkpoint, the C encoder (`native/ptcg/encoder.c`),
   the CUDA model's offsets, and the tables blob. If you must change it: bump
   `VERSION`, update the C and CUDA sides, re-export the blob, run
   `tools/validate_obs_v4.py`, `make -C native parity` and `model_parity.py`.
6. **fp32 only in the native backend.** Card and attack ids travel inside the
   observation as floats; bf16 corrupts ids above 256.
7. **Killing a multi-GPU run takes two kills.** The rank-1+ workers are
   `multiprocessing.spawn` children whose command line does not contain
   `train_native.py`; find them with `nvidia-smi --query-compute-apps=pid` and
   kill by pid. Never `pkill -f` a pattern that matches your own shell.
8. **Never hand-edit a live run's state files** (`experiments/native-<run>/pool_version`,
   `state_latest.bin`). Pool and weight changes go through the request-file
   protocol in `docs/deck-pool.md`; versions only go up; a rejected version is
   never retried under the same number.
9. **Comments describe mechanisms, not history.** Do not add dates, run names,
   machine names, or narrative about past incidents to code or docs.

## Standard workflows

**After editing the C env**: `make -C native test`; if you touched
`tables.c`, `env.c` deck handling or `ptcg_env.h`, also `make -C native reload`,
`decklock` and `deckweights` (see `native/README.md` for the blob arguments);
then rebuild `_C` with `bash native/build_native.sh` on the GPU machine.

**After editing the CUDA model or the Python model**: `model_parity.py`
(strict fp32: `PTCG_TF32=0`) must report argmax agreement of 100% and a KL
near 1e-16 against the torch model.

**After editing the encoder or tracker (Python)**: `tools/validate_obs_v4.py`
and `make -C native parity` (the C encoder must match bit for bit).

**Launching a run**: write `runs/native_<run>.args` (one flag per line, `#`
comments allowed) and `runs/native_<run>.env` (`TRAIN_GPUS`, `EVAL_GPU`,
`PTCG_TABLES`, `PY`), then `setsid nohup bash native/tools/supervisor_native.sh <run> > runs/native_supervisor.log 2>&1 < /dev/null &`.
Confirm with `pgrep -af train_native` and the first `epoch ... SPS ...` log
line. Never launch a benchmark with a live run's `.args` unedited (it carries
`--wandb-name`).

**Reading a run**: the epoch line prints per-rank steps; aggregate steps are
`epoch x total_agents x 64 x gpus`. Healthy: `win_vs_rand` near 1.0, `ent`
around 0.5, `kl` around 0.02-0.03, `vf` around 0.015, `vfskip` 0. A rising
`vf` or `vfskip` is the critic-divergence signature; the trainer aborts on its
own thresholds and the supervisor stops relaunching after three aborts in an
hour.

**Evaluating a checkpoint**: convert with `native_weights.py import`, then
either `python -m ptcg.rl.arena` for one matchup or `ptcg/rl/eval_watch.py`
for the battery. Build opponents from checkpoints with `tools/make_opponents.sh`.
Count worker processes before believing a battery is running in parallel.

**Without a GPU**: you can build and run every C env test, export blobs, run
the observation validator, run the deck lab against an existing deck matrix,
and play games with any checkpoint on CPU. You cannot build `_C`, train, or
run the CUDA parity tools.

## The deck lab: agent workflow

The mechanical layer is `tools/decklab_*.py` (documented in `docs/decklab.md`).
An orchestrating agent runs the cycle below, spawning proposer and reviewer
agents. The methodology rules are binding: pre-registration before any game,
a two-stage test, replication or post-append scoring, and honest reporting of
nulls.

### State you read first, every cycle

1. `decklab/LEDGER.jsonl`: what has been tried. Never re-propose a swap that
   already failed on the same base deck unless explicitly asked.
2. `decklab/config.json`: lanes, checkpoints, game counts, accept bars,
   matrices.
3. `decklab/notes.jsonl` (via `tools/dlab.py notes`): durable facts from
   earlier cycles.

### Phase 1: evidence

```bash
python3 tools/decklab_pack.py field
python3 tools/decklab_pairs.py                 # if decklab/packs/pairs_*.csv is missing or stale
python3 tools/decklab_pack.py swapstats
python3 tools/decklab_pack.py deck --name <target>
python3 tools/decklab_pack.py cards --deck decks/pool/<target>.csv
```

Quick lookups go through `tools/dlab.py` (`deck`, `card`, `diff`, `near`,
`find`, `history`, `mat cell|row|rank|check`), never ad-hoc scripts.
`dlab.py history <card or deck>` before proposing anything.

Target selection when nothing is queued: prefer decks with high field share
and a low win rate for us (the dossier's targeted gauntlet says what to fix),
and strong decks whose worst field-relevant matchup looks addressable. Skip
decks the evaluator checkpoint pilots badly: the instrument cannot measure
improvements to them, so say so instead of testing.

### Phase 2: proposers (one agent per target, in parallel)

Each proposer gets the deck dossier, the cards pack, `packs/swapstats.md`,
and this contract:

- Read the matchup profile and the single-swap natural experiments first; the
  cheapest good proposal transplants a measured-positive swap.
- `data/carddb.json` (through `dlab.py card`) is the card-text ground truth.
- Produce one or two proposals: `decklab/proposals/<id>/deck.csv` (60 ids) and
  `meta.json` with `id`, `base_deck`, `rationale` (citing specific pack lines
  and numbers), `target_opponents` (3-6 pool decks, normally the dossier's
  suggested gauntlet, committed before any game), `predicted` directions,
  `status: "proposed"`. Use the fewest swaps the mechanism needs (a 1-2 swap
  stays card-attributable; a structural rebuild up to `max_swaps` is accepted
  or rejected only as a whole deck). Only card ids present in the card
  database. Get the id from
  `python3 -c "import sys;sys.path.insert(0,'tools');import decklab_common as C;print(C.next_proposal_id())"`
  and suffix it with a short slug.
- Probing is allowed while ideating:
  `python3 tools/decklab_ab.py probe --deck <proposal-dir|deck.csv> [--opps a.csv,b.csv] [--games 60] [--vs-base]`.
  Budget at most 600 probe games per proposal. A probe is hypothesis
  generation: it may be cited in the rationale, it is never acceptance
  evidence, and a proposal whose probe looked bad is usually dropped rather
  than re-rolled. Every probe is ledgered; re-probing until a number looks
  good is visible and forbidden.

**Evaluator-OOV cards.** Cards outside the evaluator checkpoint's training
vocabulary (`decklab_common.evaluator_vocab()`) were never piloted by it, so
its games carry no information about them. `decklab_validate.py` stamps such
proposals `evidence_class: reasoning`. They are not game-gated at all:
`decklab_ab.py run` refuses them, probes are informational only and never a
drop reason, and the adversarial review of the first-principles case (exact
card text, the concrete line it enables turn by turn, why each in-vocabulary
alternative is worse, expected cost, ideally engine-verified mechanics) is the
gate. On a PROCEED verdict the orchestrator sets `status: accepted_reasoning`
plus a ledger event, and the deck stages like any accepted deck; the live run
then learns the cards after the append and `decklab_score.py` is the only
measurement that counts.

### Phase 3: adversarial review (one agent per proposal)

The reviewer tries to refute the proposal against the cards pack: rules
misread, energy curve broken, combo cannot execute, claim not supported by the
cited evidence. Verdict: proceed / revise (one round at most) / drop. A drop
needs a broken premise (card-text misread, engine-refuted mechanism, illegal or
self-defeating list). Pessimism about effect size or piloting is a residual
risk to record, not a drop reason: measured-class proposals have the A/B to
decide and reasoning-class proposals have post-append scoring. Do not reject a
Stage 2 line without its basic as "dead" (Rare Candy lines and attack
selectors are real). Record drops with
`decklab_common.ledger_append({"event": "dropped_review", "id": ..., "reason": ...})`.

### Phase 4: validate and test (sequential, one arena)

```bash
python3 tools/decklab_validate.py decklab/proposals/<id>
python3 tools/decklab_ab.py run decklab/proposals/<id>            # full profile
python3 tools/decklab_ab.py run decklab/proposals/<id> --profile screen
```

Run A/Bs one at a time and let them finish; no peeking at partial samples.
The harness records results, computes the verdict and appends the ledger.
Predictions and target opponents are frozen at validation; testing after
editing them is a new proposal id.

### Phase 5: stage

Acceptance is both stages passing on the primary checkpoint (and on the
replicate checkpoint when `replication_required` is set). Then
`python3 tools/decklab_batch.py --dry-run` and report what is ready. The
append to a live run is a human action (`docs/deck-pool.md`). After an append
lands, the next cycle runs `python3 tools/decklab_score.py --since <epoch>` to
score inserted decks against their parents at training volume; with
single-checkpoint acceptance this post-insertion score is the replication, so
report it faithfully.

### Report and notes

End every cycle with one table: proposal | swaps | targeted delta ± 2SE |
field delta ± 2SE | status. State failures plainly; "no measurable effect at
n" is not "neutral" and never gets re-rolled for a better seed. Then write the
cycle's durable facts as lab notes (`tools/dlab.py note deck|card <key> "<fact>" --by <tag>`,
at most 240 characters each, one to three per cycle): dose maps, engine-proven
mechanics, watch-items, "X already failed on base Y".

### Hard rules

- Pre-committed targets and predictions; never chosen after seeing games.
- Never mark a proposal accepted without the harness verdict.
- Arena failures abort the cycle with the error surfaced; no silent retries.
- At most six proposals tested per cycle unless the operator raises it.
- Both arms of one experiment run on the same lane; the lane is part of the
  instrument.
