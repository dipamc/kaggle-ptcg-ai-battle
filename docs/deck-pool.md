# Deck pools

Decklists are inputs to the environment, not code. The trainer draws a deck per seat at every
episode start, and everything downstream — anchors, the deck matrix, sampling weights, the
analysis tools — addresses a deck by a numeric **deck id**. This document covers the file format,
the id rule, the binary table blob, how the trainer samples decks, and how to grow or reweight the
pool while a run is training.

## 1. Deck files and the two pools

A decklist is a plain `.csv` of **exactly 60 lines, one engine card id per line**, no header. Ids
are the card ids of the competition card sheet (`data/EN_Card_Data.csv`, column `Card ID`) as the
engine exposes them through `all_card_data()`. Line order carries no meaning; a deck is a multiset.

| directory | files | naming | who draws it |
|---|---|---|---|
| `decks/pool` | 616 | `<archetype>__<nn>` | every seat |
| `decks/pool_alt` | 87 | `coverage_<energy type>__<nn>` | the random opponent's seat only |

`decks/pool` is the **training pool**: 616 competitive lists over 105 archetype prefixes with a
two-digit variant suffix, e.g. `crustle_jumbo_ice_cream__01`. These are the decks the learner
pilots, and the only decks in the deck matrix.

`decks/pool_alt` is the **coverage pool**, named by basic energy type (`coverage_colorless`,
`coverage_darkness`, `coverage_dragon`, `coverage_fighting`, `coverage_fire`, `coverage_grass`,
`coverage_lightning`, `coverage_metal`, `coverage_psychic`, `coverage_water`). Its job is gradient
coverage, not strength: between them these lists reach card ids the training pool never plays, so
every card the model has a feature row for is seen at least sometimes. Only the random opponent's
seat holds one, so the learner meets those cards without the training pool carrying unplayable
lists. Confirm the counts with `ls decks/pool/*.csv | wc -l` (616) and `ls decks/pool_alt/*.csv |
wc -l` (87).

## 2. The deck-id rule

> **Deck id = the deck's position in the sorted glob of 60-line `.csv` files in its pool
> directory.**

Training ids are `[0, n)`. Coverage ids start at a **fixed base**, `PT_DECK_ALT_BASE = 1024`
(`native/ptcg/ptcg_env.h`), mirrored by hand as `DECK_ALT_BASE` in `tools/deck_matrix.py` and
`tools/pool_names.py` — change them together. The gap is deliberate: appending a training deck can
never renumber a coverage deck. The same constant caps the training pool at 1024, and the loader
rejects a blob that reaches it rather than letting the id spaces collide.

Every tool derives ids through the same construction: `_load_decks` in
`native/tools/export_tables.py` (writes the table), `pool_deck_names` in `native/train_native.py`
(anchors, weights), `deck_names` in `tools/deck_matrix.py` (matrix labels), `pool_names` in
`tools/pool_names.py` (the pinned listing). Files that are not 60 lines, or that do not parse as
integers, are skipped and the skip is printed; metadata files use a leading underscore by
convention, so they sort first and the 60-line filter drops them before they can take an id.

Two consequences, both with teeth. **A training deck whose filename sorts mid-list renumbers
everything after it** — archetype-style names usually do not sort last, so a deck intended as an
append routinely lands as an insert, and decks added to a running pool must be named so they sort
after every existing name. And **a blob re-exported from a changed pool must never be paired with
checkpoints, matrices or reports from the old pool**: nothing in the id space is self-describing,
so a drifted pool yields a blob that is not incomplete but silently relabelled.

```bash
python3 tools/pool_names.py --pool decks/pool --alt-pool decks/pool_alt
python3 tools/pool_names.py --pool decks/pool --write data/pool_names.txt
```

The written file is the **pinned listing**: what `export_tables.py --expect-names` / `--append-to`
and `validate_deck_batch.py --pinned` check a pool against, and the `--names` argument the matrix
tools label ids with. Keep one with every run.

## 3. Building the tables blob

The C/CUDA backend reads one binary file holding every static table it needs:

```bash
PYTHONPATH=data:. python3 native/tools/export_tables.py native/ptcg_tables.bin \
    --pool decks/pool --alt-pool decks/pool_alt
```

Magic `PTCGTAB1`, little-endian, 15 int32 layout constants the C side static-asserts against, then
named sections (the `export_tables.py` docstring lists them all): **environment tables** (attack
damage and costs, per-card energy type, weakness, resistance, retreat, top attacks, membership
masks for supporters / shields / Pokemon); **effect records**, the flattened lingering-effect rows
for attacks and card plays plus their magnitudes, each describing kind, target, condition, duration
and lock; **decks**, as `decks (n_decks, 60)` int32 in sorted-glob order plus `decks_alt
(n_decks_alt, 60)` when `--alt-pool` is given (omitting it writes no section, which the C side
reads as zero coverage decks); and **model static tables**, the per-card and per-attack feature
rows the CUDA model gathers from plus PCA-reduced card / attack / ability text embeddings loaded
from `data/embed_pca.npz`.

Card tables are sized by the whole card universe, not by the pool, so a deck built from legal cards
needs no table regeneration — only `decks` changes. Two guard flags: `--expect-names <listing>`
fails unless the pool's sorted glob is *exactly* the pinned listing in order, and `--append-to
<listing>` fails unless the pool is that listing plus new names **at the end**, naming the
offending deck before any GPU time is spent. **Keep the blob with the run, as an artifact, not a
build product:** it pins the id space, and rebuilt later from a pool that has moved on it loads
fine and means something different.

## 4. How the trainer samples decks

`native/train_native.py` reads the pool directory only to resolve *names*; the decks come from the
blob that `PTCG_TABLES` points at. At each episode start (`game_start`, `native/ptcg/env.c`) the
environment picks an opponent kind — mirror self-play with probability `--mix-self` (default
`0.98`), a random opponent otherwise, or a frozen league bank when the row is tagged
(`--league-ckpts`, `--league-pct`) — then draws one deck per seat:

| seat | draws from |
|---|---|
| learner | training pool |
| self-play or league opponent | training pool |
| random opponent | training pool **plus** the coverage pool |

Anchors are an optional oversampling hook. `--anchor-decks` takes a comma list of deck names (a
six-deck default lives in `ANCHOR_DECKS`); `resolve_anchor_indices` maps names to ids and asserts
the pool glob matches the blob row for row before trusting any index. `--p-anchor-learner`,
`--p-anchor-self-opp` and `--p-anchor-league-opp` all default to `0.0`, reproducing the plain
uniform draw exactly; above 0 that seat picks uniformly among the anchors with that probability.
Anchors are training ids, reachable under either draw bound, and their per-deck win rates are
logged in `--anchor-decks` order.

Two properties are invisible from the command line, so `log_pool_composition` prints them on rank 0
at startup. The coverage pool widens *who the random opponent can be*, and that split lives in the
blob. And **games against the random opponent are excluded from the deck matrix**: that seat is not
a policy, so counting those games would measure the random agent rather than the matchup. It is
also the only seat that can hold a coverage deck, which is why no coverage id reaches the matrix.

## 5. Growing the pool while a run is training

The pool grows mid-run without stopping the trainer, under an **append-only** contract: new decks
take ids `[n, n + k)`, existing rows never move, coverage is unchanged. Removal is a stop/resume
operation — use a zero weight (§6) instead.

Launch with `--deck-pool-file <path>`. The trainer polls that file every `--pool-check-every`
epochs (default 20) between `_C.train()` returning and the next `_C.rollouts()`, the one window
where the environment threads are parked. The file holds one line, `<version> <blob path>`,
published with `os.replace` so no rank reads a partial file. `_maybe_swap_pool` allreduces the
requested version and **skips the round unless every rank sees the same number** (a split read just
means the writer landed mid-check), calls `pt_reload_decks`, allreduces the result and treats a
split apply as **fatal** (ranks disagreeing about what a deck id means corrupts the training data
and the matrix with nothing downstream to notice), then prints `deck pool -> vN, n_decks M, at
epoch E` and records `<run_dir>/pool_version`, the trainer's memory across resumes. Versions
**start at 2**: the initial pool arrives via `PTCG_TABLES`, not a request, and version 0 means "no
request". A rejected version is remembered and never retried — fix the blob and publish a new
number.

`pt_reload_decks` (`native/ptcg/tables.c`) enforces the contract in C. It rejects, and says why, a
blob with fewer training decks than the live pool (a removal), one reaching `PT_DECK_ALT_BASE`, one
changing any byte of an existing deck row (a reorder wearing an append's clothes), or one changing
the coverage pool's size or contents. Rejection leaves the live table untouched and training
continues on the old pool, so **absence of an error is not proof** — check the count.

### Procedure

```bash
REQ=<the --deck-pool-file path>
# validate: 60 lines, ids exist, sorts last, no duplicates, under the cap
python3 tools/validate_deck_batch.py <batch_dir> \
        --pool decks/pool --pinned data/pool_names.txt --cap 1024
cp <batch_dir>/*.csv decks/pool/          # names must sort AFTER every existing deck
PYTHONPATH=data:. python3 native/tools/export_tables.py native/ptcg_tables_v2.bin \
    --pool decks/pool --alt-pool decks/pool_alt --append-to data/pool_names.txt
printf '2 %s\n' "$PWD/native/ptcg_tables_v2.bin" > "$REQ.tmp" && mv -f "$REQ.tmp" "$REQ"
grep -E "deck pool -> v|REJECTED|REMOVAL|FATAL" <run log>      # verify it landed
python3 tools/pool_names.py --pool decks/pool --write data/pool_names.txt
```

Silence for up to `--pool-check-every` epochs is normal. Afterwards repoint `PTCG_TABLES` at the
new blob: the request grows a pool that is *already running*, while `PTCG_TABLES` is what a
**restart** loads, so a stale value means the next restart silently trains on the smaller pool. The
pool directory must move with the blob, since ids are positions in its sorted glob.
`native/tools/pool_append_watch.py` is the writer half on a schedule: it watches a run log and
publishes successive blobs at chosen **global** step counts (the log prints rank 0's own step, so
global progress is `printed_step * world`), each blob built with `--append-to` against the previous
listing.

```bash
PYTHONPATH=. python3 native/tools/pool_append_watch.py --log <run log> \
    --request "$REQ" --world <n ranks> --at 15000000=native/ptcg_tables_v2.bin
```

### C-side contract tests

Three CPU-only tests exercise the contract; no CUDA needed. The `make` run targets export
`DYLD_LIBRARY_PATH` for the engine library themselves on macOS, where the linker's `-rpath` is
inert. Build a baseline blob, the same pool plus exactly three decks named so they sort last, and
the same pool plus one deck named so it sorts first:

```bash
rm -rf /tmp/pool_grown && cp -R decks/pool /tmp/pool_grown
cp <three new>.csv /tmp/pool_grown/                # names sort LAST
rm -rf /tmp/pool_ins && cp -R decks/pool /tmp/pool_ins
cp <one new>.csv /tmp/pool_ins/aaa_insert.csv      # name sorts FIRST
for p in decks/pool:/tmp/v1.bin /tmp/pool_grown:/tmp/grown.bin /tmp/pool_ins:/tmp/ins.bin; do
  PYTHONPATH=data:. python3 native/tools/export_tables.py "${p#*:}" \
      --pool "${p%%:*}" --alt-pool decks/pool_alt
done
make -C native reload      V1=/tmp/v1.bin GROWN=/tmp/grown.bin INSERTED=/tmp/ins.bin
make -C native decklock    V1=/tmp/v1.bin GROWN=/tmp/grown.bin
make -C native deckweights V1=/tmp/v1.bin GROWN=/tmp/grown.bin
```

`reload` (`native/ptcg/test_reload.c`) is the full append contract; order matters, since the
reorder blob is tried while the pool is still at the baseline size so the prefix check fires rather
than the removal check, and `GROWN` must be baseline-plus-three. `decklock`
(`native/ptcg/test_deck_lock.c`) proves the deck-table rwlock has teeth: a swapper holding the
write lock stops game starts dead, and hammering reload against live stepper threads corrupts
nothing — draw and copy are one critical section, because `deck_row` returns a pointer *into* the
table and the copy dereferences it. `deckweights` (`native/ptcg/test_deck_weights.c`) measures the
*distribution*, not a return code: uniform stays uniform, a requested ratio is reproduced,
zero-weight decks are drawn exactly zero times, the coverage share is invariant under scaling,
every reject path leaves the previous distribution intact, and an append under weights lands new
decks at the mean.

## 6. Deck sampling weights

`--deck-weights-file <path>` reshapes the training-pool sampling distribution mid-run through the
same request protocol. Format (`read_weight_request` in `native/train_native.py`):

```
version 3
default 1.0
<deck name> <weight>          # '#' comments and blank lines are ignored
```

* **Names, never indices.** One unknown name rejects the whole file: it usually means the file was
  written against a different pool, and applying the rest would weight the wrong decks.
* **Weight 0 disables a deck on both seats** without removing it. Its id and its recorded history
  survive, so old matrices stay summable — that is the reason to weight rather than delete.
* Weights cover the **training pool only** and are normalised to mean 1 within it, so they are
  relative: scaling the file changes nothing, and cannot change how often the random opponent
  reaches into the coverage pool, which stays uniform. Decks appended later enter at the mean, so
  an append cannot reshape the existing distribution.
* **Applied at startup and every `--pool-check-every` epochs**, so a supervised restart re-applies
  the live distribution unattended. The weights check runs *after* the pool check in the same epoch
  boundary, so a batch that appends and then reweights resolves names against the grown pool.
* An anchor cannot be zeroed while any `--p-anchor-*` is above 0: the anchor branch runs before the
  weighted draw, so the deck would keep being played while every report said it was off.
* In-flight games are untouched — weights are read at game start, and a running episode holds its
  own copy of both decklists.

On apply, rank 0 prints and logs the **effective sample size** `ESS = (sum w)^2 / sum w^2`, as in
`deck weights -> v3, 611/616 active, ESS 611.0, min 0 max 1, at epoch 2420`. ESS is what catches a
fat-fingered file: a typo putting most of the mass on one deck collapses it toward 1 while `min`
and `max` still look plausible. A malformed line anywhere makes the whole request unparseable and
nothing is applied; the trainer warns once per edit rather than once per check. Exercise the
shipped parser first with `python3 native/tools/test_weight_request.py`.

## 7. Reading the deck matrix

With `--deck-matrix-every <episodes>` the environment writes `<run_dir>/deckmat/r<rank>_e<epoch>_<seq>.bin`:
a games/wins matrix over the deck pool from the learner seat's perspective, reset after each flush,
so summing files gives any window. Matrices are pre-sized to a constant width
(`PTCG_DECK_MATRIX_MAX`, default `PT_DECK_ALT_BASE`) to stay summable across a mid-run append; rows
past the real pool are all-zero. All four readers below take `--pool` and `--alt-pool`; the three
cluster-based ones also take `--names <listing>`, so a matrix is read against the pinned listing of
the pool it was written from rather than a live glob.

`tools/deck_matrix.py <dir-or-files...>` is the raw reader: per-deck win rate, game counts, 95%
intervals, and `--pair <substring>` for one deck's best and worst matchups. Its `res` column is the
**resolved fraction** — only decided games reach the matrix, so a deck below 1.0 stalls out that
often and its win rate is conditional on the game ending. `--epoch-min` / `--epoch-max` slice by
epoch; `--by-epoch` tracks one deck over training.

`tools/deck_generalists.py <deckmat-dir>` ranks by **floor** (worst matchup) rather than mean, on
the argument that the best mean in a counter-cycling field usually belongs to a deck that folds to
one archetype. Three corrections make the floor mean something: shrinkage toward each deck's own
mean, with a prior strength fitted from the data (observed spread minus the binomial part);
grouping opponents into **matchup classes** by profile correlation, so an archetype shipping seven
variants is not counted seven times; and excluding decks below `--res-warn`. It also reports
`spread`, the noise-corrected variation across classes.

`tools/deck_nash.py <deckmat-dir>` corrects for the opponent *distribution* instead. Decks are
drawn per file, so a raw mean is weighted by how many variants of each archetype the pool happens
to ship. `macro` averages the shrunk per-class rates with equal weight per class; `nash` solves the
symmetric zero-sum game for an equilibrium mixture by multiplicative-weights self-play and reports
each deck's win rate against it, so opponents no rational field would play get weight near zero
regardless of file count. The printed exploitability is the convergence check.

`tools/deck_evalset.py <deckmat-dir>` picks an eval *opponent* set, the opposite problem: between
them the opponents must be able to expose any hole. It selects generalists (strong everywhere, from
distinct classes, ranked by `macro`, gated on floor and equilibrium support) plus specialists
chosen by greedy set cover over the classes the generalists do not already punish. The printed
coverage matrix is the deliverable: it names any class no opponent in the set beats.

## 8. Deck construction rules

| rule | note |
|---|---|
| exactly 60 cards | anything else is dropped silently by the exporter |
| max 4 copies per card **name** | counted across ids: one name printed in several sets still shares the limit |
| basic energy exempt from the 4-copy rule | special energy is **not** exempt |
| at least one Basic Pokemon | a deck with none cannot start |
| max 1 ACE SPEC card in total | not per name |

There is no other ban list: any card id in the engine's card database is legal. Because many card
names exist under several ids, deduplicate by **name**, not by id, when counting copies.
`tools/decklab_validate.py` enforces the copy, ACE SPEC and card-existence rules on a single
proposal; `tools/validate_deck_batch.py` adds the checks that protect the id space rather than the
rules — the filename must sort after every pinned name, the list must not duplicate (by md5) an
existing pool deck or another deck in the same batch, the batch must be internally sorted, and pool
plus batch must stay under the 1024 training-id cap. It exits non-zero on any failure, so it gates
a build.
