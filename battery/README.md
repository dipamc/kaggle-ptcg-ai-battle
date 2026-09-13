# battery/ — the evaluation battery

The battery is the fixed matrix every checkpoint was scored on: the candidate
checkpoint pilots each **candidate deck** against each **anchor opponent**,
policy-only, seats alternating. It has two layers:

1. **Continuous** (`battery_eval` in the write-up): the bench matrix inside
   `ptcg/rl/eval_watch.py`, run on every new checkpoint of a live run and
   logged under `--bench-prefix eval` (`eval/win_rate_<deck>_vs_<opp>`,
   `eval/mean5_vs_<opp>`, `eval/macro_mean`).
2. **Arena-backed** (the numbers that picked the final submission):
   `battery/run_battery.sh`, many `ptcg.rl.arena` processes with greedy play on
   both sides, pooled by `battery/aggregate.py`.

Only code and deck lists ship here. The anchor policy is a training
checkpoint that is not redistributed; rebuild opponents from any checkpoint
as shown below.

## Configuration used for the final (d256, 6B-step target) run

**Anchor opponents** — six Kaggle-format bundles `opponents/e4235_<short>`,
all piloted by the same policy: epoch 4235 of run `native-sp3bfix4-8x`
(2.22B aggregate steps, d128 / 4 layers / 4 heads / FFN 256, so `HEADS = 4`).
Each bundle plays one anchor deck from `battery/opponent_decks/`:

| short | deck file | byte-identical list in `decks/pool` |
|---|---|---|
| marnie | `marnie_s_grimmsnarl_ex__lf_404125.csv` | `marnie_s_grimmsnarl_ex_spikemuth_gym__04` |
| garchomp | `cynthia_s_garchomp_ex__ladder.csv` | `cynthia_s_garchomp_ex_cynthia_s_roserade_cynth__01` |
| fez | `fezandipiti_ex__lf_a4066a.csv` | `alakazam_enhanced_hammer__03` |
| dragapult | `dragapult_ex__ladder.csv` | not in the pool |
| festival_lead | `festival_lead__v12.csv` | `rillaboom_dipplin_festival_grounds__03` |
| crustle | `crustle__v1.csv` | `crustle_jumbo_ice_cream__01` |

These are also the trainer's default `--anchor-decks`, so the run oversampled
the decks the battery measured it against.

**Candidate decks** — the five in `battery/candidate_decks/`, the candidate
checkpoint pilots each of them:

| deck file | byte-identical list in `decks/pool` |
|---|---|
| `alakazam_enhanced_hammer_ld03.csv` | `alakazam_enhanced_hammer__03` |
| `dragapult_crushing_ld08.csv` | `dragapult_ex_crushing_hammer__08` |
| `hydrapple_meganium_ld03.csv` | `hydrapple_ex_meganium_forest_of_vitality__03` (the submitted deck, `submission/deck.csv`) |
| `lopunny_mist_energy_lm08.csv` | `dudunsparce_mega_lopunny_ex_mist_energy__08` |
| `mega_lucario_hariyama_ppp.csv` | `mega_lucario_ex_hariyama_premium_power_pro__01` |

**Continuous battery**: 5 decks × 6 opponents = 30 cells, `--bench-games 50`
(1500 games per checkpoint), sampled play (`--temp 1`) on the candidate,
`--bench-prefix eval`. The launch, with the eval watcher of `docs/eval.md`:

```bash
PYTHONPATH=data:. python3 -m ptcg.rl.eval_watch --run-name <run> --heads 8 \
    --bench-opponents $(ls -d opponents/e4235_* | paste -sd, -) \
    --bench-decks $(ls battery/candidate_decks/*.csv | paste -sd, -) \
    --bench-games 50 --bench-prefix eval --bench-workers <cores-1> --bench-device cpu
```

**Arena-backed measurement** (final checkpoint and deck choice): the same
30 cells at `--temp 0` on the candidate, greedy opponents (the bundles are
greedy by construction), 100 games per cell and seed, seats alternating, on
CPU workers. Recorded for the submitted deck at three late checkpoints of the
run, mean over the six anchors: epoch 9350 0.675, epoch 10065 0.708,
epoch 10450 (5.48B steps) 0.727 with cells garchomp 0.98, crustle 0.81,
marnie 0.73, dragapult 0.71, festival_lead 0.65, fez 0.48.

## Running it

Build the six anchor bundles from a checkpoint (`.pt`, converted with
`native/tools/native_weights.py import` if needed; pass that checkpoint's
head count):

```bash
POOL=battery/opponent_decks tools/make_opponents.sh <anchor.pt> e4235 4 \
    marnie=marnie_s_grimmsnarl_ex__lf_404125 garchomp=cynthia_s_garchomp_ex__ladder \
    fez=fezandipiti_ex__lf_a4066a dragapult=dragapult_ex__ladder \
    festival_lead=festival_lead__v12 crustle=crustle__v1
```

Then score a candidate checkpoint on the full matrix (defaults: the five
candidate decks, `opponents/e4235_*`, 100 games × 2 seed chunks per cell,
greedy, CPU, one process per core):

```bash
battery/run_battery.sh <candidate.pt> 8 runs/battery_<name>
python3 battery/aggregate.py runs/battery_<name> --json runs/battery_<name>/summary.json
```

`--decks`, `--opponents`, `--games`, `--chunks`, `--jobs`, `--device` and
`--temp` narrow or scale the run; `run_battery.sh --games 4 --chunks 1` is a
minute-long smoke test. Any Kaggle-format agent directory works as an
opponent, so the same driver scores a candidate against public agents.
