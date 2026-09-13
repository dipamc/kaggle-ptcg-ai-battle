# Final submission: Hydrapple ex / Meganium

This directory is the agent submitted to the Kaggle competition as submission
**55559155** (Team Unown Gradiant):
<https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/submissions?submissionId=55559155>

| file | what |
|---|---|
| `main.py` | the Kaggle agent: greedy policy-only play (`SearchAgent(cfg=None, temp=0.0)`), rejection guard, time-budget valve |
| `model.pt` | the trained policy: token transformer d=256, 4 layers, 8 heads, FFN 512 (`HEADS = 8` in `main.py`) |
| `deck.csv` | the deck list, identical to `decks/pool/hydrapple_ex_meganium_forest_of_vitality__03.csv` |
| `ptcg/` | the runtime package as shipped (encoder, tracker, effects, model, agent) |
| `data/` | `embed_pca.npz` and `lingering_effects.json`, the static tables the encoder and model read |

`ptcg/` here is the runtime exactly as it was submitted and is deliberately
not synced with the top-level `ptcg/`, which has since gained a
frozen-opponent slot in the observation layout (`rl/buffers.py`) and the
optional ensemble, win-line, bench-guard, lethal and value-rank overrides in
`rl/search_agent.py`. Diffs between the two copies are expected; play this
bundle with this copy.

Two files that were part of the uploaded tarball are not redistributed here
because they are competition data: the simulator SDK directory `cg/` and the
card sheet `EN_Card_Data.csv` (at the bundle root and under `data/`). Get both
from the competition Data tab (see `../data/README.md`) and put them in place:

```bash
cp -r ../data/cg cg
cp ../data/EN_Card_Data.csv EN_Card_Data.csv
cp ../data/EN_Card_Data.csv data/EN_Card_Data.csv
```

Pack it the way it was submitted (entries at the archive root, no
`__pycache__`):

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} +
COPYFILE_DISABLE=1 tar czf ../hydrapple_submission.tar.gz --exclude=README.md .
```

Play it locally against any Kaggle-format agent directory with the arena:

```bash
cd ..
PYTHONPATH=data:. python3 -m ptcg.rl.arena --ckpt submission/model.pt \
    --deck submission/deck.csv --opponent opponents/<some-agent> \
    --games 20 --heads 8 --temp 0 --device cpu --out results.jsonl
```
