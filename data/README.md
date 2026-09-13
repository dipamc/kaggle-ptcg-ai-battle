# data/

Files that ship with the repo:

| file | what |
|---|---|
| `lingering_effects.json` | card-text-derived effect records (durations, locks, damage modifiers) the tracker and the C env use |
| `embed_pca.npz` | PCA-reduced card / attack / ability text embeddings the model's static tables read |

Files you must add from the Kaggle competition
([pokemon-tcg-ai-battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle),
Data tab) before anything runs:

| path | what | from |
|---|---|---|
| `data/cg/` | the simulator SDK: `api.py`, `game.py`, `sim.py`, `utils.py`, `__init__.py` and the engine shared library (`libcg.so` on Linux x86_64, `libcg-arm64.so`, `libcg.dylib` on macOS) | the competition's starter package (the `cg` directory) |
| `data/EN_Card_Data.csv` | card sheet (names, expansions, printed text) | the competition Data tab |

Where they are in the competition archive (`pokemon-tcg-ai-battle.zip`,
about 300 MB, Kaggle login and accepted rules required):

| archive entry | copy to |
|---|---|
| `sample_submission/sample_submission/cg/` | `data/cg/` |
| `EN Card Data.csv` (the name has spaces) | `data/EN_Card_Data.csv` |

The rest of the archive (`JP Card Data.csv`, the card-ID PDFs, the engine
C++ sources under `ptcg_engine/`) is not used. With the Kaggle CLI:

    kaggle competitions download -c pokemon-tcg-ai-battle -p /tmp/kdata
    unzip -q /tmp/kdata/pokemon-tcg-ai-battle.zip \
        'sample_submission/sample_submission/cg/*' 'EN Card Data.csv' -d /tmp/kdata
    cp -r /tmp/kdata/sample_submission/sample_submission/cg data/cg
    cp '/tmp/kdata/EN Card Data.csv' data/EN_Card_Data.csv

The PyPI package `kaggle-environments` (1.32.7 and later) ships the same
engine library under `kaggle_environments/envs/cabt/cg/` (`libcg.so`,
`sim.py`, `game.py`) but not `api.py`, `utils.py` or the card sheet, so it is
not a substitute: the SDK has to come from the competition archive.

Then build the derived card database:

    PYTHONPATH=data python3 tools/build_carddb.py      # -> data/carddb.json

Everything else under `data/` is generated locally and gitignored.
