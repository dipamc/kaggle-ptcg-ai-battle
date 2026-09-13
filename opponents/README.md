# opponents/

Kaggle-format agent directories used as fixed evaluation opponents:

    opponents/<name>/main.py     defines agent(obs_dict) -> list[int]
    opponents/<name>/deck.csv    60 card ids, one per line
    ...                          anything else the agent needs (model, data, cg/)

`ptcg.rl.arena.load_kaggle_agent` exec-loads `main.py` the way the Kaggle
runner does (no `__file__`, cwd switched to the directory).

Build opponents from your own checkpoints with `tools/make_opponents.sh`, or
drop in any public Kaggle agent. Point the eval battery at them with
`--bench-opponents opponents/a,opponents/b`.

This directory is gitignored except for this file.
