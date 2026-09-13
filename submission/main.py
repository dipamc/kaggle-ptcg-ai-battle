"""Kaggle-format agent: greedy policy-only play with the trained
token-transformer (ptcg.rl.search_agent.SearchAgent with cfg=None, which
degrades to the pure policy: sequential pick + STOP decomposition, no
search; temp=0.0 so every move is the argmax). Per-decision cost is one
to a few CPU forwards.

Bundle layout (built by tools/make_opponents.sh, also the submission
layout): main.py, model.pt, deck.csv, ptcg/ (this repo's package), cg/
(the competition SDK) and data/ (embed_pca.npz, lingering_effects.json,
EN_Card_Data.csv). All module data paths resolve relative to the bundle,
so it works from any cwd.

Rejection guard: obs["logs"] is the delta since the last ACCEPTED
selection; the runner re-delivers a byte-identical obs on a true
rejection. Same-signature select with changed logs -> legitimate
re-present -> normal policy answer; unchanged logs -> policy answer unless
it repeats one already tried for this select, else _recover enumeration.
A true rejection never sees a repeated answer (no reject loops).

Time budget: remainingOverageTime below 30 s switches to zero-model-cost
legal answers. Any internal failure degrades to a legal random answer (an
invalid answer or crash forfeits the episode; a bad-but-legal answer does
not).
"""
import json
import os
import random
import sys
import time

_LOAD_T0 = time.time()


def _agent_dir():
    """kaggle-environments exec()s this file WITHOUT defining __file__
    and with an arbitrary cwd (its loader only appends the agent dir to
    sys.path) — so resolve our data dir the way the official sample
    does: try __file__ if it exists, then cwd, then the documented
    absolute runtime location."""
    try:
        d = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(d, "deck.csv")):
            return d
    except NameError:
        pass
    if os.path.exists("deck.csv"):
        return os.getcwd()
    return "/kaggle_simulations/agent"


AGENT_DIR = _agent_dir()
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import torch

torch.set_num_threads(2)   # container: ~1.6 vCPU

from ptcg.rl.search_agent import SearchAgent

# Attention heads the SHIPPED checkpoint was trained with. Width, depth and
# ffn are recovered from the weights, but heads cannot be (the projections are
# d x d whatever the head count) and a wrong value loads SILENTLY and computes
# different attention. tools/make_opponents.sh stamps this line via HEADS=<n>.
HEADS = 8

SA = SearchAgent(os.path.join(AGENT_DIR, "model.pt"),
                 os.path.join(AGENT_DIR, "deck.csv"),
                 cfg=None, seed=0, device="cpu", data_dir=AGENT_DIR,
                 heads=HEADS, temp=0.0)
DECK = list(SA.deck)

# internal overage estimate for hosts that omit remainingOverageTime
# (local validation); kaggle's own number is preferred when present
ROT = {"est": 600.0, "boot": False}


def _legal_random(sel):
    n = len(sel["option"])
    k = random.randint(sel["minCount"], min(sel["maxCount"], n))
    return sorted(random.sample(range(n), k))


def _sel_sig(obs):
    """Cheap identity for a select. A re-presented signature is EITHER
    a rejected answer (runner has no retry guard — a repeated identical
    answer would loop until clock death) OR a legitimate identical
    re-present after an accepted answer; obs["logs"] advancement
    separates the two (see module docstring)."""
    sel = obs["select"]
    cur = obs["current"]
    return (cur.get("turn"), cur.get("yourIndex"), sel.get("type"),
            sel.get("context"), len(sel.get("option") or []),
            sel.get("minCount"), sel.get("maxCount"),
            json.dumps(sel.get("option"), sort_keys=True)[:600])


REJ = {"sig": None, "tried": []}
LAST = {"sig": None, "logs": None}   # previous presentation (sig, logs-json)
GUARD = {"fresh": 0, "repeat_policy": 0, "repeat_recover": 0, "panic": 0}


def _recover(sel):
    """Previous answer(s) to this exact select were rejected: produce a
    legal answer we have NOT tried yet, escalating from policy-shaped
    picks to plain option enumeration to random."""
    n = len(sel["option"])
    lo = max(sel["minCount"], 1)   # empty submits are the known reject
    for k in range(lo, min(sel["maxCount"], n) + 1):
        for start in range(n):
            cand = sorted((start + j) % n for j in range(k))
            if len(set(cand)) == k and cand not in REJ["tried"]:
                return cand
    return _legal_random(sel)      # exhausted: last resort


def agent(obs_dict: dict) -> list[int]:
    t0 = time.time()
    if obs_dict.get("select") is None:
        # episode start: hand over the deck, reset per-episode state
        REJ["sig"], REJ["tried"] = None, []
        LAST["sig"], LAST["logs"] = None, None
        ROT["est"] = 600.0
        if not ROT["boot"]:      # module import + model load bill
            ROT["est"] -= time.time() - _LOAD_T0
            ROT["boot"] = True
        ans = SA.agent(obs_dict)          # resets SA tracker, returns deck
        ROT["est"] -= time.time() - t0
        return list(ans)
    sel = obs_dict["select"]
    try:
        rot = obs_dict.get("remainingOverageTime")
        if rot is None:
            rot = ROT["est"]
        sig = _sel_sig(obs_dict)
        logs_j = json.dumps(obs_dict.get("logs") or [], sort_keys=True)
        represented = sig == LAST["sig"]
        # accepted answers always advance obs["logs"] (delta since last
        # ACCEPTED selection); a rejection re-delivers them verbatim
        advanced = represented and logs_j != LAST["logs"]
        LAST["sig"], LAST["logs"] = sig, logs_j
        if not represented or advanced:
            REJ["sig"], REJ["tried"] = sig, []   # fresh decision instance
        # possible true rejection: same sig, logs did not advance
        ambiguous = represented and not advanced and bool(REJ["tried"])
        if rot < 30:
            # panic valve: stop paying for forwards, answer legally
            # (recover still wins there — reject loops must break even
            # during clock death)
            GUARD["panic"] += 1
            ans = _recover(sel) if ambiguous else _legal_random(sel)
        elif not represented:
            GUARD["fresh"] += 1
            ans = SA.agent(obs_dict)
        else:
            # re-presented sig: plain policy answer (no search exists
            # in this bundle; nothing to budget)
            ans = SA.agent(obs_dict)
            if ambiguous and ans in REJ["tried"]:
                # policy wants a possibly-rejected answer: never repeat
                ans = _recover(sel)
                GUARD["repeat_recover"] += 1
            else:
                GUARD["repeat_policy"] += 1
        REJ["tried"].append(ans)
        ROT["est"] -= time.time() - t0
        return ans
    except Exception:
        if os.environ.get("PTCG_DEBUG"):
            import traceback
            traceback.print_exc()
        try:
            return _legal_random(sel)
        except Exception:
            return [0] if sel.get("option") else []
