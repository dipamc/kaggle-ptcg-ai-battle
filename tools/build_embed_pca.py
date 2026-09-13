"""PCA-reduce raw text embeddings -> model-aligned static tables.

Dims chosen from an explained-variance / NN@10-overlap study:
cards 128 (0.77/0.81), attacks 64 (0.67/0.70), skills 64 (0.66/0.74) —
smooth curves, no elbow; sized so the text channel doesn't dwarf the
learned id embeddings (64/48/48). Projections are whitened then row
L2-normalized (comparable per-component scale for the token MLPs).

Output data/embed_pca.npz: cards (N_CARDS,128), attacks (N_ATTACKS,64),
skills (N_SKILLS,64) — row index = model table index (id-aligned; row 0
and slack rows are zero). Run after any embed_card_texts.py refresh.
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from ptcg.rl.buffers import N_CARDS, N_ATTACKS          # noqa: E402
from ptcg.rl.cards import N_SKILLS                      # noqa: E402

DIMS = {"cards": 128, "attacks": 64, "skills": 64}
SIZES = {"cards": N_CARDS, "attacks": N_ATTACKS, "skills": N_SKILLS}


def reduce(V, k):
    Vc = V - V.mean(0)
    U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
    P = (Vc @ Vt[:k].T) / (S[:k] / np.sqrt(len(V) - 1))   # whiten
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)


def main():
    raw = np.load(os.path.join(ROOT, "data", "embed_vectors_te3l.npz"))
    out = {}
    for kind, k in DIMS.items():
        ids, vecs = raw[f"{kind}_ids"], raw[f"{kind}_vecs"]
        table = np.zeros((SIZES[kind], k), dtype=np.float32)
        table[ids] = reduce(vecs, k)
        out[kind] = table
        print(f"{kind}: table {table.shape}, {len(ids)} rows filled")
    path = os.path.join(ROOT, "data", "embed_pca.npz")
    np.savez(path, **out)
    print(f"wrote {os.path.normpath(path)} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
