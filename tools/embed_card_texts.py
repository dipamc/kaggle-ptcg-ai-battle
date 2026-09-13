"""Embed data/embed_texts.json with OpenAI text-embedding-3-large.

Saves RAW 3072-dim float32 vectors to data/embed_vectors_te3l.npz
(arrays: {cards,attacks,skills}_ids / _vecs; row order = the JSON list
order, which for skills IS the model's skill vocab order — keep it).
PCA/reduction happens at model-integration time, not here.

Key: OPENAI_API_KEY from the environment.
Run: python3 tools/embed_card_texts.py     (~145k tokens, ~$0.02)
The npz is not committed; regenerate via this script.
"""
import json
import os
import time
import urllib.request

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "data", "embed_texts.json")
OUT = os.path.join(ROOT, "data", "embed_vectors_te3l.npz")
MODEL = "text-embedding-3-large"
BATCH = 256


def api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENAI_API_KEY in the environment")
    return key


def embed_batch(texts, key, tries=5):
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps({"model": MODEL, "input": texts}).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    for t in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            return [d["embedding"] for d in
                    sorted(data["data"], key=lambda d: d["index"])]
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code in (429, 500, 502, 503) and t < tries - 1:
                time.sleep(2 ** t)
                continue
            raise SystemExit(f"API error {e.code}: {body}")


def main():
    key = api_key()
    src = json.load(open(SRC))
    out = {}
    for kind in ("cards", "attacks", "skills"):
        rows = src[kind]
        vecs = []
        for i in range(0, len(rows), BATCH):
            vecs += embed_batch([r["text"] for r in rows[i:i + BATCH]], key)
            print(f"{kind}: {min(i + BATCH, len(rows))}/{len(rows)}")
        out[f"{kind}_ids"] = np.array([r["id"] for r in rows], dtype=np.int64)
        out[f"{kind}_vecs"] = np.array(vecs, dtype=np.float32)
    np.savez(OUT, **out)
    for kind in ("cards", "attacks", "skills"):
        v = out[f"{kind}_vecs"]
        print(f"{kind}: {v.shape}, norm mean {np.linalg.norm(v, axis=1).mean():.4f}")
    print(f"wrote {os.path.normpath(OUT)} "
          f"({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
