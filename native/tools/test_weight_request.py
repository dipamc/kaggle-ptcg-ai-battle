#!/usr/bin/env python3
"""Unit tests for the deck-weights REQUEST FILE: the operator-facing surface.

Imports train_native for real (its module-level imports are stdlib only --
torch and the CUDA extension load lazily inside functions), so these exercise
the shipped parser rather than a copy of it.

    python3 native/tools/test_weight_request.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_native as T                                    # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'} {what}")
    if not cond:
        fails += 1


def parse(text):
    with tempfile.NamedTemporaryFile("w", suffix=".weights", delete=False) as fh:
        fh.write(text)
        p = fh.name
    try:
        return T.read_weight_request(p)
    finally:
        os.unlink(p)


print("1. a well-formed request")
ver, dflt, per = parse("version 3\ndefault 0.5\nalpha 0.0\nbeta 2\n")
ok(ver == 3, "version parsed")
ok(dflt == 0.5, "default parsed")
ok(per == {"alpha": 0.0, "beta": 2.0}, "per-deck weights parsed")

print("\n2. comments, blanks and extra whitespace")
ver, dflt, per = parse("# leading comment\n\nversion 4   \n  alpha 1.5  # trailing\n\n")
ok((ver, per) == (4, {"alpha": 1.5}), "comments and blank lines ignored")
ok(dflt == 1.0, "default defaults to 1.0")

print("\n3. malformed input is SKIPPED, never raised")
for bad, why in [
    ("version\n", "a key with no value"),
    ("version x\n", "a non-integer version"),
    ("version 1\nalpha notanumber\n", "a non-numeric weight"),
    ("version 1\nalpha 1 2 3\n", "too many fields"),
]:
    try:
        v, _, _ = parse(bad)
        ok(v == 0, f"{why} -> version 0 (skip)")
    except Exception as e:                                   # noqa: BLE001
        ok(False, f"{why} raised {type(e).__name__}")

print("\n4. absent file is not an error")
ok(T.read_weight_request("/nonexistent/path.weights") == (0, 1.0, {}),
   "missing file reads as no request")
ok(T.read_weight_request(None) == (0, 1.0, {}), "None path reads as no request")

print("\n5. a request with no version line is inert")
ver, _, per = parse("default 0.0\nalpha 1\n")
ok(ver == 0, "no version -> nothing is applied")

print("\n6. weight_vector maps names to deck-id order")
names = ["a", "b", "c", "d"]
w, unknown = T.weight_vector(names, 1.0, {"c": 0.0, "a": 3.0})
ok(w == [3.0, 1.0, 0.0, 1.0], "named decks placed at their ids, rest defaulted")
ok(unknown == [], "no unknown names")

print("\n7. one unknown name fails the WHOLE request")
w, unknown = T.weight_vector(names, 1.0, {"a": 0.0, "zzz": 1.0})
ok(w is None and unknown == ["zzz"],
   "an unknown name rejects everything rather than weighting the wrong decks")

print("\n8. the id order is the pool's sorted 60-line glob")
# Checked against the pinned full-pool listing (tools/pool_names.py --write
# data/pool_names.txt), when the checkout carries one; it is generated rather
# than tracked, so it can be absent in a fresh clone.
listing = "data/pool_names.txt"
if not os.path.exists(listing) or not os.path.isdir("decks/pool"):
    print("  SKIP no data/pool_names.txt listing (tools/pool_names.py --write)")
else:
    expected = [l.strip() for l in open(listing) if l.strip()]
    pool_names, _ = T.pool_deck_names("decks/pool")
    ok(pool_names == expected,
       f"pool_deck_names matches the pinned listing ({len(expected)})")

print(f"\n{'SOME CHECKS FAILED' if fails else 'ALL CHECKS PASSED'} ({fails} failures)")
sys.exit(1 if fails else 0)
