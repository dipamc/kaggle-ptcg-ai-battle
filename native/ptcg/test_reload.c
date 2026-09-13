// Full append-contract test for pt_reload_decks, per docs/deck-pool.md.
// argv: <v1 blob> <grown blob (v1+3, names sort last)> <inserted blob (v1+1 at front)>
//
// Order matters: each guard must be exercised from a baseline where it is the
// guard that actually fires. The reorder blob has n0+1 decks, so it must be
// tried while the pool is still n0 -- from an n0+3 pool it would trip the
// removal guard instead and prove nothing about prefix checking.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ptcg_env.h"

static int fails = 0;
static void ok(int cond, const char* what) {
    printf("  %s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) fails++;
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s v1 grown inserted\n", argv[0]); return 2; }
    const char *v1 = argv[1], *grown = argv[2], *inserted = argv[3];

    const PtTables* T = pt_tables();
    int n0 = T->n_decks, a0 = T->n_decks_alt;
    printf("baseline: n_decks=%d n_decks_alt=%d\n\n", n0, a0);

    size_t pre_bytes = (size_t)n0 * 60 * sizeof(int32_t);
    int32_t* snapshot = malloc(pre_bytes);
    memcpy(snapshot, T->decks, pre_bytes);
    size_t alt_bytes = (size_t)a0 * 60 * sizeof(int32_t);
    int32_t* alt_snap = malloc(alt_bytes ? alt_bytes : 1);
    if (a0) memcpy(alt_snap, T->decks_alt, alt_bytes);

    // ---- 1. REORDER, from the n0 baseline so the count check passes and the
    //         prefix check is the thing under test.
    printf("1. REORDER (deck inserted at the FRONT, n0+1 decks) must be REJECTED\n");
    int r = pt_reload_decks(inserted);
    ok(r == -1, "returns -1 (prefix differs, not a count problem)");
    ok(T->n_decks == n0, "pool NOT mutated");
    ok(memcmp(T->decks, snapshot, pre_bytes) == 0, "rows untouched");

    // ---- 2. APPEND
    printf("\n2. APPEND (v1 + 3, names sort last) must be ACCEPTED\n");
    r = pt_reload_decks(grown);
    ok(r == n0 + 3, "returns old_n + 3");
    ok(T->n_decks == n0 + 3, "n_decks grew by 3");
    ok(memcmp(T->decks, snapshot, pre_bytes) == 0,
       "every pre-existing deck row is bytewise unchanged");
    ok(T->n_decks_alt == a0, "coverage count unchanged");
    ok(a0 == 0 || memcmp(T->decks_alt, alt_snap, alt_bytes) == 0,
       "coverage contents unchanged");

    printf("\n3. the 3 NEW rows are reachable and non-empty\n");
    int nonzero = 0;
    for (int i = n0; i < T->n_decks; i++)
        for (int c = 0; c < 60; c++)
            if (T->decks[(size_t)i * 60 + c] != 0) { nonzero++; break; }
    ok(nonzero == 3, "all 3 appended rows have content");

    // ---- 4. SHRINK
    printf("\n4. SHRINK (back to v1) must be REJECTED\n");
    r = pt_reload_decks(v1);
    ok(r == -1, "returns -1");
    ok(T->n_decks == n0 + 3, "pool NOT mutated by the rejected shrink");

    // ---- 5. idempotent re-append
    printf("\n5. re-applying the SAME grown blob is a no-op, not an error\n");
    r = pt_reload_decks(grown);
    ok(r == n0 + 3, "returns the same count");
    ok(memcmp(T->decks, snapshot, pre_bytes) == 0, "prefix still intact");

    printf("\n%s (%d failures)\n",
           fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED", fails);
    free(snapshot); free(alt_snap);
    return fails ? 1 : 0;
}
