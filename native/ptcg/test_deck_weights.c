// Does per-deck sampling weighting actually shape the draw, and does it fail
// safe?
//
// The draw is the whole feature, so these tests measure the DISTRIBUTION
// rather than checking that a setter returned 0. Draws are collected by
// calling c_reset() in a loop: c_reset -> game_start -> draw_deck, which is
// the same path a real episode boundary takes, without playing the game out.
//
// Groups, not per-deck cells: with several hundred decks and a few tens of
// thousands of draws, a per-deck count is ~50 and proves little, while a
// half-pool group share has a standard error under half a percent.
//
// Usage: test_deck_weights <v1 blob> [grown blob]   (paths repo-relative)
//   with a grown blob it also covers append-under-weights.
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ptcg_env.h"

static int fails = 0;
static void ok(int cond, const char* what) {
    printf("  %s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) fails++;
}

// ------------------------------------------------------------------ helpers
static void env_init(Env* e, float* obs, float* act, float* rew, float* term,
                     unsigned char* mask, float mix_self, uint64_t seed) {
    memset(e, 0, sizeof(*e));
    e->observations = obs; e->actions = act; e->rewards = rew;
    e->terminals = term; e->action_mask = mask;
    e->num_agents = 1;
    e->mix_self = mix_self;          // 1.0 -> both seats draw TRAINING only
    e->max_engine_steps = 3000;
    e->win_r = 1.0f;
    ptcg_env_init(e, seed);
}

// Collect `n` deck draws into hist (length `width`, ids >= width counted in
// `over`, which is how coverage-pool draws are tallied since they live at
// PT_DECK_ALT_BASE).
static void draw_hist(Env* e, int n, long* hist, int width, long* over) {
    *over = 0;
    memset(hist, 0, (size_t)width * sizeof(long));
    for (int i = 0; i < n; i++) {
        c_reset(e);
        for (int s = 0; s < 2; s++) {
            int d = e->game.deck_idx[s];
            if (d >= 0 && d < width) hist[d]++;
            else if (d >= 0) (*over)++;
        }
    }
}

static double group_share(const long* hist, int lo, int hi, int n_decks) {
    long in = 0, all = 0;
    for (int i = 0; i < n_decks; i++) { all += hist[i]; if (i >= lo && i < hi) in += hist[i]; }
    return all ? (double)in / (double)all : -1.0;
}

static long count_range(const long* hist, int lo, int hi) {
    long s = 0;
    for (int i = lo; i < hi; i++) s += hist[i];
    return s;
}

#define DRAWS 12000

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <v1 blob> [grown blob]\n", argv[0]); return 2; }
    const char* grown = argc > 2 ? argv[2] : NULL;

    pt_engine_init();
    const PtTables* T = pt_tables();
    int n = T->n_decks, na = T->n_decks_alt;
    printf("baseline n_decks=%d n_decks_alt=%d, %d draws per measurement\n\n",
           n, na, DRAWS * 2);

    static float obs[PT_OBS_SIZE]; static unsigned char mask[PT_MAX_OPTIONS];
    static float act, rew, term;
    static Env e;
    long* hist = (long*)calloc((size_t)n + 8, sizeof(long));
    long over = 0;
    float* w = (float*)malloc((size_t)n * sizeof(float));

    // ---- 1. no weights == the plain uniform draw -----------------------
    printf("1. baseline: no weights set\n");
    env_init(&e, obs, &act, &rew, &term, mask, 1.0f, 12345ULL);
    draw_hist(&e, DRAWS, hist, n, &over);
    double half = group_share(hist, 0, n / 2, n);
    long zero_decks = 0;
    for (int i = 0; i < n; i++) if (hist[i] == 0) zero_decks++;
    printf("     first-half share %.4f (expect ~%.4f), decks never drawn: %ld\n",
           half, (double)(n / 2) / n, zero_decks);
    ok(fabs(half - (double)(n / 2) / n) < 0.02, "uniform draw is uniform");
    ok(zero_decks == 0, "every deck is reachable with no weights set");

    // ---- 2. group weights reshape the draw ------------------------------
    printf("\n2. first half weight 3, second half weight 1 -> expect 0.75\n");
    for (int i = 0; i < n; i++) w[i] = (i < n / 2) ? 3.0f : 1.0f;
    int active = pt_set_deck_weights(w, n);
    draw_hist(&e, DRAWS, hist, n, &over);
    half = group_share(hist, 0, n / 2, n);
    double want = 3.0 * (n / 2) / (3.0 * (n / 2) + (n - n / 2));
    printf("     active=%d  first-half share %.4f (expect %.4f)\n", active, half, want);
    ok(active == n, "all decks reported active");
    ok(fabs(half - want) < 0.02, "observed share matches the requested ratio");

    // ---- 3. weight 0 makes a deck UNREACHABLE, not just rare ------------
    printf("\n3. zeroing the first 50 decks\n");
    for (int i = 0; i < n; i++) w[i] = (i < 50) ? 0.0f : 1.0f;
    active = pt_set_deck_weights(w, n);
    draw_hist(&e, DRAWS, hist, n, &over);
    long zeroed = count_range(hist, 0, 50), rest = count_range(hist, 50, n);
    printf("     active=%d  draws from the zeroed 50: %ld   from the rest: %ld\n",
           active, zeroed, rest);
    ok(active == n - 50, "active count excludes the zeroed decks");
    ok(zeroed == 0, "a zero-weight deck is never drawn");
    ok(rest > 0, "the rest of the pool still plays");

    // ---- 4. scale invariance: coverage share must not move --------------
    // The random opponent draws from training+coverage slots. Weights are
    // normalised to mean 1 precisely so that scaling the file cannot change
    // how often it reaches into the coverage pool.
    if (na > 0) {
        printf("\n4. coverage share is invariant to the weight SCALE\n");
        static Env e2;
        env_init(&e2, obs, &act, &rew, &term, mask, 0.0f, 999ULL);  // random opp
        pt_clear_deck_weights();
        draw_hist(&e2, DRAWS, hist, n, &over);
        double cov_uniform = (double)over / (DRAWS * 2);
        for (int i = 0; i < n; i++) w[i] = 7.0f;          // uniform but scaled
        pt_set_deck_weights(w, n);
        draw_hist(&e2, DRAWS, hist, n, &over);
        double cov_scaled = (double)over / (DRAWS * 2);
        printf("     coverage share uniform %.4f vs all-weights-7 %.4f\n",
               cov_uniform, cov_scaled);
        ok(fabs(cov_uniform - cov_scaled) < 0.02,
           "scaling every weight leaves the coverage share alone");
    }

    // ---- 5. bad requests are refused AND change nothing ------------------
    printf("\n5. reject paths\n");
    for (int i = 0; i < n; i++) w[i] = (i < 50) ? 0.0f : 1.0f;
    pt_set_deck_weights(w, n);                   // known-good state to protect
    struct { const char* what; int len; int idx; float val; } bad[] = {
        {"wrong length",       n - 1, -1, 0.0f},
        {"negative weight",    n,      3, -1.0f},
        {"NaN weight",         n,      3, NAN},
        {"infinite weight",    n,      3, INFINITY},
    };
    for (size_t k = 0; k < sizeof(bad) / sizeof(bad[0]); k++) {
        float save = 0;
        if (bad[k].idx >= 0) { save = w[bad[k].idx]; w[bad[k].idx] = bad[k].val; }
        int rc = pt_set_deck_weights(w, bad[k].len);
        if (bad[k].idx >= 0) w[bad[k].idx] = save;
        printf("     %-18s -> %d\n", bad[k].what, rc);
        ok(rc == -1, bad[k].what);
    }
    float* zeros = (float*)calloc((size_t)n, sizeof(float));
    ok(pt_set_deck_weights(zeros, n) == -1, "all-zero weights");
    free(zeros);
    // the protected state must have survived every rejection
    draw_hist(&e, DRAWS, hist, n, &over);
    ok(count_range(hist, 0, 50) == 0,
       "a rejected request leaves the previous distribution untouched");

    // ---- 6. in-flight games keep the decks they were dealt ---------------
    printf("\n6. a game already running is not disturbed\n");
    static Env e3;
    env_init(&e3, obs, &act, &rew, &term, mask, 1.0f, 777ULL);
    for (int i = 0; i < n; i++) w[i] = 1.0f;
    pt_set_deck_weights(w, n);
    c_reset(&e3);
    int live0 = e3.game.deck_idx[0], live1 = e3.game.deck_idx[1];
    for (int i = 0; i < n; i++) w[i] = (i == live0 || i == live1) ? 0.0f : 1.0f;
    pt_set_deck_weights(w, n);                   // zero the decks in play
    int same = (e3.game.deck_idx[0] == live0 && e3.game.deck_idx[1] == live1);
    term = 0.0f;
    act = 0.0f;
    c_step(&e3);                                  // must not crash or reshuffle
    printf("     in-play decks %d,%d -> after zeroing them: %d,%d\n",
           live0, live1, e3.game.deck_idx[0], e3.game.deck_idx[1]);
    ok(same, "the running game keeps its decks");

    // ---- 7. append under weights ----------------------------------------
    if (grown) {
        printf("\n7. appending decks while weights are live\n");
        for (int i = 0; i < n; i++) w[i] = (i < n / 2) ? 3.0f : 1.0f;
        pt_set_deck_weights(w, n);
        int n2 = pt_reload_decks(grown);
        printf("     reload -> %d (was %d)\n", n2, n);
        if (n2 > n) {
            free(hist);
            hist = (long*)calloc((size_t)n2 + 8, sizeof(long));
            draw_hist(&e, DRAWS, hist, n2, &over);
            double old_first = (double)count_range(hist, 0, n / 2);
            double old_second = (double)count_range(hist, n / 2, n);
            double newdecks = (double)count_range(hist, n, n2);
            // appended decks enter at the mean, i.e. weight 1 after
            // normalisation -- the same rate as a second-half deck
            double per_new = newdecks / (n2 - n);
            double per_old2 = old_second / (n - n / 2);
            printf("     per-deck rate: appended %.1f vs old weight-1 %.1f "
                   "(old weight-3 %.1f)\n",
                   per_new, per_old2, old_first / (n / 2));
            ok(fabs(old_first / old_second - 3.0) < 0.35,
               "the 3:1 ratio among pre-existing decks survives the append");
            ok(per_new > 0 && fabs(per_new / per_old2 - 1.0) < 0.35,
               "appended decks enter at the mean weight");
        } else {
            printf("     (grown blob was not an append; skipping)\n");
        }
        n = n2 > n ? n2 : n;
    }

    // ---- 8. clearing returns to uniform ----------------------------------
    printf("\n8. clear_deck_weights\n");
    pt_clear_deck_weights();
    draw_hist(&e, DRAWS, hist, n, &over);
    half = group_share(hist, 0, n / 2, n);
    printf("     first-half share %.4f (expect ~%.4f)\n", half, (double)(n / 2) / n);
    ok(fabs(half - (double)(n / 2) / n) < 0.02, "back to uniform");

    printf("\n%s (%d failures)\n",
           fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED", fails);
    free(hist); free(w);
    return fails ? 1 : 0;
}
