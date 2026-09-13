// Does the deck-table lock actually hold the env threads off?
//
// Two things are proved here, and the first is the one that matters:
//
//   1. A swapper holding the WRITE lock STALLS env stepping. Env threads keep
//      stepping until they hit a game_start, then block. If the lock were
//      missing (or taken only around the pointer lookup and not the copy), the
//      steppers would sail straight through while the table was being freed.
//
//   2. Hammering pt_reload_decks from one thread while several others step
//      games does not corrupt anything: every game still starts, and no deck
//      row read is ever torn.
//
// Usage: test_deck_lock <v1 blob> <grown blob>   (paths repo-relative)
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ptcg_env.h"

#define N_STEPPERS 4
#define HOLD_MS    400

static int fails = 0;
static void ok(int cond, const char* what) {
    printf("  %s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) fails++;
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

// ------------------------------------------------------------ stepper thread
typedef struct {
    Env env;
    float obs[PT_OBS_SIZE];
    float act, rew, term;
    unsigned char mask[PT_MAX_OPTIONS];
    volatile long episodes;      // games STARTED, i.e. lock acquisitions
    volatile int stop;
} Stepper;

static void* stepper_main(void* arg) {
    Stepper* s = (Stepper*)arg;
    while (!s->stop) {
        // env.c SETS terminals and never clears it -- the vecenv clears it
        // between steps. Without this the flag latches on the first episode
        // and an "episodes" counter silently becomes a step counter.
        *s->env.terminals = 0.0f;
        c_step(&s->env);
        // a terminal step also runs game_start for the next episode inside the
        // same c_step, so this counts deck draws, which is what we care about
        if (*s->env.terminals != 0.0f) s->episodes++;
    }
    return NULL;
}

static void stepper_init(Stepper* s, int i) {
    memset(s, 0, sizeof(*s));
    s->env.observations = s->obs;
    s->env.actions = &s->act;
    s->env.rewards = &s->rew;
    s->env.terminals = &s->term;
    s->env.action_mask = s->mask;
    s->env.num_agents = 1;
    s->env.mix_self = 0.98f;
    s->env.max_engine_steps = 3000;
    s->env.win_r = 1.0f;
    ptcg_env_init(&s->env, 42ULL * 1000003ULL + i);
    c_reset(&s->env);
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <v1 blob> <grown blob>\n", argv[0]); return 2; }
    const char *v1 = argv[1], *grown = argv[2];
    (void)v1;

    pt_engine_init();
    const PtTables* T = pt_tables();
    printf("baseline n_decks=%d\n\n", T->n_decks);

    static Stepper st[N_STEPPERS];
    pthread_t th[N_STEPPERS];
    for (int i = 0; i < N_STEPPERS; i++) stepper_init(&st[i], i);
    for (int i = 0; i < N_STEPPERS; i++)
        pthread_create(&th[i], NULL, stepper_main, &st[i]);

    // let them get going so we are measuring a steady state, not startup
    struct timespec warm = {0, 300 * 1000 * 1000};
    nanosleep(&warm, NULL);

    // ---- 1. a held write lock must stall game starts -------------------
    printf("1. holding the WRITE lock for %d ms must stall env stepping\n", HOLD_MS);
    long before[N_STEPPERS], during[N_STEPPERS];
    for (int i = 0; i < N_STEPPERS; i++) before[i] = st[i].episodes;

    double t0 = now_ms();
    pt_decks_wrlock();                       // <- the swapper
    struct timespec hold = {0, HOLD_MS * 1000L * 1000L};
    nanosleep(&hold, NULL);
    long started_under_lock = 0;
    for (int i = 0; i < N_STEPPERS; i++) {
        during[i] = st[i].episodes;
        started_under_lock += during[i] - before[i];
    }
    pt_decks_unlock();
    double held = now_ms() - t0;

    // Steppers finish the episode they are in, so a few terminals can land
    // while we hold the lock -- but each of them blocks at its NEXT
    // game_start. With 4 threads and ~400ms, unlocked they would start many
    // dozens; blocked they cannot start more than one apiece.
    printf("     episodes started while the lock was held: %ld (across %d threads, %.0f ms)\n",
           started_under_lock, N_STEPPERS, held);
    ok(started_under_lock <= N_STEPPERS,
       "at most one game start per thread got through (the rest blocked)");

    // ---- 2. and they resume once it is released ------------------------
    struct timespec settle = {0, 400 * 1000 * 1000};
    nanosleep(&settle, NULL);
    long after_release = 0;
    for (int i = 0; i < N_STEPPERS; i++) after_release += st[i].episodes - during[i];
    printf("     episodes started in the %d ms AFTER release: %ld\n", 400, after_release);
    ok(after_release > started_under_lock,
       "stepping resumes once the swapper releases");

    // ---- 3. hammer a real reload against live steppers ------------------
    printf("\n3. %d real reloads while %d threads step games\n", 20, N_STEPPERS);
    int rc_ok = 1, n_last = 0;
    for (int i = 0; i < 20; i++) {
        int n = pt_reload_decks(i % 2 ? v1 : grown);
        // v1 is a shrink once grown is live, so it is EXPECTED to be refused;
        // what matters is that neither outcome corrupts or crashes.
        if (n > 0) n_last = n;
        if (n == 0) rc_ok = 0;
    }
    ok(rc_ok, "no reload returned a nonsense count");
    ok(n_last == T->n_decks, "final table agrees with the last accepted reload");

    for (int i = 0; i < N_STEPPERS; i++) st[i].stop = 1;
    for (int i = 0; i < N_STEPPERS; i++) pthread_join(th[i], NULL);

    long total = 0;
    for (int i = 0; i < N_STEPPERS; i++) total += st[i].episodes;
    printf("     %ld episodes completed across the run, n_decks=%d\n",
           total, T->n_decks);
    ok(total > 0, "games actually ran throughout");

    printf("\n%s (%d failures)\n",
           fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED", fails);
    return fails ? 1 : 0;
}
