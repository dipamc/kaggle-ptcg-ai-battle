// Standalone self-drive test: N envs, uniform-random legal actions from the
// action mask, invariant checks. No pufferlib, no CUDA — runs anywhere the
// engine lib loads (macOS included).
//
// Invariants checked (mirroring the python env-level guarantees):
//   1. every encoded row offers >= 1 legal action
//   2. reward=win: nonzero rewards appear ONLY on terminal steps, |r| == 1
//   3. env-internal p0-currency invariant: log.ep_ret_bad == 0
//   4. no env_errors, no illegal actions when sampling from the mask
//   5. episodes complete (log.n advances)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ptcg_env.h"

static uint64_t rs = 0x853c49e6748fea9bULL;
static uint32_t xr(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (uint32_t)rs;
}

int main(int argc, char** argv) {
    int n_envs = argc > 1 ? atoi(argv[1]) : 16;
    long n_steps = argc > 2 ? atol(argv[2]) : 20000;
    // league mode: tag envs round-robin across n_banks frozen banks (plus
    // untagged controls). No network here — random actions drive BOTH seats —
    // but the full league row flow (opponent rows encoded, obs tag stamping,
    // per-bank stats, actor-currency rewards) is exercised.
    int n_banks = argc > 3 ? atoi(argv[3]) : 0;
    if (n_banks > PT_MAX_BANKS) n_banks = PT_MAX_BANKS;
    pt_engine_init();
    pt_tables();

    Env* envs = calloc(n_envs, sizeof(Env));
    float* obs = calloc((size_t)n_envs * PT_OBS_SIZE, sizeof(float));
    float* act = calloc(n_envs, sizeof(float));
    float* rew = calloc(n_envs, sizeof(float));
    float* term = calloc(n_envs, sizeof(float));
    unsigned char* mask = calloc((size_t)n_envs * PT_MAX_OPTIONS, 1);

    for (int i = 0; i < n_envs; i++) {
        Env* e = &envs[i];
        e->observations = obs + (size_t)i * PT_OBS_SIZE;
        e->actions = act + i;
        e->rewards = rew + i;
        e->terminals = term + i;
        e->action_mask = mask + (size_t)i * PT_MAX_OPTIONS;
        e->num_agents = 1;
        e->mix_self = 0.98f;
        e->max_engine_steps = 3000;
        e->win_r = 1.0f;
        e->prize_rewards = 0;
        if (n_banks > 0) e->tag = i % (n_banks + 1);   // 0 = untagged control
        ptcg_env_init(e, 42 * 1000003ULL + i);
        c_reset(e);
    }

    long bad_mask = 0, bad_reward = 0, terminals = 0, mid_reward = 0;
    long bad_tag = 0, tag_rows = 0;
    double ret_sum = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long s = 0; s < n_steps; s++) {
        for (int i = 0; i < n_envs; i++) {
            unsigned char* m = envs[i].action_mask;
            int legal[PT_MAX_OPTIONS], nl = 0;
            for (int j = 0; j < PT_MAX_OPTIONS; j++) if (m[j]) legal[nl++] = j;
            if (nl == 0) { bad_mask++; act[i] = 0; }
            else act[i] = (float)legal[xr() % nl];
        }
        memset(rew, 0, n_envs * sizeof(float));
        memset(term, 0, n_envs * sizeof(float));
        for (int i = 0; i < n_envs; i++) c_step(&envs[i]);
        for (int i = 0; i < n_envs; i++) {
            float fz = obs[(size_t)i * PT_OBS_SIZE + PT_FROZEN_COL];
            if (fz != 0.0f && fz != (float)envs[i].tag) bad_tag++;
            if (fz != 0.0f) tag_rows++;
        }
        for (int i = 0; i < n_envs; i++) {
            if (term[i] != 0.0f) {
                terminals++;
                float a = rew[i] < 0 ? -rew[i] : rew[i];
                if (a > 1.0f + 1e-5 || (a > 1e-5 && a < 1.0f - 1e-5)) bad_reward++;
                ret_sum += rew[i];
            } else if (rew[i] != 0.0f) {
                mid_reward++;   // win mode: rewards only at terminals
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    Log agg;
    memset(&agg, 0, sizeof(agg));
    for (int i = 0; i < n_envs; i++) {
        float* a = (float*)&agg;
        float* l = (float*)&envs[i].log;
        for (size_t k = 0; k < sizeof(Log) / sizeof(float); k++) a[k] += l[k];
    }
    double sps = (double)n_steps * n_envs / secs;
    printf("steps=%ld envs=%d  %.1f s  %.0f agent-steps/s/core\n",
           n_steps, n_envs, secs, sps);
    printf("episodes=%.0f terminals=%ld draws=%.0f mirror=%.0f rand=%.0f "
           "w_rand=%.0f ep_len=%.1f\n",
           agg.n, terminals, agg.draws, agg.eps_mirror, agg.eps_rand,
           agg.w_rand, agg.n ? agg.ep_len / agg.n : 0);
    printf("INVARIANTS: bad_mask=%ld bad_reward=%ld mid_reward=%ld "
           "ep_ret_bad=%.0f env_errors=%.0f illegal=%.0f\n",
           bad_mask, bad_reward, mid_reward, agg.ep_ret_bad, agg.env_errors,
           agg.illegal);
    printf("win_vs_rand=%.3f first_win_rate=%.3f turns=%.1f/%.1f "
           "engine_steps/ep=%.0f\n",
           agg.eps_rand ? agg.w_rand / agg.eps_rand : 0,
           agg.n - agg.draws > 0 ? agg.first_wins / (agg.n - agg.draws) : 0,
           agg.n ? agg.turns_first / agg.n : 0,
           agg.n ? agg.turns_second / agg.n : 0,
           agg.n ? agg.engine_steps / agg.n : 0);
    int league_fail = 0;
    if (n_banks > 0) {
        printf("LEAGUE: eps=%.0f w=%.0f win=%.3f len=%.1f tag_rows=%ld "
               "bad_tag=%ld per-bank:",
               agg.eps_league, agg.w_league,
               agg.eps_league ? agg.w_league / agg.eps_league : 0,
               agg.eps_league ? agg.league_len / agg.eps_league : 0,
               tag_rows, bad_tag);
        for (int b = 0; b < n_banks; b++)
            printf(" %d:%.0f/%.0f", b + 1, agg.league_w_b[b], agg.league_n_b[b]);
        printf("\n");
        // tagged envs must actually play league games, opponent rows must
        // surface (tag_rows > 0), tags must match the env's bank exactly,
        // and with random actions on both seats the win rate is ~0.5
        double wr = agg.eps_league ? agg.w_league / agg.eps_league : 0;
        league_fail = agg.eps_league < 1 || tag_rows < 1 || bad_tag > 0
                      || (agg.eps_league >= 200 && (wr < 0.35 || wr > 0.65));
    } else {
        league_fail = agg.eps_league > 0 || tag_rows > 0 || bad_tag > 0;
    }
    for (int i = 0; i < n_envs; i++) c_close(&envs[i]);
    int fail = bad_mask || bad_reward || mid_reward || agg.ep_ret_bad > 0
               || agg.env_errors > 0 || agg.illegal > 0 || agg.n < 1
               || league_fail;
    printf(fail ? "FAIL\n" : "OK\n");
    return fail;
}
