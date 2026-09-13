// Stall-cap test for the Mega Venusaur ex (card 652) Solar Transfer loop
// (docs/training.md). Run against a blob whose TRAINING pool is
// Venusaur decks:
//
//   PTCG_TABLES=<stall blob> ./build/test_stall [n_envs] [n_steps] [mix_self]
//
// The driver is STALL-GREEDY: at every decision, if any mask-legal option is
// an activation of card 652's ability, it picks one (else uniform-random
// legal). Without the cap that loops inside one turn until max_engine_steps
// kills the episode; with the cap this asserts:
//   1. no (seat, turn) ever exceeds PT_AB652_CAP driver-counted activations
//   2. the cap BINDS and is airtight: some turns reach exactly the cap, and
//      an over-budget activation is never mask-legal again that turn
//   3. zero truncations: every episode ends decided (or an honest draw)
//   4. the standard invariants (mask nonempty, |r| == win_r at terminals
//      only, ep_ret_bad == 0, env_errors == 0, illegal == 0)
// mix_self 1.0 drives both seats stall-greedy (mirror); 0.0 plays the random
// opponent, exercising the random_answer filter on that seat.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ptcg_env.h"

#define STALL_CARD 652
#define STALL_CAP  10          // must match PT_AB652_CAP in env.c

static uint64_t rs = 0x9e3779b97f4a7c15ULL;
static uint32_t xr(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (uint32_t)rs;
}

// Independent copy of the env's activation predicate — the test measures the
// env, so it must not share its code. Mirrors encoder.c resolve_card for the
// board slot the option points at.
static int stall_opt(const PObs* obs, const POption* o) {
    if (o->type != OT_ABILITY && o->type != OT_SKILL) return 0;
    int cid = o->cardId == PT_NONE ? 0 : o->cardId;
    if (!cid) {
        int p = o->playerIndex == PT_NONE ? obs->yourIndex : o->playerIndex;
        int idx = o->index == PT_NONE ? -1 : o->index;
        if (p < 0 || p > 1 || idx < 0) return 0;
        const PPlayer* pl = &obs->players[p];
        if (o->area == AREA_ACTIVE)
            cid = (idx < pl->n_active && pl->active[idx].present)
                      ? pl->active[idx].id : 0;
        else if (o->area == AREA_BENCH && idx < pl->n_bench)
            cid = pl->bench[idx].id;
    }
    return cid == STALL_CARD;
}

typedef struct { int turn, cnt[2]; } TurnCnt;

int main(int argc, char** argv) {
    int n_envs = argc > 1 ? atoi(argv[1]) : 8;
    long n_steps = argc > 2 ? atol(argv[2]) : 60000;
    float mix = argc > 3 ? (float)atof(argv[3]) : 1.0f;
    pt_engine_init();
    pt_tables();

    Env* envs = calloc(n_envs, sizeof(Env));
    float* obs = calloc((size_t)n_envs * PT_OBS_SIZE, sizeof(float));
    float* act = calloc(n_envs, sizeof(float));
    float* rew = calloc(n_envs, sizeof(float));
    float* term = calloc(n_envs, sizeof(float));
    unsigned char* mask = calloc((size_t)n_envs * PT_MAX_OPTIONS, 1);
    TurnCnt* tc = calloc(n_envs, sizeof(TurnCnt));

    for (int i = 0; i < n_envs; i++) {
        Env* e = &envs[i];
        e->observations = obs + (size_t)i * PT_OBS_SIZE;
        e->actions = act + i;
        e->rewards = rew + i;
        e->terminals = term + i;
        e->action_mask = mask + (size_t)i * PT_MAX_OPTIONS;
        e->num_agents = 1;
        e->mix_self = mix;
        e->max_engine_steps = 3000;
        e->win_r = 1.0f;
        e->prize_rewards = 0;
        ptcg_env_init(e, 777 * 1000003ULL + i);
        c_reset(e);
        tc[i].turn = -1;
    }

    long bad_mask = 0, bad_reward = 0, mid_reward = 0, terminals = 0;
    long stall_picks = 0, over_cap = 0, offered_over = 0, cap_turns = 0;
    int max_per_turn = 0;
    for (long s = 0; s < n_steps; s++) {
        for (int i = 0; i < n_envs; i++) {
            const PObs* po = &envs[i].game.obs;
            const PSelect* sel = &po->select;
            if (tc[i].turn != po->turn) {
                tc[i].turn = po->turn;
                tc[i].cnt[0] = tc[i].cnt[1] = 0;
            }
            unsigned char* m = envs[i].action_mask;
            int legal[PT_MAX_OPTIONS], nl = 0;
            int stall[PT_MAX_OPTIONS], ns = 0;
            int n = sel->n_options < PT_MAX_OPTIONS - 1 ? sel->n_options
                                                        : PT_MAX_OPTIONS - 1;
            for (int j = 0; j < PT_MAX_OPTIONS; j++) {
                if (!m[j]) continue;
                legal[nl++] = j;
                if (j < n && sel->has_select && stall_opt(po, &sel->option[j]))
                    stall[ns++] = j;
            }
            int seat = po->yourIndex;
            int c = (seat == 0 || seat == 1) ? tc[i].cnt[seat] : 0;
            if (ns > 0 && c >= STALL_CAP) offered_over++;   // invariant 2
            int a;
            if (nl == 0) { bad_mask++; a = 0; }
            else if (ns > 0) {
                a = stall[xr() % ns];
                stall_picks++;
                if (seat == 0 || seat == 1) {
                    int v = ++tc[i].cnt[seat];
                    if (v > max_per_turn) max_per_turn = v;
                    if (v > STALL_CAP) over_cap++;          // invariant 1
                    if (v == STALL_CAP) cap_turns++;
                }
            } else a = legal[xr() % nl];
            act[i] = (float)a;
        }
        memset(rew, 0, n_envs * sizeof(float));
        memset(term, 0, n_envs * sizeof(float));
        for (int i = 0; i < n_envs; i++) c_step(&envs[i]);
        for (int i = 0; i < n_envs; i++) {
            if (term[i] != 0.0f) {
                terminals++;
                tc[i].turn = -1;      // episode over: force a counter reset
                float a = rew[i] < 0 ? -rew[i] : rew[i];
                if (a > 1.0f + 1e-5 || (a > 1e-5 && a < 1.0f - 1e-5))
                    bad_reward++;
            } else if (rew[i] != 0.0f) {
                mid_reward++;
            }
        }
    }

    Log agg;
    memset(&agg, 0, sizeof(agg));
    for (int i = 0; i < n_envs; i++) {
        float* a = (float*)&agg;
        float* l = (float*)&envs[i].log;
        for (size_t k = 0; k < sizeof(Log) / sizeof(float); k++) a[k] += l[k];
    }
    long trunc = (long)(agg.n - agg.decided - agg.draws);

    printf("steps=%ld envs=%d mix_self=%.2f\n", n_steps, n_envs, (double)mix);
    printf("episodes=%.0f decided=%.0f draws=%.0f TRUNC=%ld ep_len=%.1f "
           "engine_steps/ep=%.0f\n",
           agg.n, agg.decided, agg.draws, trunc,
           agg.n ? agg.ep_len / agg.n : 0,
           agg.n ? agg.engine_steps / agg.n : 0);
    printf("STALL: picks=%ld picks/ep=%.1f max_per_turn=%d capped_turns=%ld "
           "over_cap=%ld offered_over=%ld\n",
           stall_picks, agg.n ? stall_picks / agg.n : 0, max_per_turn,
           cap_turns, over_cap, offered_over);
    printf("INVARIANTS: bad_mask=%ld bad_reward=%ld mid_reward=%ld "
           "ep_ret_bad=%.0f env_errors=%.0f illegal=%.0f terminals=%ld\n",
           bad_mask, bad_reward, mid_reward, agg.ep_ret_bad, agg.env_errors,
           agg.illegal, terminals);

    // cap_turns == 0 means the driver never even reached the cap: the
    // predicate missed (wrong option type / card id) and the test is vacuous.
    int fail = over_cap > 0 || offered_over > 0 || trunc > 0 || bad_mask > 0
               || bad_reward > 0 || mid_reward > 0 || agg.ep_ret_bad > 0
               || agg.env_errors > 0 || agg.illegal > 0 || agg.n < 5
               || cap_turns == 0 || stall_picks == 0;
    for (int i = 0; i < n_envs; i++) c_close(&envs[i]);
    printf(fail ? "FAIL\n" : "OK\n");
    return fail;
}
