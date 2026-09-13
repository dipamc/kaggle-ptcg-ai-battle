// pufferlib static-vec binding for the PTCG env.
#include <stdio.h>      // snprintf, for the anchor_<i> config keys
#include "ptcg_env.h"

#define OBS_SIZE 10944
#define NUM_ATNS 1
#define ACT_SIZES {64}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK 64
#define MY_VEC_INIT
#define MY_USES_TAGS    // league: env->tag = frozen bank id owning the row
#define MY_DECODER_EXPORT  // event log: mirror the decoder row to env->dec

// compile-time check against the header-derived layout
typedef char pt_obs_size_check[(OBS_SIZE == PT_OBS_SIZE) ? 1 : -1];
typedef char pt_mask_check[(MY_ACTION_MASK == PT_MAX_OPTIONS) ? 1 : -1];

#include "vecenv.h"

// League routing support: absolute obs column of the frozen-actor tag the
// env stamps on league opponent rows. The trainer (pufferlib.cu) resolves
// this weak symbol to know where to read (and zero, pre-forward) the tag.
int env_league_tag_col(void) { return PT_FROZEN_COL; }

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    DictItem* it;
    env->mix_self = (it = dict_get_unsafe(kwargs, "mix_self")) ? (float)it->value : 0.98f;
    env->max_engine_steps = (it = dict_get_unsafe(kwargs, "max_engine_steps")) ? (int)it->value : 3000;
    int reward_win = (it = dict_get_unsafe(kwargs, "reward_win")) ? (int)it->value : 1;
    env->win_r = reward_win ? 1.0f : 12.0f;
    env->prize_rewards = !reward_win;

    // Anchor decks. The Dict carries scalars only, so the index list arrives as
    // n_anchors + anchor_0..anchor_{n-1}. Defaults disable the feature
    // entirely: n_anchors 0 => every deck drawn uniformly.
    env->n_anchors = (it = dict_get_unsafe(kwargs, "n_anchors")) ? (int)it->value : 0;
    if (env->n_anchors > PT_MAX_ANCHORS) env->n_anchors = PT_MAX_ANCHORS;
    if (env->n_anchors < 0) env->n_anchors = 0;
    for (int i = 0; i < env->n_anchors; i++) {
        char key[24];
        snprintf(key, sizeof(key), "anchor_%d", i);
        env->anchor_idx[i] = (it = dict_get_unsafe(kwargs, key)) ? (int)it->value : 0;
    }
    env->p_anchor_learner =
        (it = dict_get_unsafe(kwargs, "p_anchor_learner")) ? (float)it->value : 0.0f;
    env->p_anchor_self_opp =
        (it = dict_get_unsafe(kwargs, "p_anchor_self_opp")) ? (float)it->value : 0.0f;
    env->p_anchor_league_opp =
        (it = dict_get_unsafe(kwargs, "p_anchor_league_opp")) ? (float)it->value : 0.0f;
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    pt_engine_init();
    (void)pt_tables();   // fail fast (and once) if the blob is missing
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;
    DictItem* sd = dict_get_unsafe(env_kwargs, "seed");
    uint64_t seed = sd ? (uint64_t)sd->value : 42;

    Env* envs = (Env*)calloc(total_agents, sizeof(Env));
    for (int i = 0; i < total_agents; i++) {
        my_init(&envs[i], env_kwargs);
        ptcg_env_init(&envs[i], seed * 1000003ULL + (uint64_t)i);
    }
    int buf = 0, buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < total_agents; i++) {
        buf_agents += 1;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }
    *num_envs_out = total_agents;
    return envs;
}

void my_log(Log* log, Dict* out) {
    // log arrives episode-averaged (vecenv divides every field by n)
    float eps = 1.0f;   // all fields are per-episode means
    dict_set(out, "ep_len", log->ep_len);
    dict_set(out, "prize_margin", log->prize_margin);
    float dec = eps - log->draws;
    dict_set(out, "first_win_rate", dec > 0 ? log->first_wins / dec : 0);
    dict_set(out, "turns_first", log->turns_first);
    dict_set(out, "turns_second", log->turns_second);
    float turns = log->turns_first + log->turns_second;
    dict_set(out, "dec_per_turn", turns > 0 ? log->ep_len / turns : 0);
    dict_set(out, "illegal", log->illegal);
    dict_set(out, "env_errors", log->env_errors);
    dict_set(out, "ep_ret_abs", log->ep_ret_abs);
    dict_set(out, "ep_ret_bad", log->ep_ret_bad);
    dict_set(out, "mirror_frac", log->eps_mirror);
    if (log->eps_rand > 0) {
        dict_set(out, "win_vs_rand", log->w_rand / log->eps_rand);
        dict_set(out, "score", log->w_rand / log->eps_rand);
    } else {
        dict_set(out, "score", 0.0);
    }
    if (log->decided > 0)
        dict_set(out, "prize_margin_dec", log->prize_margin_dec / log->decided);
    float end_tot = log->end_prize + log->end_bench + log->end_deck + log->end_other;
    if (end_tot > 0) {
        dict_set(out, "end_prize", log->end_prize / end_tot);
        dict_set(out, "end_bench", log->end_bench / end_tot);
        dict_set(out, "end_deck", log->end_deck / end_tot);
        if (log->end_other > 0) dict_set(out, "end_other", log->end_other / end_tot);
    }
    float wend_tot = log->wend_prize + log->wend_bench + log->wend_deck + log->wend_other;
    if (wend_tot > 0) {
        dict_set(out, "wend_prize", log->wend_prize / wend_tot);
        dict_set(out, "wend_bench", log->wend_bench / wend_tot);
        dict_set(out, "wend_deck", log->wend_deck / wend_tot);
        if (log->wend_other > 0) dict_set(out, "wend_other", log->wend_other / wend_tot);
    }
    float lend_tot = log->lend_prize + log->lend_bench + log->lend_deck + log->lend_other;
    if (lend_tot > 0) {
        dict_set(out, "lend_prize", log->lend_prize / lend_tot);
        dict_set(out, "lend_bench", log->lend_bench / lend_tot);
        dict_set(out, "lend_deck", log->lend_deck / lend_tot);
        if (log->lend_other > 0) dict_set(out, "lend_other", log->lend_other / lend_tot);
    }
    if (log->eps_league > 0) {
        dict_set(out, "league_frac", log->eps_league);
        dict_set(out, "win_vs_league", log->w_league / log->eps_league);
        dict_set(out, "league_len", log->league_len / log->eps_league);
        float lgw_tot = log->lgw_prize + log->lgw_bench + log->lgw_deck + log->lgw_other;
        if (lgw_tot > 0) {
            dict_set(out, "lgw_prize", log->lgw_prize / lgw_tot);
            dict_set(out, "lgw_bench", log->lgw_bench / lgw_tot);
            dict_set(out, "lgw_deck", log->lgw_deck / lgw_tot);
            if (log->lgw_other > 0) dict_set(out, "lgw_other", log->lgw_other / lgw_tot);
        }
        float lgl_tot = log->lgl_prize + log->lgl_bench + log->lgl_deck + log->lgl_other;
        if (lgl_tot > 0) {
            dict_set(out, "lgl_prize", log->lgl_prize / lgl_tot);
            dict_set(out, "lgl_bench", log->lgl_bench / lgl_tot);
            dict_set(out, "lgl_deck", log->lgl_deck / lgl_tot);
            if (log->lgl_other > 0) dict_set(out, "lgl_other", log->lgl_other / lgl_tot);
        }
        // Per-bank win rates, keyed by PYTHON bank index (0-based: the order
        // checkpoints were passed to load_frozen_bank; env tag = idx + 1).
        // dict_set stores the key POINTER, so keys must be static storage.
        static char kw[PT_MAX_BANKS][16], kn[PT_MAX_BANKS][16];
        for (int b = 0; b < PT_MAX_BANKS; b++) {
            if (log->league_n_b[b] <= 0) continue;
            if (!kw[b][0]) {
                snprintf(kw[b], sizeof(kw[b]), "league_w_%d", b);
                snprintf(kn[b], sizeof(kn[b]), "league_n_%d", b);
            }
            dict_set(out, kw[b], log->league_w_b[b] / log->league_n_b[b]);
            dict_set(out, kn[b], log->league_n_b[b]);
        }
    }
    dict_set(out, "engine_steps", log->engine_steps);
    dict_set(out, "episode_return", log->episode_return);

    // Per-deck TRAINING win rates, indexed by position in --anchor-decks.
    // deck_wr_<i> = win rate over decided games the learner played with that
    // deck; deck_n_<i> = those games per episode (i.e. that deck's share of
    // training games). Emitted only when the deck was actually seen this
    // interval, so a run with no anchors configured logs nothing new.
    // dict_set stores the key POINTER, so these MUST be static literals --
    // a snprintf'd stack buffer would dangle.
    static const char* DECK_WR_KEYS[PT_MAX_ANCHORS] = {
        "deck_wr_0","deck_wr_1","deck_wr_2","deck_wr_3","deck_wr_4","deck_wr_5",
        "deck_wr_6","deck_wr_7","deck_wr_8","deck_wr_9","deck_wr_10",
        "deck_wr_11","deck_wr_12","deck_wr_13","deck_wr_14","deck_wr_15"};
    static const char* DECK_N_KEYS[PT_MAX_ANCHORS] = {
        "deck_n_0","deck_n_1","deck_n_2","deck_n_3","deck_n_4","deck_n_5",
        "deck_n_6","deck_n_7","deck_n_8","deck_n_9","deck_n_10",
        "deck_n_11","deck_n_12","deck_n_13","deck_n_14","deck_n_15"};
    for (int i = 0; i < PT_MAX_ANCHORS; i++) {
        if (log->deck_games[i] > 0.0f) {
            dict_set(out, DECK_WR_KEYS[i], log->deck_wins[i] / log->deck_games[i]);
            dict_set(out, DECK_N_KEYS[i], log->deck_games[i]);
        }
    }
}
