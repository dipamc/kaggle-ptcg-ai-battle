// InfoTracker + EffectTracker + OracleLedger — line-for-line port of
// ptcg/tracker.py, ptcg/effects.py and ptcg/rl/oracle.py.
#include <string.h>
#include <stdio.h>
#include "ptcg_env.h"

// ---------------------------------------------------------------- helpers
static void ccount_clear(CCount c) { memset(c, 0, sizeof(uint16_t) * PT_N_CARDS); }

static int ccount_sum(const CCount c) {
    int s = 0;
    for (int i = 0; i < PT_N_CARDS; i++) s += c[i];
    return s;
}

static void ccount_from_deck(CCount c, const int32_t* deck60) {
    ccount_clear(c);
    for (int i = 0; i < 60; i++) {
        int id = deck60[i];
        if (id >= 0 && id < PT_N_CARDS) c[id]++;
    }
}

static inline void ccount_add(CCount c, int id, int n) {
    if (id >= 0 && id < PT_N_CARDS) c[id] = (uint16_t)(c[id] + n);
}

// a -= b, clamped at zero (python Counter subtraction semantics)
void pt_ccount_sub_clamp(uint16_t* a, const uint16_t* b) {
    for (int i = 0; i < PT_N_CARDS; i++) a[i] = a[i] > b[i] ? a[i] - b[i] : 0;
}
#define ccount_sub_clamp pt_ccount_sub_clamp

// "visible in a player's open zones" — tracker.py _visible_player_cards:
// board stacks (with energyCards/tools/preEvolution), discard, face-UP prizes
static void visible_player_cards(const PPlayer* pl, CCount seen) {
    ccount_clear(seen);
    for (int bi = -1; bi < pl->n_bench; bi++) {
        const PPokemon* pk = bi < 0 ? (pl->n_active ? &pl->active[0] : NULL)
                                    : &pl->bench[bi];
        if (!pk || !pk->present) continue;
        ccount_add(seen, pk->id, 1);
        for (int i = 0; i < pk->n_energyCards; i++) ccount_add(seen, pk->energyCards[i].id, 1);
        for (int i = 0; i < pk->n_tools; i++) ccount_add(seen, pk->tools[i].id, 1);
        for (int i = 0; i < pk->n_preEvolution; i++) ccount_add(seen, pk->preEvolution[i].id, 1);
    }
    for (int i = 0; i < pl->n_discard; i++) ccount_add(seen, pl->discard[i].id, 1);
    for (int i = 0; i < pl->n_prize; i++)
        if (pl->prize[i].id != PT_NONE) ccount_add(seen, pl->prize[i].id, 1);
}

// oracle.py _visible_out_of_deck: board stacks + discard + own stadium;
// prizes deliberately EXCLUDED
static void visible_out_of_deck(const PObs* obs, int pidx, CCount seen) {
    const PPlayer* pl = &obs->players[pidx];
    ccount_clear(seen);
    for (int bi = -1; bi < pl->n_bench; bi++) {
        const PPokemon* pk = bi < 0 ? (pl->n_active ? &pl->active[0] : NULL)
                                    : &pl->bench[bi];
        if (!pk || !pk->present) continue;
        ccount_add(seen, pk->id, 1);
        for (int i = 0; i < pk->n_energyCards; i++) ccount_add(seen, pk->energyCards[i].id, 1);
        for (int i = 0; i < pk->n_tools; i++) ccount_add(seen, pk->tools[i].id, 1);
        for (int i = 0; i < pk->n_preEvolution; i++) ccount_add(seen, pk->preEvolution[i].id, 1);
    }
    for (int i = 0; i < pl->n_discard; i++) ccount_add(seen, pl->discard[i].id, 1);
    for (int i = 0; i < obs->n_stadium; i++)
        if (obs->stadium[i].playerIndex == pidx) ccount_add(seen, obs->stadium[i].id, 1);
}

// =============================================================== EffectTracker
void effstate_init(EffState* e) {
    memset(e, 0, sizeof(*e));
    e->first_player = -1;
    e->actives[0] = e->actives[1] = PT_NONE;
    e->last_attack_serial[0] = e->last_attack_serial[1] = PT_NONE;
    e->last_ko_turn[0] = e->last_ko_turn[1] = -9;
    smap_clear(&e->evolved_now);
    smap_clear(&e->healed_now);
}

static int fx_turn_player(const EffState* e, int t) {
    if (e->first_player < 0) return -1;
    return (t % 2 == 1) ? e->first_player : 1 - e->first_player;
}

static int fx_next_turn_of(const EffState* e, int seat, int t) {
    return fx_turn_player(e, t + 1) == seat ? t + 1 : t + 2;
}

static void fx_bind_range(const EffState* e, const FxRec* rec, int owner, int t,
                          int* a, int* b) {
    switch (rec->duration) {
    case FXD_SAME_TURN: *a = t; *b = t; return;
    case FXD_MY_NEXT: *a = *b = fx_next_turn_of(e, owner, t); return;
    case FXD_OPP_NEXT: *a = *b = fx_next_turn_of(e, 1 - owner, t); return;
    case FXD_UNTIL_END_MY_NEXT: *a = t; *b = fx_next_turn_of(e, owner, t); return;
    case FXD_UNTIL_END_OPP_NEXT: *a = t; *b = fx_next_turn_of(e, 1 - owner, t); return;
    case FXD_WHILE_CONDITION: *a = t; *b = PT_NONE; return;
    default: *a = t; *b = t + 2; return;    // OTHER: coarse
    }
}

static void fx_add_inst(EffState* e, const FxRec* rec, int bind_from, int bind_to,
                        bool soft, int serial, int player) {
    if (e->n_inst >= PT_MAX_FX_INST) return;   // effects are rare; cap is deep
    e->inst[e->n_inst++] = (FxInst){rec, bind_from, bind_to, soft, serial, player, true};
}

// _register(rec, owner, defender, t, soft, src). from_attack distinguishes
// the attack_fx table (source_kind == "attack") for the SELF-target rule.
static void fx_register(EffState* e, const FxRec* rec, int owner, int defender,
                        int t, bool soft, bool from_attack) {
    int a, b;
    fx_bind_range(e, rec, owner, t, &a, &b);
    if (rec->lock) {
        int tgt = rec->target == FXT_OPP_PLAYER ? 1 - owner : owner;
        fx_add_inst(e, rec, a, b, soft, PT_NONE, tgt);
        return;
    }
    if (rec->kind < 0) return;
    int serial = PT_NONE;
    if (rec->target == FXT_SELF) {
        serial = from_attack ? e->last_attack_serial[owner] : PT_NONE;
    } else if (rec->target == FXT_DEFENDING) {
        serial = defender;
    } else if ((rec->target == FXT_ALL_MY || rec->target == FXT_ALL_OPP)
               && rec->kind == 2) {         // EFF_DMG_DEALT_MOD -> dmg_boost
        int tgt = rec->target == FXT_ALL_MY ? owner : 1 - owner;
        fx_add_inst(e, rec, a, b, soft, PT_NONE, 100 + tgt);  // 100+seat = boost
        return;
    }
    if (serial != PT_NONE) fx_add_inst(e, rec, a, b, soft, serial, PT_NONE);
}

static void fx_clear_serial(EffState* e, int serial) {
    if (serial == PT_NONE) return;
    for (int i = 0; i < e->n_inst; i++)
        if (e->inst[i].serial == serial) e->inst[i].live = false;
}

static void fx_compact(EffState* e) {
    int w = 0;
    for (int i = 0; i < e->n_inst; i++)
        if (e->inst[i].live) e->inst[w++] = e->inst[i];
    e->n_inst = w;
}

static bool fx_coin_after(const PObs* obs, int i) {
    for (int j = i + 1; j < obs->n_logs; j++) {
        int ty = obs->logs[j].type;
        if (ty == LOG_COIN) return obs->logs[j].head == 1;
        if (ty == LOG_ATTACK || ty == LOG_TURN_END || ty == LOG_PLAY) return false;
    }
    return false;
}

static void fx_expire(EffState* e, int t) {
    for (int i = 0; i < e->n_inst; i++)
        if (e->inst[i].bind_to != PT_NONE && t > e->inst[i].bind_to)
            e->inst[i].live = false;
    fx_compact(e);
}

static void fx_update(EffState* e, const PObs* obs) {
    if (!obs->has_current) return;
    const PtTables* T = pt_tables();
    if (e->first_player < 0) e->first_player = obs->firstPlayer;

    // side-wide ability shields + per-holder attachment shields
    bool shielded[2] = {false, false};
    SMap holder_shield;
    smap_clear(&holder_shield);
    for (int s = 0; s < 2; s++) {
        const PPlayer* pl = &obs->players[s];
        for (int bi = -1; bi < pl->n_bench; bi++) {
            const PPokemon* pk = bi < 0 ? (pl->n_active ? &pl->active[0] : NULL)
                                        : &pl->bench[bi];
            if (!pk || !pk->present) continue;
            if (pk->id >= 0 && pk->id < PT_N_CARDS && T->shields[pk->id]) shielded[s] = true;
            bool hs = false;
            for (int i = 0; i < pk->n_energyCards; i++)
                if (pk->energyCards[i].id >= 0 && pk->energyCards[i].id < PT_N_CARDS
                    && T->shields[pk->energyCards[i].id]) hs = true;
            for (int i = 0; i < pk->n_tools; i++)
                if (pk->tools[i].id >= 0 && pk->tools[i].id < PT_N_CARDS
                    && T->shields[pk->tools[i].id]) hs = true;
            if (hs && pk->serial != PT_NONE) smap_put(&holder_shield, pk->serial, 1);
        }
    }

    int t = e->turn;
    for (int i = 0; i < obs->n_logs; i++) {
        const PLog* lg = &obs->logs[i];
        int ty = lg->type, p = lg->playerIndex;
        if (ty == LOG_TURN_START) {
            t += 1;
            smap_clear(&e->evolved_now);
            smap_clear(&e->healed_now);
        } else if (ty == LOG_TURN_END) {
            int ended = fx_turn_player(e, t);
            if ((ended == 0 || ended == 1) && e->supporter_now[ended]) {
                e->supporter_prev[ended] = e->supporter_now[ended];
                e->supporter_now[ended] = 0;
            }
        } else if (ty == LOG_MOVE_CARD) {
            int serial = lg->serial, fa = lg->fromArea, ta = lg->toArea;
            if (ta == AREA_ACTIVE) {
                if (p == 0 || p == 1) e->actives[p] = serial;
            } else if (fa == AREA_ACTIVE) {
                if ((p == 0 || p == 1) && e->actives[p] == serial) e->actives[p] = PT_NONE;
                fx_clear_serial(e, serial);
            } else if (fa == AREA_BENCH && (ta == AREA_DISCARD || ta == AREA_HAND
                                            || ta == AREA_DECK)) {
                fx_clear_serial(e, serial);
            }
            if ((fa == AREA_ENERGY || fa == AREA_TOOL) && ta == AREA_DISCARD
                && (p == 0 || p == 1) && fx_turn_player(e, t) == 1 - p) {
                if (fa == AREA_ENERGY) e->energy_removed[p]++;
                else e->tools_removed[p]++;
            }
            if ((fa == AREA_ACTIVE || fa == AREA_BENCH) && ta == AREA_DISCARD
                && (p == 0 || p == 1) && lg->cardId >= 0 && lg->cardId < PT_N_CARDS
                && T->pokemon_ids[lg->cardId]) {
                e->last_ko_turn[p] = t;
            }
        } else if (ty == LOG_SWITCH) {
            int old = lg->serialActive, nw = lg->serialBench;
            if (p == 0 || p == 1) e->actives[p] = nw;
            fx_clear_serial(e, old);
            if ((p == 0 || p == 1) && fx_turn_player(e, t) == 1 - p)
                e->forced_switches[p]++;
        } else if (ty == LOG_CHANGE) {
            if (p == 0 || p == 1) e->actives[p] = lg->serialAfter;
            fx_clear_serial(e, lg->serialBefore);
        } else if (ty == LOG_EVOLVE || ty == LOG_DEVOLVE) {
            int old = lg->serialTarget, nw = lg->serial;
            if ((p == 0 || p == 1) && e->actives[p] == old) e->actives[p] = nw;
            fx_clear_serial(e, old);
            if (ty == LOG_EVOLVE && nw != PT_NONE) smap_put(&e->evolved_now, nw, 1);
        } else if (ty == LOG_HP_CHANGE) {
            if (lg->value != PT_NONE && lg->value > 0 && lg->serial != PT_NONE)
                smap_put(&e->healed_now, lg->serial, 1);
        } else if (ty == LOG_ATTACK) {
            int aid = lg->attackId, serial = lg->serial;
            if (p == 0 || p == 1) {
                e->last_attack_id[p] = aid == PT_NONE ? 0 : aid;
                e->last_attack_serial[p] = serial;
            }
            if (aid >= 0 && aid < PT_N_ATTACKS && (p == 0 || p == 1)) {
                int head = T->attack_fx_head[aid], cnt = T->attack_fx_cnt[aid];
                for (int r = 0; r < (head < 0 ? 0 : cnt); r++) {
                    const FxRec* rec = &T->attack_fx[head + r];
                    bool ok = true, soft = false;
                    if (rec->condition == FXC_COIN_HEADS) ok = fx_coin_after(obs, i);
                    else if (rec->condition == FXC_COIN_TAILS
                             || rec->condition == FXC_OTHER) soft = true;
                    int defender = e->actives[1 - p];
                    if (rec->target == FXT_DEFENDING
                        && (shielded[1 - p]
                            || (defender != PT_NONE
                                && smap_get(&holder_shield, defender) != PT_NONE)))
                        soft = true;
                    if (ok) fx_register(e, rec, p, e->actives[1 - p], t, soft, true);
                }
            }
        } else if (ty == LOG_PLAY) {
            int cid = lg->cardId;
            if (cid >= 0 && cid < PT_N_CARDS && (p == 0 || p == 1)) {
                if (T->supporters[cid]) e->supporter_now[p] = cid;
                int head = T->play_fx_head[cid], cnt = T->play_fx_cnt[cid];
                for (int r = 0; r < (head < 0 ? 0 : cnt); r++) {
                    const FxRec* rec = &T->play_fx[head + r];
                    bool soft = rec->condition != FXC_NONE;
                    fx_register(e, rec, p, e->actives[1 - p], t, soft, false);
                }
            }
        }
    }

    if (t != obs->turn) e->turn_drift++;
    e->turn = obs->turn;
    fx_expire(e, e->turn);
    // re-seed running actives from obs ground truth
    for (int s = 0; s < 2; s++) {
        const PPlayer* pl = &obs->players[s];
        if (pl->n_active && pl->active[0].present)
            e->actives[s] = pl->active[0].serial;
    }
}

static float fx_strength(const FxInst* x, int t) {
    float v = (x->bind_from <= t && (x->bind_to == PT_NONE || t <= x->bind_to))
                  ? 1.0f : 0.5f;
    return v * (x->soft ? 0.5f : 1.0f);
}

// flag-name -> value for one board Pokemon at turn t (writes into tok_f row)
void fx_token_flags(const EffState* e, int serial, int t, float* f) {
    if (serial == PT_NONE) return;
    for (int i = 0; i < e->n_inst; i++) {
        const FxInst* x = &e->inst[i];
        if (x->serial != serial) continue;
        const FxRec* rec = x->rec;
        float s = fx_strength(x, t);
        float mag = rec->magnitude;
        if (rec->kind == 1 || rec->kind == 2) {          // signed sums
            f[rec->flag_tokf] += rec->sign * s * mag / 100.0f;
        } else if (rec->kind == 3) {                     // delayed dmg: max
            float v = s * mag / 10.0f;
            if (v > f[rec->flag_tokf]) f[rec->flag_tokf] = v;
        } else {                                         // plain max
            if (s > f[rec->flag_tokf]) f[rec->flag_tokf] = s;
        }
    }
}

// (lock-kind -> value, dmg_boost) for one seat at turn t
void fx_player_flags(const EffState* e, int seat, int t,
                     float locks[4], float* boost) {
    locks[0] = locks[1] = locks[2] = locks[3] = 0.0f;
    *boost = 0.0f;
    for (int i = 0; i < e->n_inst; i++) {
        const FxInst* x = &e->inst[i];
        if (x->player == seat && x->rec->lock) {
            float s = fx_strength(x, t);
            if (s > locks[x->rec->lock]) locks[x->rec->lock] = s;
        } else if (x->player == 100 + seat) {            // dmg_boost
            *boost += fx_strength(x, t) * x->rec->magnitude / 100.0f;
        }
    }
}

int fx_last_attack_id(const EffState* e, int seat, int active_serial) {
    if (e->last_attack_serial[seat] == PT_NONE) return 0;
    return e->last_attack_serial[seat] == active_serial
               ? e->last_attack_id[seat] : 0;
}

// =============================================================== InfoTracker
void tracker_init(Tracker* t, const int32_t* deck60) {
    memcpy(t->my_deck_list, deck60, 60 * sizeof(int32_t));
    smap_clear(&t->opp_known_hand);
    smap_clear(&t->opp_known_deck);
    ccount_clear(t->my_prize_known);
    t->my_prize_known_set = false;
    effstate_init(&t->fx);
}

static void tracker_consume_logs(Tracker* t, const PObs* obs, int opp_index) {
    for (int i = 0; i < obs->n_logs; i++) {
        const PLog* lg = &obs->logs[i];
        if (lg->type != LOG_MOVE_CARD || lg->playerIndex != opp_index) continue;
        int serial = lg->serial, card_id = lg->cardId;
        if (serial == PT_NONE || card_id == PT_NONE || card_id == 0) continue;
        int ta = lg->toArea, fa = lg->fromArea;
        if (ta == AREA_HAND) {
            smap_put(&t->opp_known_hand, serial, card_id);
            smap_del(&t->opp_known_deck, serial);
        } else if (fa == AREA_HAND) {
            smap_del(&t->opp_known_hand, serial);
        }
        if (ta == AREA_DECK) {
            smap_put(&t->opp_known_deck, serial, card_id);
        } else if (fa == AREA_DECK) {
            smap_del(&t->opp_known_deck, serial);
        }
    }
}

static void tracker_maintain_prize(Tracker* t, const PObs* obs, int you) {
    if (!t->my_prize_known_set) return;
    for (int i = 0; i < obs->n_logs; i++) {
        const PLog* lg = &obs->logs[i];
        if (lg->playerIndex != you) continue;
        if (lg->type == LOG_MOVE_CARD && lg->fromArea == AREA_PRIZE
            && lg->cardId != PT_NONE && lg->cardId != 0) {
            if (lg->cardId < PT_N_CARDS && t->my_prize_known[lg->cardId] > 0)
                t->my_prize_known[lg->cardId]--;
        } else if (lg->toArea == AREA_PRIZE) {
            if (lg->type == LOG_MOVE_CARD && lg->cardId != PT_NONE && lg->cardId != 0) {
                ccount_add(t->my_prize_known, lg->cardId, 1);
            } else {                                     // face-down / unknown
                t->my_prize_known_set = false;
                return;
            }
        }
    }
}

// serial visible anywhere in my zones? (tracker.py _serial_visible)
static bool serial_visible(const PObs* obs, int you, int serial) {
    if (serial == PT_NONE) return false;
    const PPlayer* me = &obs->players[you];
    for (int bi = -1; bi < me->n_bench; bi++) {
        const PPokemon* pk = bi < 0 ? (me->n_active ? &me->active[0] : NULL)
                                    : &me->bench[bi];
        if (!pk || !pk->present) continue;
        if (pk->serial == serial) return true;
        for (int i = 0; i < pk->n_energyCards; i++)
            if (pk->energyCards[i].serial == serial) return true;
        for (int i = 0; i < pk->n_tools; i++)
            if (pk->tools[i].serial == serial) return true;
        for (int i = 0; i < pk->n_preEvolution; i++)
            if (pk->preEvolution[i].serial == serial) return true;
    }
    if (me->n_hand > 0)
        for (int i = 0; i < me->n_hand; i++)
            if (me->hand[i].serial == serial) return true;
    for (int i = 0; i < me->n_discard; i++)
        if (me->discard[i].serial == serial) return true;
    for (int i = 0; i < obs->n_stadium; i++)
        if (obs->stadium[i].serial == serial) return true;
    return false;
}

// my (deck ∪ prizes) multiset incl. the mid-effect-limbo correction
void tracker_my_unseen(const Tracker* t, const PObs* obs, CCount out) {
    int you = obs->yourIndex;
    const PPlayer* me = &obs->players[you];
    CCount seen;
    visible_player_cards(me, seen);
    if (me->n_hand > 0)
        for (int i = 0; i < me->n_hand; i++) ccount_add(seen, me->hand[i].id, 1);
    for (int i = 0; i < obs->n_stadium; i++)
        if (obs->stadium[i].playerIndex == you) ccount_add(seen, obs->stadium[i].id, 1);
    ccount_from_deck(out, t->my_deck_list);
    ccount_sub_clamp(out, seen);

    const PSelect* sel = &obs->select;
    if (sel->has_select && sel->has_effect && sel->effect.playerIndex == you
        && sel->effect.id >= 0 && sel->effect.id < PT_N_CARDS
        && out[sel->effect.id] > 0
        && !serial_visible(obs, you, sel->effect.serial)) {
        out[sel->effect.id]--;
    }
}

static void tracker_learn_deck_reveal(Tracker* t, const PObs* obs, int you) {
    const PSelect* sel = &obs->select;
    if (!sel->has_select || sel->n_deck <= 0) return;
    int n_mine = 0;
    CCount mine_ids;
    ccount_clear(mine_ids);
    for (int i = 0; i < sel->n_deck; i++) {
        const PCard* c = &sel->deck[i];
        if (c->playerIndex == you) {
            n_mine++;
            ccount_add(mine_ids, c->id, 1);
        } else if (c->serial != PT_NONE && c->id != PT_NONE) {
            smap_put(&t->opp_known_deck, c->serial, c->id);
        }
    }
    const PPlayer* me = &obs->players[you];
    if (n_mine > 0 && n_mine == me->deckCount) {
        CCount inferred;
        tracker_my_unseen(t, obs, inferred);
        ccount_sub_clamp(inferred, mine_ids);
        int n_down = 0;
        for (int i = 0; i < me->n_prize; i++)
            if (me->prize[i].id == PT_NONE) n_down++;
        if (ccount_sum(inferred) == n_down) {
            memcpy(t->my_prize_known, inferred, sizeof(inferred));
            t->my_prize_known_set = true;
        }
    }
}

void tracker_update(Tracker* t, const PObs* obs) {
    if (!obs->has_current) return;
    int you = obs->yourIndex;
    tracker_consume_logs(t, obs, 1 - you);
    tracker_maintain_prize(t, obs, you);
    fx_update(&t->fx, obs);
    tracker_learn_deck_reveal(t, obs, you);
}

// opponent known-hand / known-deck id multisets for the encoder
void tracker_opp_known_hand(const Tracker* t, CCount out) {
    ccount_clear(out);
    for (int i = 0; i < PT_MAP_CAP; i++)
        if (t->opp_known_hand.keys[i] != -1)
            ccount_add(out, t->opp_known_hand.vals[i], 1);
}

void tracker_opp_known_deck(const Tracker* t, CCount out) {
    ccount_clear(out);
    for (int i = 0; i < PT_MAP_CAP; i++)
        if (t->opp_known_deck.keys[i] != -1)
            ccount_add(out, t->opp_known_deck.vals[i], 1);
}

// =============================================================== OracleLedger
void oracle_init(Oracle* o, const int32_t* d0, const int32_t* d1) {
    ccount_from_deck(o->decks[0], d0);
    ccount_from_deck(o->decks[1], d1);
    ccount_clear(o->hand[0]); ccount_clear(o->hand[1]);
    ccount_clear(o->prize[0]); ccount_clear(o->prize[1]);
    o->frozen = false;
}

// one VisualizeData parse: prize six per seat from the LAST frame
static bool oracle_parse_prizes(Oracle* o, PtVizFn vizfn, void* ctx) {
    const char* viz = vizfn ? vizfn(ctx) : NULL;
    if (!viz) return false;
    cJSON* root = cJSON_Parse(viz);
    if (!root) return false;
    cJSON* frames = root;
    if (!cJSON_IsArray(root)) {
        static const char* keys[] = {"entry", "frames", "visualize"};
        frames = NULL;
        for (int i = 0; i < 3; i++) {
            cJSON* v = cJSON_GetObjectItemCaseSensitive(root, keys[i]);
            if (v && cJSON_IsArray(v)) { frames = v; break; }
        }
        if (!frames) { cJSON_Delete(root); return false; }
    }
    int n = cJSON_GetArraySize(frames);
    if (n == 0) { cJSON_Delete(root); return false; }
    cJSON* last = cJSON_GetArrayItem(frames, n - 1);
    cJSON* cur = cJSON_GetObjectItemCaseSensitive(last, "current");
    cJSON* players = cur ? cJSON_GetObjectItemCaseSensitive(cur, "players") : NULL;
    if (!players) { cJSON_Delete(root); return false; }
    bool ok = false;
    for (int s = 0; s < 2; s++) {
        cJSON* pl = cJSON_GetArrayItem(players, s);
        cJSON* prize = pl ? cJSON_GetObjectItemCaseSensitive(pl, "prize") : NULL;
        if (!prize) continue;
        int np = cJSON_GetArraySize(prize), nid = 0;
        int ids[8];
        cJSON* c;
        cJSON_ArrayForEach(c, prize) {
            if (cJSON_IsNull(c)) continue;
            cJSON* id = cJSON_GetObjectItemCaseSensitive(c, "id");
            if (id && cJSON_IsNumber(id) && nid < 8) ids[nid++] = (int)id->valuedouble;
        }
        if (nid == np && nid > 0) {
            ccount_clear(o->prize[s]);
            for (int i = 0; i < nid; i++) ccount_add(o->prize[s], ids[i], 1);
            ok = true;
        }
    }
    cJSON_Delete(root);
    return ok;
}

void oracle_on_seat_obs(Oracle* o, const PObs* obs, PtVizFn viz, void* ctx) {
    if (!obs->has_current) return;
    int s = obs->yourIndex;
    if (!o->frozen) {
        if (obs->players[0].n_prize == 6 && obs->players[1].n_prize == 6) {
            if (oracle_parse_prizes(o, viz, ctx) && obs->turn >= 1)
                o->frozen = true;
        }
    } else {
        for (int i = 0; i < obs->n_logs; i++) {
            const PLog* lg = &obs->logs[i];
            if (lg->playerIndex != s || lg->type != LOG_MOVE_CARD) continue;
            if (lg->fromArea == AREA_PRIZE && lg->cardId != PT_NONE && lg->cardId != 0) {
                if (lg->cardId < PT_N_CARDS && o->prize[s][lg->cardId] > 0)
                    o->prize[s][lg->cardId]--;
            } else if (lg->toArea == AREA_PRIZE && lg->cardId != PT_NONE
                       && lg->cardId != 0) {
                ccount_add(o->prize[s], lg->cardId, 1);
            }
        }
    }
    if (obs->players[s].n_hand >= 0) {
        ccount_clear(o->hand[s]);
        for (int i = 0; i < obs->players[s].n_hand; i++)
            ccount_add(o->hand[s], obs->players[s].hand[i].id, 1);
    }
}

// emit(): opp_hand / opp_deck / opp_prize / my_prize for the acting seat
void oracle_emit(const Oracle* o, const PObs* obs,
                 CCount opp_hand, CCount opp_deck,
                 CCount opp_prize, CCount my_prize) {
    int me = obs->yourIndex, opp = 1 - me;
    const PPlayer* opp_pl = &obs->players[opp];

    memcpy(opp_hand, o->hand[opp], sizeof(CCount));
    int n_now = opp_pl->handCount;
    int n_snap = ccount_sum(opp_hand);
    if (n_now > n_snap) {
        ccount_add(opp_hand, PT_UNK_CARD, n_now - n_snap);
    } else if (n_now < n_snap) {
        // drop from the highest card ids first (python: sorted reverse)
        for (int cid = PT_N_CARDS - 1; cid >= 0 && n_snap > n_now; cid--) {
            int drop = opp_hand[cid] < n_snap - n_now ? opp_hand[cid] : n_snap - n_now;
            opp_hand[cid] -= drop;
            n_snap -= drop;
        }
    }

    CCount visible;
    visible_out_of_deck(obs, opp, visible);
    memcpy(opp_deck, o->decks[opp], sizeof(CCount));
    ccount_sub_clamp(opp_deck, visible);
    ccount_sub_clamp(opp_deck, o->prize[opp]);
    CCount hand_known;                       // opp_hand minus its UNK rows
    memcpy(hand_known, opp_hand, sizeof(CCount));
    hand_known[PT_UNK_CARD] = 0;
    ccount_sub_clamp(opp_deck, hand_known);

    int deck_total = opp_pl->deckCount;
    int n_deck = ccount_sum(opp_deck);
    if (n_deck > deck_total) {
        opp_deck[PT_UNK_CARD] = 0;
        n_deck = ccount_sum(opp_deck);
        int excess = n_deck - deck_total;
        for (int cid = PT_N_CARDS - 1; cid >= 0 && excess > 0; cid--) {
            int drop = opp_deck[cid] < excess ? opp_deck[cid] : excess;
            opp_deck[cid] -= drop;
            excess -= drop;
        }
    } else if (n_deck < deck_total) {
        ccount_add(opp_deck, PT_UNK_CARD, deck_total - n_deck);
    }

    // prizes: exact when ledger total matches public count, else UNK rows
    for (int side = 0; side < 2; side++) {
        int seat = side == 0 ? opp : me;
        uint16_t* dst = side == 0 ? opp_prize : my_prize;
        int n = obs->players[seat].n_prize;
        memcpy(dst, o->prize[seat], sizeof(CCount));
        if (ccount_sum(dst) != n) {
            ccount_clear(dst);
            ccount_add(dst, PT_UNK_CARD, n);
        }
    }
}
