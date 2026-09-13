// GetBattleData JSON -> PObs. Field names per data/cg/api.py dataclasses.
#include <string.h>
#include <stdio.h>
#include "ptcg_env.h"

static int ji(const cJSON* o, const char* k) {
    const cJSON* v = cJSON_GetObjectItemCaseSensitive((cJSON*)o, k);
    if (!v || cJSON_IsNull(v)) return PT_NONE;
    if (cJSON_IsBool(v)) return cJSON_IsTrue(v) ? 1 : 0;
    if (!cJSON_IsNumber(v)) return PT_NONE;
    return (int)v->valuedouble;
}

static bool jb(const cJSON* o, const char* k) {
    const cJSON* v = cJSON_GetObjectItemCaseSensitive((cJSON*)o, k);
    return v && cJSON_IsTrue(v);
}

static void jcard(const cJSON* o, PCard* c) {
    if (!o || cJSON_IsNull(o)) { c->id = PT_NONE; c->serial = PT_NONE; c->playerIndex = PT_NONE; return; }
    c->id = ji(o, "id");
    c->serial = ji(o, "serial");
    c->playerIndex = ji(o, "playerIndex");
}

static void jpokemon(const cJSON* o, PPokemon* p) {
    memset(p, 0, sizeof(*p));
    if (!o || cJSON_IsNull(o)) { p->present = false; return; }
    p->present = true;
    p->id = ji(o, "id");
    p->serial = ji(o, "serial");
    p->hp = ji(o, "hp");
    p->maxHp = ji(o, "maxHp");
    p->appearThisTurn = jb(o, "appearThisTurn");
    const cJSON* it;
    const cJSON* arr = cJSON_GetObjectItemCaseSensitive((cJSON*)o, "energies");
    cJSON_ArrayForEach(it, arr) {
        if (p->n_energies < 16) p->energies[p->n_energies++] = (int)it->valuedouble;
    }
    arr = cJSON_GetObjectItemCaseSensitive((cJSON*)o, "energyCards");
    cJSON_ArrayForEach(it, arr) {
        if (p->n_energyCards < 16) jcard(it, &p->energyCards[p->n_energyCards++]);
    }
    arr = cJSON_GetObjectItemCaseSensitive((cJSON*)o, "tools");
    cJSON_ArrayForEach(it, arr) {
        if (p->n_tools < 4) jcard(it, &p->tools[p->n_tools++]);
    }
    arr = cJSON_GetObjectItemCaseSensitive((cJSON*)o, "preEvolution");
    cJSON_ArrayForEach(it, arr) {
        if (p->n_preEvolution < 4) jcard(it, &p->preEvolution[p->n_preEvolution++]);
    }
}

static void jplayer(const cJSON* o, PPlayer* pl) {
    memset(pl, 0, sizeof(*pl));
    const cJSON* it;
    const cJSON* arr = cJSON_GetObjectItemCaseSensitive((cJSON*)o, "active");
    cJSON_ArrayForEach(it, arr) {
        if (pl->n_active < 1) jpokemon(it, &pl->active[pl->n_active++]);
    }
    arr = cJSON_GetObjectItemCaseSensitive((cJSON*)o, "bench");
    cJSON_ArrayForEach(it, arr) {
        if (pl->n_bench < 8) jpokemon(it, &pl->bench[pl->n_bench++]);
    }
    pl->benchMax = ji(o, "benchMax");
    pl->deckCount = ji(o, "deckCount");
    pl->handCount = ji(o, "handCount");
    arr = cJSON_GetObjectItemCaseSensitive((cJSON*)o, "discard");
    cJSON_ArrayForEach(it, arr) {
        if (pl->n_discard < PT_MAX_LIST) jcard(it, &pl->discard[pl->n_discard++]);
    }
    arr = cJSON_GetObjectItemCaseSensitive((cJSON*)o, "prize");
    cJSON_ArrayForEach(it, arr) {
        if (pl->n_prize < 8) jcard(it, &pl->prize[pl->n_prize++]);
    }
    arr = cJSON_GetObjectItemCaseSensitive((cJSON*)o, "hand");
    if (!arr || cJSON_IsNull(arr)) pl->n_hand = -1;
    else cJSON_ArrayForEach(it, arr) {
        if (pl->n_hand < PT_MAX_LIST) jcard(it, &pl->hand[pl->n_hand++]);
    }
    pl->poisoned = jb(o, "poisoned");
    pl->burned = jb(o, "burned");
    pl->asleep = jb(o, "asleep");
    pl->paralyzed = jb(o, "paralyzed");
    pl->confused = jb(o, "confused");
}

int pt_obs_parse(const char* json, PObs* out) {
    memset(out, 0, sizeof(*out));
    cJSON* root = cJSON_Parse(json);
    if (!root) return -1;

    const cJSON* cur = cJSON_GetObjectItemCaseSensitive(root, "current");
    if (cur && !cJSON_IsNull(cur)) {
        out->has_current = true;
        out->turn = ji(cur, "turn");
        out->turnActionCount = ji(cur, "turnActionCount");
        out->yourIndex = ji(cur, "yourIndex");
        out->firstPlayer = ji(cur, "firstPlayer");
        out->result = ji(cur, "result");
        out->supporterPlayed = jb(cur, "supporterPlayed");
        out->stadiumPlayed = jb(cur, "stadiumPlayed");
        out->energyAttached = jb(cur, "energyAttached");
        out->retreated = jb(cur, "retreated");
        const cJSON* it;
        const cJSON* arr = cJSON_GetObjectItemCaseSensitive((cJSON*)cur, "stadium");
        cJSON_ArrayForEach(it, arr) {
            if (out->n_stadium < 1) jcard(it, &out->stadium[out->n_stadium++]);
        }
        arr = cJSON_GetObjectItemCaseSensitive((cJSON*)cur, "looking");
        if (!arr || cJSON_IsNull(arr)) out->n_looking = 0;
        else cJSON_ArrayForEach(it, arr) {
            if (out->n_looking < PT_MAX_DECKSEL) jcard(it, &out->looking[out->n_looking++]);
        }
        const cJSON* players = cJSON_GetObjectItemCaseSensitive((cJSON*)cur, "players");
        jplayer(cJSON_GetArrayItem((cJSON*)players, 0), &out->players[0]);
        jplayer(cJSON_GetArrayItem((cJSON*)players, 1), &out->players[1]);
    }

    const cJSON* sel = cJSON_GetObjectItemCaseSensitive(root, "select");
    if (sel && !cJSON_IsNull(sel)) {
        out->select.has_select = true;
        PSelect* s = &out->select;
        s->type = ji(sel, "type");
        s->context = ji(sel, "context");
        s->minCount = ji(sel, "minCount");
        s->maxCount = ji(sel, "maxCount");
        s->remainDamageCounter = ji(sel, "remainDamageCounter");
        s->remainEnergyCost = ji(sel, "remainEnergyCost");
        const cJSON* it;
        const cJSON* arr = cJSON_GetObjectItemCaseSensitive((cJSON*)sel, "option");
        cJSON_ArrayForEach(it, arr) {
            if (s->n_options >= PT_MAX_OPTS) break;
            POption* o = &s->option[s->n_options++];
            o->type = ji(it, "type");
            o->number = ji(it, "number");
            o->area = ji(it, "area");
            o->index = ji(it, "index");
            o->playerIndex = ji(it, "playerIndex");
            o->toolIndex = ji(it, "toolIndex");
            o->energyIndex = ji(it, "energyIndex");
            o->count = ji(it, "count");
            o->inPlayArea = ji(it, "inPlayArea");
            o->inPlayIndex = ji(it, "inPlayIndex");
            o->attackId = ji(it, "attackId");
            o->cardId = ji(it, "cardId");
            o->serial = ji(it, "serial");
            o->specialConditionType = ji(it, "specialConditionType");
        }
        arr = cJSON_GetObjectItemCaseSensitive((cJSON*)sel, "deck");
        if (!arr || cJSON_IsNull(arr)) s->n_deck = -1;
        else cJSON_ArrayForEach(it, arr) {
            if (s->n_deck < PT_MAX_DECKSEL) jcard(it, &s->deck[s->n_deck++]);
        }
        const cJSON* eff = cJSON_GetObjectItemCaseSensitive((cJSON*)sel, "effect");
        if (eff && !cJSON_IsNull(eff)) { s->has_effect = true; jcard(eff, &s->effect); }
        const cJSON* cc = cJSON_GetObjectItemCaseSensitive((cJSON*)sel, "contextCard");
        if (cc && !cJSON_IsNull(cc)) { s->has_contextCard = true; jcard(cc, &s->contextCard); }
    }

    const cJSON* logs = cJSON_GetObjectItemCaseSensitive(root, "logs");
    const cJSON* lg;
    cJSON_ArrayForEach(lg, logs) {
        if (out->n_logs >= PT_MAX_LOGS) break;
        PLog* L = &out->logs[out->n_logs++];
        L->type = ji(lg, "type");
        L->playerIndex = ji(lg, "playerIndex");
        L->cardId = ji(lg, "cardId");
        L->serial = ji(lg, "serial");
        L->fromArea = ji(lg, "fromArea");
        L->toArea = ji(lg, "toArea");
        L->serialActive = ji(lg, "serialActive");
        L->serialBench = ji(lg, "serialBench");
        L->serialBefore = ji(lg, "serialBefore");
        L->serialAfter = ji(lg, "serialAfter");
        L->serialTarget = ji(lg, "serialTarget");
        L->attackId = ji(lg, "attackId");
        L->value = ji(lg, "value");
        L->head = ji(lg, "head");
        L->result = ji(lg, "result");
        L->isRecover = ji(lg, "isRecover");
        L->reason = ji(lg, "reason");
    }

    cJSON_Delete(root);
    return 0;
}
