"""Static per-card / per-attack tables from engine data (model-side).

Built once per process from all_card_data()/all_attack(); the model
gathers through these to build CardVec/AttackVec/SkillVec fresh each
forward (cheap: ~1.3k card rows), which is the practical form of the
"precomputed lookup tables after each weight update" in docs/model.md.

Skill vocab: distinct (name, text) pairs, index 0 = NULL — own vocab,
no engine ids (§1). v4: frozen text-embedding channels ride alongside
the id embeddings — data/embed_pca.npz (tools/build_embed_pca.py),
row-aligned to these tables (skills MUST keep this vocab order).
"""
import os

import numpy as np

from .buffers import N_CARDS, N_ATTACKS

N_SKILLS = 421 + 27  # measured 421 distinct + slack; index 0 = NULL
MAX_ATTACKS = 4
MAX_SKILLS = 2
CARD_STATIC_F = 7 + 12 + 8 + 3 + 26 + 2   # = 58
ATTACK_STATIC_F = 15
CARD_TEXT_F, ATTACK_TEXT_F, SKILL_TEXT_F = 128, 64, 64

_PCA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "data", "embed_pca.npz")


def load_text_tables():
    z = np.load(_PCA)
    c, a, s = z["cards"], z["attacks"], z["skills"]
    assert c.shape == (N_CARDS, CARD_TEXT_F), c.shape
    assert a.shape == (N_ATTACKS, ATTACK_TEXT_F), a.shape
    assert s.shape == (N_SKILLS, SKILL_TEXT_F), s.shape
    return c, a, s


def build_tables():
    from cg.api import all_card_data, all_attack

    cards, attacks = all_card_data(), all_attack()

    att_static = np.zeros((N_ATTACKS, ATTACK_STATIC_F), dtype=np.float32)
    for a in attacks:
        f = att_static[a.attackId]
        f[0] = a.damage / 300.0
        for e in a.energies:
            if 0 <= e < 12:
                f[1 + e] += 1 / 4.0
        f[13] = len(a.energies) / 5.0
        f[14] = float(bool(a.text))

    card_static = np.zeros((N_CARDS, CARD_STATIC_F), dtype=np.float32)
    card_attacks = np.zeros((N_CARDS, MAX_ATTACKS), dtype=np.int64)
    card_skills = np.zeros((N_CARDS, MAX_SKILLS), dtype=np.int64)
    skill_vocab: dict[tuple, int] = {}

    for c in cards:
        i = c.cardId
        f = card_static[i]
        f[c.cardType] = 1.0                      # 0..6
        f[7 + c.energyType] = 1.0                # 7..18
        f[19:27] = [c.basic, c.stage1, c.stage2, c.ex, c.megaEx, c.tera,
                    c.aceSpec, c.evolvesFrom is not None]
        f[27] = c.hp / 300.0
        f[28] = c.retreatCost / 5.0
        f[29] = (3 if c.megaEx else 2 if c.ex else 1) / 3.0  # prize liability
        f[30 + (c.weakness if c.weakness is not None else 12)] = 1.0
        f[43 + (c.resistance if c.resistance is not None else 12)] = 1.0
        f[56] = len(c.attacks) / 4.0
        f[57] = len(c.skills) / 2.0
        for j, aid in enumerate(c.attacks[:MAX_ATTACKS]):
            card_attacks[i, j] = aid
        for j, s in enumerate(c.skills[:MAX_SKILLS]):
            key = (s.name, s.text)
            sid = skill_vocab.setdefault(key, len(skill_vocab) + 1)
            if sid < N_SKILLS:
                card_skills[i, j] = sid

    assert len(skill_vocab) < N_SKILLS, f"skill vocab overflow: {len(skill_vocab)}"
    return card_static, card_attacks, card_skills, att_static


def build_env_tables():
    """Numpy-only tables for the ENV encoder's derived features (§3):
    attack affordability/damage/deficit, weakness/resistance matchups,
    retreat affordability. No torch dependency — safe in env workers."""
    from cg.api import all_card_data, all_attack

    atk_damage = np.zeros(N_ATTACKS, dtype=np.float32)
    atk_cost = np.zeros((N_ATTACKS, 12), dtype=np.int8)  # count per EnergyType
    for a in all_attack():
        atk_damage[a.attackId] = a.damage
        for e in a.energies:
            if 0 <= e < 12:
                atk_cost[a.attackId, e] += 1

    energy_type = np.zeros(N_CARDS, dtype=np.int8)
    weakness = np.full(N_CARDS, -1, dtype=np.int8)
    resistance = np.full(N_CARDS, -1, dtype=np.int8)
    retreat = np.zeros(N_CARDS, dtype=np.int8)
    top_attacks = np.zeros((N_CARDS, 4), dtype=np.int64)  # sorted by damage desc
    for c in all_card_data():
        i = c.cardId
        energy_type[i] = c.energyType
        weakness[i] = -1 if c.weakness is None else c.weakness
        resistance[i] = -1 if c.resistance is None else c.resistance
        retreat[i] = c.retreatCost
        for j, aid in enumerate(sorted(c.attacks, key=lambda a: -atk_damage[a])[:4]):
            top_attacks[i, j] = aid
    return dict(atk_damage=atk_damage, atk_cost=atk_cost,
                energy_type=energy_type, weakness=weakness,
                resistance=resistance, retreat=retreat,
                top_attacks=top_attacks)
