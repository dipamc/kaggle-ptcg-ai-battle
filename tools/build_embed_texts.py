"""Render card/attack/skill texts for the text-embedding side channel.

Three lists aligned to the model's embedding tables (ptcg/rl/cards.py):
- cards[i]  <-> card_id_emb rows, keyed by engine cardId (full card sheet)
- attacks[i] <-> attack_id_emb rows, keyed by engine attackId
- skills[i]  <-> skill_id_emb rows, keyed by the SAME vocab construction
  as cards.build_tables (distinct (name, text) of c.skills[:2] in
  all_card_data() order, 1-based, 0 = NULL) — do not reorder.

Output: data/embed_texts.json. The embedding step (API call, once a key
is available) reads this and saves raw full-dim vectors; PCA happens at
model-integration time.

Run: PYTHONPATH=data python3 tools/build_embed_texts.py
"""
import json
import os
import re

from cg.api import all_card_data, all_attack, CardType, EnergyType

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data", "embed_texts.json")

ENERGY_NAME = {
    EnergyType.COLORLESS: "Colorless", EnergyType.GRASS: "Grass",
    EnergyType.FIRE: "Fire", EnergyType.WATER: "Water",
    EnergyType.LIGHTNING: "Lightning", EnergyType.PSYCHIC: "Psychic",
    EnergyType.FIGHTING: "Fighting", EnergyType.DARKNESS: "Darkness",
    EnergyType.METAL: "Metal", EnergyType.DRAGON: "Dragon",
    EnergyType.RAINBOW: "Rainbow", EnergyType.TEAM_ROCKET: "Team Rocket",
}
SYMBOL = {"{G}": "Grass", "{R}": "Fire", "{W}": "Water", "{L}": "Lightning",
          "{P}": "Psychic", "{F}": "Fighting", "{D}": "Darkness",
          "{M}": "Metal", "{C}": "Colorless", "{N}": "Dragon",
          "{ex}": "ex", "{ACE SPEC}": "ACE SPEC", "{V}": "V"}
_sym_re = re.compile("|".join(re.escape(k) for k in SYMBOL))


def expand(text):
    return _sym_re.sub(lambda m: SYMBOL[m.group(0)], text or "").strip()


def cost_str(energies):
    return " ".join(ENERGY_NAME[e] for e in energies) if energies else "free"


def attack_line(a):
    line = f"{a.name} — cost {cost_str(a.energies)}"
    if a.damage:
        line += f", {a.damage} damage"
    if a.text:
        line += f". {expand(a.text)}"
    return line


KIND = {CardType.ITEM: "Item", CardType.TOOL: "Pokémon Tool",
        CardType.SUPPORTER: "Supporter", CardType.STADIUM: "Stadium",
        CardType.BASIC_ENERGY: "Basic Energy",
        CardType.SPECIAL_ENERGY: "Special Energy"}


def card_text(c, attacks_by_id):
    name = expand(c.name)
    if c.cardType != CardType.POKEMON:
        parts = [f"{name} — {KIND[c.cardType]}."]
        if c.aceSpec:
            parts.append("ACE SPEC (only 1 ACE SPEC card allowed per deck).")
        for s in c.skills:
            body = expand(s.text)
            if expand(s.name) != name:
                body = f"{expand(s.name)}: {body}"
            parts.append(body)
        return " ".join(p for p in parts if p)

    stage = ("Basic" if c.basic else "Stage 1" if c.stage1
             else "Stage 2" if c.stage2 else "")
    kind = f"{stage} Pokémon".strip()
    if c.megaEx:
        kind = f"Mega Evolution {kind} ex"
    elif c.ex:
        kind += " ex"
    head = f"{name} — {kind}"
    if c.evolvesFrom:
        head += f" (evolves from {expand(c.evolvesFrom)})"
    head += f", {ENERGY_NAME[c.energyType]} type, {c.hp} HP"
    if c.weakness is not None:
        head += f", weakness {ENERGY_NAME[c.weakness]}"
    if c.resistance is not None:
        head += f", resistance {ENERGY_NAME[c.resistance]}"
    head += f", retreat cost {c.retreatCost}."
    parts = [head]
    if c.tera:
        parts.append("Tera Pokémon: takes no damage from attacks while on the Bench.")
    if c.megaEx:
        parts.append("When Knocked Out, the opponent takes 3 Prize cards.")
    elif c.ex:
        parts.append("When Knocked Out, the opponent takes 2 Prize cards.")
    for s in c.skills:
        parts.append(f"Ability: {expand(s.name)} — {expand(s.text)}")
    for aid in c.attacks:
        parts.append(f"Attack: {attack_line(attacks_by_id[aid])}")
    return " ".join(parts)


def main():
    cards, attacks = all_card_data(), all_attack()
    attacks_by_id = {a.attackId: a for a in attacks}

    card_rows = [{"id": c.cardId, "text": card_text(c, attacks_by_id)}
                 for c in cards]
    attack_rows = [{"id": a.attackId, "text": attack_line(a)} for a in attacks]

    # replicate cards.build_tables skill vocab EXACTLY (order + [:2] cap)
    skill_vocab = {}
    for c in cards:
        for s in c.skills[:2]:
            skill_vocab.setdefault((s.name, s.text), len(skill_vocab) + 1)
    skill_rows = [{"id": sid, "text": f"{expand(name)}. {expand(text)}"}
                  for (name, text), sid in skill_vocab.items()]

    out = {"meta": {"n_cards": len(card_rows), "n_attacks": len(attack_rows),
                    "n_skills": len(skill_rows),
                    "skill_vocab": "cards.build_tables order, 1-based, 0=NULL"},
           "cards": card_rows, "attacks": attack_rows, "skills": skill_rows}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    chars = sum(len(r["text"]) for rows in (card_rows, attack_rows, skill_rows)
                for r in rows)
    print(f"cards {len(card_rows)}, attacks {len(attack_rows)}, "
          f"skills {len(skill_rows)}; {chars} chars (~{chars // 4} tokens)")
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
