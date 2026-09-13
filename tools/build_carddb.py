#!/usr/bin/env python3
"""Build data/carddb.json: engine card data (exact attack ids, flags) merged
with EN_Card_Data.csv (expansion, collection number, display damage/cost).

    PYTHONPATH=data python3 tools/build_carddb.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data'))
from cg.api import all_card_data, all_attack  # noqa: E402

CARD_TYPE = ['Pokemon', 'Item', 'Tool', 'Supporter', 'Stadium', 'Basic Energy', 'Special Energy']

def na(v):
    v = (v or '').strip()
    return '' if v.lower() in ('n/a', 'nan') else v

def main():
    csv_rows = {}
    with open(ROOT / 'data' / 'EN_Card_Data.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row['Card ID'])
            except ValueError:
                continue
            csv_rows.setdefault(cid, []).append(row)

    attacks = {a.attackId: a for a in all_attack()}
    out_attacks = {
        str(aid): {
            'name': a.name.strip(),
            'text': a.text,
            'damage': a.damage,
            'energies': a.energies,
        } for aid, a in attacks.items()
    }

    cards = {}
    for c in all_card_data():
        rows = csv_rows.get(c.cardId, [])
        first = rows[0] if rows else {}
        # display damage strings ("30×", "160+") and cost glyphs, keyed by move name
        move_rows = {}
        for r in rows:
            mv = na(r.get('Move Name'))
            if mv:
                move_rows[mv.replace('[Ability] ', '').strip()] = {
                    'cost': na(r.get('Cost')),
                    'damage': na(r.get('Damage')),
                    'effect': na(r.get('Effect Explanation')),
                }
        card_effect = ''
        for r in rows:
            if not na(r.get('Move Name')) and na(r.get('Effect Explanation')):
                card_effect = na(r.get('Effect Explanation'))
                break

        atk_list = []
        for aid in c.attacks:
            a = attacks.get(aid)
            if not a:
                continue
            disp = move_rows.get(a.name.strip(), {})
            atk_list.append({
                'id': aid,
                'name': a.name.strip(),
                'text': a.text or disp.get('effect', ''),
                'damage': disp.get('damage') or (str(a.damage) if a.damage else ''),
                'energies': a.energies,
            })

        cards[str(c.cardId)] = {
            'name': c.name,
            'cardType': CARD_TYPE[c.cardType],
            'hp': c.hp or None,
            'energyType': c.energyType,
            'weakness': c.weakness,
            'resistance': c.resistance,
            'retreat': c.retreatCost,
            'basic': c.basic, 'stage1': c.stage1, 'stage2': c.stage2,
            'ex': c.ex, 'megaEx': c.megaEx, 'tera': c.tera, 'aceSpec': c.aceSpec,
            'evolvesFrom': c.evolvesFrom,
            'skills': [{'name': s.name.strip(), 'text': s.text} for s in c.skills],
            'attacks': atk_list,
            'cardEffect': card_effect,
            'expansion': na(first.get('Expansion')),
            'collectionNo': na(first.get('Collection No.')),
            'stageText': na(first.get('Stage (Pokémon)/Type (Energy and Trainer)')),
            'rule': na(first.get('Rule')),
        }

    out = {'cards': cards, 'attacks': out_attacks}
    dest = ROOT / 'data' / 'carddb.json'
    dest.write_text(json.dumps(out, ensure_ascii=False))
    print(f'{dest}: {len(cards)} cards, {len(out_attacks)} attacks, {dest.stat().st_size//1024} KB')

if __name__ == '__main__':
    main()
