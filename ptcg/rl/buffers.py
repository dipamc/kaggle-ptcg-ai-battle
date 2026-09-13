"""Frozen observation/action buffer layout for the PufferLib env.

Single flat float32 vector per observation; ids are stored as floats
(exact for values < 2^24) and cast back in the model. Shapes here are
FROZEN — docs/model.md requires the oracle block to freeze; the rest is
sized with slack so phase-2 encoders fill in without a layout change. Bump VERSION on any change; checkpoints are layout-bound.

Sections (offsets computed below, in order):
  tok_int    MAX_TOKENS x 3    [card_id, zone, owner]
  tok_float  MAX_TOKENS x TOK_F  scalars, indices in TokF
  opt_int    MAX_OPTIONS x 6   [opt_type, ptr1+1, ptr2+1, card_id, attack_id, number]
  opt_float  MAX_OPTIONS x OPT_F  scalars, indices in OptF
  opt_mask   MAX_OPTIONS       1 = selectable this step (class-deduped)
  global_f   GLOBAL_F          g-encoder inputs
  dec_int    DEC_INT           [select_type, context, effect_card_id, context_card_id, ...]
  dec_float  DEC_F             [minCount/5, maxCount/10, n_options/30, remainDmg/10,
                                remainEnergy/5, picked_so_far/5, ...]
  oracle     ORACLE_F          opp hand (24x[id,count/4]) | opp deck/prize split
                               (24x[id,deck/4,prize/4]) | my prizes (8x[id,count/4])

Action space: Discrete(MAX_OPTIONS). Slots 0..62 index select.option
directly; slot 63 (STOP) submits the accumulated multi-select (legal once
picked >= minCount, or for minCount=0 single-selects meaning "pick none").
"""

VERSION = 4   # v4: prize zone, tool ids, lingering-effect flags,
              # temporal counters, last-attack/supporter dec ints

MAX_TOKENS = 160
TOK_INT = 5   # [card_id, zone, owner, tool0_id, tool1_id]
TOK_F = 56

MAX_OPTIONS = 64
STOP = MAX_OPTIONS - 1
OPT_INT = 6
OPT_F = 8

GLOBAL_F = 64
DEC_INT = 8
DEC_F = 16

ORACLE_HAND_SLOTS = 24     # opp hand: distinct ids (id, count/4)
ORACLE_SPLIT_SLOTS = 24    # opp unseen: (id, deck_count/4, prize_count/4)
ORACLE_PRIZE_SLOTS = 8     # my prizes: distinct ids (id, count/4)
ORACLE_F = ORACLE_HAND_SLOTS*2 + ORACLE_SPLIT_SLOTS*3 + ORACLE_PRIZE_SLOTS*2

# Vocab sizes (+16 appended-enum slack per docs/model.md; 0 = NULL)
N_CARDS = 1267 + 17
N_ATTACKS = 1556 + 17
N_ZONES = 16
N_OWNERS = 4
N_OPT_TYPES = 17 + 16      # engine OptionType 0..16 + our own ids below
OT_STOP = 17               # env-synthesized: submit the multi-select / decline
N_SELECT_TYPES = 13 + 16
N_CONTEXTS = 49 + 16

# Zone ids (env-side convention; 0 = padding)
class Zone:
    PAD = 0
    MY_ACTIVE = 1
    MY_BENCH = 2
    OPP_ACTIVE = 3
    OPP_BENCH = 4
    MY_HAND = 5
    STADIUM = 6
    MY_DISCARD = 7
    OPP_DISCARD = 8
    MY_UNSEEN = 9      # tracker-fed (phase 2)
    OPP_HAND_KNOWN = 10  # tracker-fed (phase 2)
    OPP_REVEALED = 11    # tracker-fed (phase 2)
    LOOKING = 12       # select.deck / state.looking pools
    MY_PRIZE_KNOWN = 13  # exact prize multiset once determined (tracker);
                         # MY_UNSEEN is then the deck-only remainder

class Owner:
    NEUTRAL = 0
    MINE = 1
    OPP = 2

# Token float indices
class TokF:
    COPIES = 0        # /4 (count-collapsed zones)
    HP_FRAC = 1       # hp/maxHp
    HP = 2            # hp/300
    MAX_HP = 3        # maxHp/300
    DMG = 4           # damage counters /30
    N_ENERGY = 5      # /5
    ENERGY0 = 6       # 6..17: attached count per EnergyType (12) /3
    N_TOOLS = 18      # /2
    STACK = 19        # evolution stack depth /3
    APPEAR = 20       # entered play this turn
    POISONED = 21
    BURNED = 22
    ASLEEP = 23
    PARALYZED = 24
    CONFUSED = 25
    IS_ACTIVE = 26
    REFERENCED = 27   # pointed at by some option this step
    PICKED = 28       # already picked this multi-select
    # derived attack features (board Pokemon; top-4 attacks by damage):
    ATK0 = 29         # 29..40: per attack [affordable-now, damage/300, cost-deficit/3]
    WEAK_HIT = 41     # opp active is weak to this Pokemon's type
    RES_HIT = 42      # opp active resists this Pokemon's type
    RETREAT_OK = 43   # attached energy covers retreat cost
    # lingering-effect flags (ptcg/effects.py registers; 1.0 = binding
    # this turn, 0.5 = registered but binds on a coming turn)
    EFF_CANT_ATTACK = 44
    EFF_CANT_RETREAT = 45
    EFF_PREVENT_ALL = 46     # prevent all damage (and/or effects)
    EFF_DMG_TAKEN_MOD = 47   # signed, +takes-more/-takes-less, /100
    EFF_DMG_DEALT_MOD = 48   # signed, its attacks do +/-, /100
    EFF_DELAYED_DMG = 49     # damage counters incoming /10
    EFF_DELAYED_KO = 50      # will be Knocked Out at trigger
    EFF_ATTACK_LOCKED = 51   # one specific attack locked
    EVOLVED_THIS_TURN = 52   # log-derived (conditional-attack fuel)
    HEALED_THIS_TURN = 53
    # 54..55 reserved

# Option float indices
class OptF:
    NUMBER = 0        # option.number / 30
    COUNT = 1         # option.count / 4
    PICKED = 2        # already picked (index-level)
    IS_STOP = 3
    SPECIAL = 4       # specialConditionType / 5
    ENERGY_IDX = 5    # energyIndex / 5
    # 6..7 reserved

# Global float indices (subset used phase-1; slack reserved)
class GlobF:
    TURN = 0            # /50
    PARITY = 1          # turn % 2
    I_AM_FIRST = 2
    FIRST_DECIDED = 3
    ACTIONS = 4         # turnActionCount /20
    SUPPORTER = 5
    STADIUM_PLAYED = 6
    ENERGY_ATTACHED = 7
    RETREATED = 8
    MY_PRIZES = 9       # remaining /6
    OPP_PRIZES = 10
    MY_DECK = 11        # /60
    OPP_DECK = 12
    MY_HAND = 13        # /20
    OPP_HAND = 14
    MY_BENCH = 15       # occupancy /5
    OPP_BENCH = 16
    BENCH_MAX = 17      # /8
    STADIUM_PRESENT = 18
    MY_STATUS0 = 19     # 19..23 my active status (5)
    OPP_STATUS0 = 24    # 24..28
    PICKS_MIN_LEFT = 29  # (minCount - picked)+ /5
    PICKS_MAX_LEFT = 30  # (maxCount - picked)+ /10
    ACTOR_SEAT = 31      # which physical seat is acting (0/1) — read by the
                         # negamax advantage to detect actor switches
    OPP_HAND_UNKNOWN = 32  # opp hand cards with unknown identity /20 (tracker)
    MY_PRIZES_KNOWN = 33   # 1 if my exact prize multiset is known (tracker)
    # lingering player-level locks (1.0 binding now, 0.5 pending)
    MY_LOCK_ITEM = 34       # I can't play Item cards
    MY_LOCK_SUPPORTER = 35
    MY_LOCK_EVOLVE = 36
    OPP_LOCK_ITEM = 37      # locks I imposed on the opponent
    OPP_LOCK_SUPPORTER = 38
    OPP_LOCK_EVOLVE = 39
    MY_DMG_BOOST = 40       # player-level attack-damage boost /100
    OPP_DMG_BOOST = 41
    # temporal opponent-behavior counters (cumulative, tracker-fed)
    MY_ENERGY_REMOVED = 42  # my attached energy removed on opp turns /10
    OPP_ENERGY_REMOVED = 43
    MY_TOOLS_REMOVED = 44   # /5
    OPP_TOOLS_REMOVED = 45
    MY_FORCED_SWITCHES = 46  # my active swapped during opp turns /5
    OPP_FORCED_SWITCHES = 47
    MY_KOD_LAST_TURN = 48    # I lost a Pokemon on opp's last turn
    OPP_KOD_LAST_TURN = 49   # ("Avenging"-style attack conditionals)
    # static one-ply lethal estimate (base damage + weak/res + known
    # effect mods; heuristic — variable attacks not modeled)
    MY_ACTIVE_IN_DANGER = 50     # opp active can KO my active now
    MY_ACTIVE_IN_DANGER_P1 = 51  # ...with one more energy
    OPP_ACTIVE_IN_DANGER = 52    # my active can KO theirs now
    OPP_ACTIVE_IN_DANGER_P1 = 53
    FROZEN_ACTOR = 54        # 0 = learner row; k>0 = acted by frozen
                             # opponent (k-1). Was a 0/1 flag; storing
                             # idx+1 keeps N=1 byte-identical. Routes
                             # rollout to frozen weights and masks the
                             # row out of the policy loss. Width is
                             # unchanged and the slot was always 0, so
                             # NO VERSION bump -- old ckpts stay valid.
    # 55..63 reserved

class DecInt:
    SELECT_TYPE = 0
    CONTEXT = 1
    EFFECT_CARD = 2
    CONTEXT_CARD = 3
    MY_LAST_ATTACK = 4    # attackId my current active used most recently
    OPP_LAST_ATTACK = 5   # (0 if that Pokemon is no longer the active)
    MY_SUPPORTER = 6      # card id of supporter I played this turn
    OPP_SUPPORTER = 7     # supporter opp played on their last turn

class DecF:
    MIN_COUNT = 0     # /5
    MAX_COUNT = 1     # /10
    N_OPTIONS = 2     # /30
    REMAIN_DMG = 3    # /10
    REMAIN_ENERGY = 4  # /5
    PICKED = 5        # picked so far /5
    FORCED_RUN = 6    # count of auto-resolved selects since last decision /5
    # 7..15 reserved


def _build_offsets():
    off, cur = {}, 0
    for name, size in [
        ("tok_int", MAX_TOKENS * TOK_INT),
        ("tok_float", MAX_TOKENS * TOK_F),
        ("opt_int", MAX_OPTIONS * OPT_INT),
        ("opt_float", MAX_OPTIONS * OPT_F),
        ("opt_mask", MAX_OPTIONS),
        ("global_f", GLOBAL_F),
        ("dec_int", DEC_INT),
        ("dec_float", DEC_F),
        ("oracle", ORACLE_F),
    ]:
        off[name] = (cur, cur + size)
        cur += size
    return off, cur


OFFSETS, OBS_SIZE = _build_offsets()

# absolute obs column of the actor-seat feature (negamax advantage reads it)
SEAT_COL = OFFSETS["global_f"][0] + GlobF.ACTOR_SEAT

# absolute obs column of the frozen-opponent flag (driver routing +
# policy-loss mask read it; both are zeroed before any forward)
FROZEN_COL = OFFSETS["global_f"][0] + GlobF.FROZEN_ACTOR

# sentinel card id for "identity unknown" oracle rows (top of the +16 slack)
UNK_CARD = N_CARDS - 1
