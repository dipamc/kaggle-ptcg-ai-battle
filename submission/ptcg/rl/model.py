"""Phase-2 policy: token transformer per docs/model.md.

Implemented: compositional static encoders (AttackVec -> CardVec, §1),
token projection + zone/owner embeddings (§3), Global/Decision encoders
broadcast-added to every token input (§3), pre-LN trunk with READOUT CLS
(§4), pointer-style option scorer over trunk outputs (§5), blind value
head (§6).

Oracle critic (§6b): a training-only top mounted ABOVE the trunk —
joint attention over [trunk outputs ‖ oracle tokens ‖ critic readout].
Oracle-V is the PPO baseline (GAE runs on it); Blind-V (§6) trains on
the same returns and is the head search will use at inference. The
policy head has no forward path from oracle inputs (leak safety is
structural). Belief aux heads (§6): opp-hand count prediction from
[r‖g], per-my-unseen-token prize probability — targets come from the
oracle block in the same obs.

Deliberately-later items: SkillVec schema-TSV/text channels +
text-initialized tables, mechanical-outcome aux heads (need the
transition corpus), prize-margin/archetype aux heads. Value heads are
linear, not tanh: PPO regresses +-12-scale returns (tanh is BC spec).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffers import (
    OFFSETS, MAX_TOKENS, TOK_INT, TOK_F, MAX_OPTIONS, OPT_INT, OPT_F,
    GLOBAL_F, DEC_F, N_CARDS, N_ATTACKS, N_ZONES, N_OWNERS,
    N_OPT_TYPES, N_SELECT_TYPES, N_CONTEXTS, DecInt, Zone,
    ORACLE_HAND_SLOTS, ORACLE_SPLIT_SLOTS, ORACLE_PRIZE_SLOTS,
)
from .cards import build_tables, load_text_tables, N_SKILLS

N_ORACLE_TOKENS = ORACLE_HAND_SLOTS + ORACLE_SPLIT_SLOTS + ORACLE_PRIZE_SLOTS


def _take(obs, name):
    a, b = OFFSETS[name]
    return obs[:, a:b]


def arch_from_state_dict(sd, heads=8):
    """Recover PTCGTransformer constructor kwargs from a checkpoint.

    Anything that loads a checkpoint has to build the module first, and
    hardcoding the defaults there breaks silently the moment a run uses
    --d/--layers/--ffn: d/layers/ffn mismatches raise on load_state_dict,
    but a wrong `heads` does NOT -- the attention projections are d x d
    whatever the head count is, so it loads cleanly and then computes
    different attention. So d/layers/ffn are read off the weights here and
    `heads` must be supplied by whoever knows how the run was launched.
    """
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    try:
        d = sd["blocks.0.ln1.weight"].shape[0]
        ffn = sd["blocks.0.ffn.0.weight"].shape[0]
        layers = 1 + max(int(k.split(".")[1]) for k in sd
                         if k.startswith("blocks."))
    except (KeyError, ValueError) as e:
        raise ValueError(f"not a PTCGTransformer state dict: {e!r}") from e
    if d % heads:
        raise ValueError(
            f"checkpoint width d={d} is not divisible by heads={heads} — "
            f"pass the run's actual head count")
    return dict(d=d, layers=layers, heads=heads, ffn=ffn)


class Block(nn.Module):
    """Pre-LN block; manual QKV + SDPA (nn.MultiheadAttention falls back
    to the weight-materializing path under a key_padding_mask — OOMs at
    training minibatch sizes)."""

    def __init__(self, d, heads, ffn):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, keep):
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, L, 3, self.heads, D // self.heads) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.ffn(self.ln2(x))


class PTCGTransformer(nn.Module):
    def __init__(self, env=None, d=256, layers=4, heads=8, ffn=512,
                 d_c=128, d_a=64, critic_layers=1, critic="oracle"):
        # critic="blind": the shared blind value head (sees exactly what
        # the policy sees) is the PPO critic; the oracle top trains as a
        # passive auxiliary. critic="oracle": §6b CTDE (oracle-V drives
        # GAE). Rollout values follow this choice via forward_eval.
        super().__init__()
        self.d, self.d_c, self.d_a = d, d_c, d_a
        self.critic_mode = critic
        card_static, card_attacks, card_skills, att_static = build_tables()
        self.register_buffer("card_static", torch.from_numpy(card_static))
        self.register_buffer("card_attacks", torch.from_numpy(card_attacks))
        self.register_buffer("card_skills", torch.from_numpy(card_skills))
        self.register_buffer("att_static", torch.from_numpy(att_static))
        card_text, att_text, skill_text = load_text_tables()
        self.register_buffer("card_text", torch.from_numpy(card_text))
        self.register_buffer("att_text", torch.from_numpy(att_text))
        self.register_buffer("skill_text", torch.from_numpy(skill_text))

        # §1 static encoders (learned id tables ‖ frozen text channels)
        self.card_id_emb = nn.Embedding(N_CARDS, 64)
        self.attack_id_emb = nn.Embedding(N_ATTACKS, 48)
        self.skill_id_emb = nn.Embedding(N_SKILLS, 48)
        self.attack_mlp = nn.Sequential(
            nn.Linear(48 + self.att_static.shape[1]
                      + self.att_text.shape[1], d_a), nn.GELU())
        self.skill_mlp = nn.Sequential(
            nn.Linear(48 + self.skill_text.shape[1], d_c), nn.GELU())
        card_in = 64 + self.card_static.shape[1] + self.card_text.shape[1] \
            + 2 * d_a + 2 * d_c
        self.card_mlp = nn.Sequential(
            nn.LayerNorm(card_in), nn.Linear(card_in, 256), nn.GELU(),
            nn.Linear(256, d_c))

        # §3 tokens + context encoders (v4: + attached-tool CardVec sum)
        self.zone_emb = nn.Embedding(N_ZONES, 8)
        self.owner_emb = nn.Embedding(N_OWNERS, 4)
        self.token_proj = nn.Linear(2 * d_c + TOK_F + 8 + 4, d)
        self.readout = nn.Parameter(torch.randn(d) * 0.02)
        self.global_mlp = nn.Sequential(
            nn.Linear(GLOBAL_F, 128), nn.GELU(), nn.Linear(128, 128))
        self.sel_type_emb = nn.Embedding(N_SELECT_TYPES, 8)
        self.ctx_emb = nn.Embedding(N_CONTEXTS, 32)
        # v4: + last-attack AttackVecs + this/last-turn supporter CardVecs
        dec_in = 8 + 32 + 2 * d_c + 2 * d_a + 2 * d_c + DEC_F
        self.dec_mlp = nn.Sequential(
            nn.Linear(dec_in, 64), nn.GELU(), nn.Linear(64, 64))
        self.bcast = nn.Linear(128 + 64, d)

        # §4 trunk
        self.blocks = nn.ModuleList(
            Block(d, heads, ffn) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)

        # §5 option scorer
        self.opt_type_emb = nn.Embedding(N_OPT_TYPES, 16)
        self.opt_card = nn.Linear(d_c, 64)
        self.opt_attack = nn.Linear(d_a, 64)
        q_in = 16 + 2 * d + 64 + 64 + OPT_F
        self.q_mlp = nn.Sequential(
            nn.Linear(q_in, d), nn.GELU(), nn.Linear(d, d))
        self.score_mlp = nn.Sequential(
            nn.Linear(3 * d + 128 + 64, 256), nn.GELU(), nn.Linear(256, 1))
        nn.init.orthogonal_(self.score_mlp[-1].weight, gain=0.01)

        # §6 blind value head (ships; search leaf evaluation)
        self.value_mlp = nn.Sequential(
            nn.Linear(d + 128, 256), nn.GELU(), nn.Linear(256, 1))

        # §6b oracle critic top (training only; GAE runs on oracle-V)
        # oracle token feats: [is_hand, is_split, is_prize, count, deck_n, prize_n]
        self.oracle_proj = nn.Linear(d_c + 6, d)
        self.critic_readout = nn.Parameter(torch.randn(d) * 0.02)
        self.critic_blocks = nn.ModuleList(
            Block(d, heads, ffn) for _ in range(critic_layers))
        self.oracle_value_mlp = nn.Sequential(
            nn.Linear(d + 128, 256), nn.GELU(), nn.Linear(256, 1))

        # §6 belief aux heads (training only)
        self.aux_hand = nn.Linear(d + 128, N_CARDS)   # opp hand count pred
        self.aux_prize = nn.Linear(d, 1)              # per-unseen-token P(prized)

    def freeze_tables(self):
        """Precompute the static CardVec/AttackVec tables once — for
        frozen policies (league opponents, eval) whose weights never
        change, rebuilding them per forward is pure waste (dominates
        batch-1 CPU inference)."""
        with torch.no_grad():
            self._frozen_tables = self._card_table()

    def _card_table(self):
        """CardVec for every card id, rebuilt each forward (§1)."""
        ft = getattr(self, "_frozen_tables", None)
        if ft is not None:
            return ft
        att = self.attack_mlp(torch.cat(
            [self.attack_id_emb.weight, self.att_static, self.att_text],
            dim=-1))
        a = att[self.card_attacks]                       # (C, 4, d_a)
        a_mask = (self.card_attacks > 0).unsqueeze(-1)
        a_mean = (a * a_mask).sum(1) / a_mask.sum(1).clamp(min=1)
        a_max = (a * a_mask).amax(1)
        sk = self.skill_mlp(torch.cat(
            [self.skill_id_emb.weight, self.skill_text],
            dim=-1))[self.card_skills]
        s_mask = (self.card_skills > 0).unsqueeze(-1)
        s_mean = (sk * s_mask).sum(1) / s_mask.sum(1).clamp(min=1)
        s_max = (sk * s_mask).amax(1)
        return self.card_mlp(torch.cat(
            [self.card_id_emb.weight, self.card_static, self.card_text,
             a_mean, a_max, s_mean, s_max], dim=-1)), att

    def _core(self, obs):
        B = obs.shape[0]
        card_vec, attack_vec = self._card_table()

        tok_i = _take(obs, "tok_int").view(B, MAX_TOKENS, TOK_INT).long()
        tok_f = _take(obs, "tok_float").view(B, MAX_TOKENS, TOK_F)
        cid = tok_i[:, :, 0].clamp(0, N_CARDS - 1)
        t0, t1 = tok_i[:, :, 3], tok_i[:, :, 4]
        tool_vec = \
            card_vec[t0.clamp(0, N_CARDS - 1)] * (t0 > 0).unsqueeze(-1) + \
            card_vec[t1.clamp(0, N_CARDS - 1)] * (t1 > 0).unsqueeze(-1)
        x = self.token_proj(torch.cat([
            card_vec[cid], tool_vec, tok_f,
            self.zone_emb(tok_i[:, :, 1].clamp(0, N_ZONES - 1)),
            self.owner_emb(tok_i[:, :, 2].clamp(0, N_OWNERS - 1))], dim=-1))

        g = self.global_mlp(_take(obs, "global_f"))
        dec_i = _take(obs, "dec_int").long()
        d_vec = self.dec_mlp(torch.cat([
            self.sel_type_emb(dec_i[:, DecInt.SELECT_TYPE].clamp(0, N_SELECT_TYPES - 1)),
            self.ctx_emb(dec_i[:, DecInt.CONTEXT].clamp(0, N_CONTEXTS - 1)),
            card_vec[dec_i[:, DecInt.EFFECT_CARD].clamp(0, N_CARDS - 1)],
            card_vec[dec_i[:, DecInt.CONTEXT_CARD].clamp(0, N_CARDS - 1)],
            attack_vec[dec_i[:, DecInt.MY_LAST_ATTACK].clamp(0, N_ATTACKS - 1)],
            attack_vec[dec_i[:, DecInt.OPP_LAST_ATTACK].clamp(0, N_ATTACKS - 1)],
            card_vec[dec_i[:, DecInt.MY_SUPPORTER].clamp(0, N_CARDS - 1)],
            card_vec[dec_i[:, DecInt.OPP_SUPPORTER].clamp(0, N_CARDS - 1)],
            _take(obs, "dec_float")], dim=-1))
        x = x + self.bcast(torch.cat([g, d_vec], dim=-1)).unsqueeze(1)

        # tokens are filled as a contiguous prefix — slice the batch to its
        # longest real prefix (typical ~40-90 of 160; halves trunk cost)
        valid = tok_i[:, :, 1] > 0
        L = int(valid.sum(1).max().item())
        x = torch.cat([self.readout.expand(B, 1, -1), x[:, :L]], dim=1)
        keep = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=obs.device),
            valid[:, :L]], dim=1).view(B, 1, 1, -1)
        for blk in self.blocks:
            x = blk(x, keep)
        h = self.ln_f(x)
        r = h[:, 0]

        opt_i = _take(obs, "opt_int").view(B, MAX_OPTIONS, OPT_INT).long()
        opt_f = _take(obs, "opt_float").view(B, MAX_OPTIONS, OPT_F)
        # ptr indices are buffer-token+1 with 0=null; h[:, 0] is READOUT so
        # index k in the buffer lives at h[:, k+1] — the +1 offsets align.
        # (pad back to MAX_TOKENS+1 for gather safety after prefix slicing;
        # real ptrs only ever target valid tokens < L.)
        hz = torch.zeros(B, MAX_TOKENS + 1, self.d,
                         dtype=h.dtype, device=h.device)
        hz[:, 1:L + 1] = h[:, 1:]
        idx1 = opt_i[:, :, 1].clamp(0, MAX_TOKENS)
        idx2 = opt_i[:, :, 2].clamp(0, MAX_TOKENS)
        p1 = hz.gather(1, idx1.unsqueeze(-1).expand(-1, -1, self.d))
        p2 = hz.gather(1, idx2.unsqueeze(-1).expand(-1, -1, self.d))
        q = self.q_mlp(torch.cat([
            self.opt_type_emb(opt_i[:, :, 0].clamp(0, N_OPT_TYPES - 1)),
            p1, p2,
            self.opt_card(card_vec[opt_i[:, :, 3].clamp(0, N_CARDS - 1)]),
            self.opt_attack(attack_vec[opt_i[:, :, 4].clamp(0, N_ATTACKS - 1)]),
            opt_f], dim=-1))
        rr = r.unsqueeze(1).expand(-1, MAX_OPTIONS, -1)
        gg = g.unsqueeze(1).expand(-1, MAX_OPTIONS, -1)
        dd = d_vec.unsqueeze(1).expand(-1, MAX_OPTIONS, -1)
        scores = self.score_mlp(torch.cat(
            [q, rr, gg, dd, q * rr], dim=-1)).squeeze(-1)
        logits = scores.masked_fill(_take(obs, "opt_mask") == 0, -1e9)
        return dict(logits=logits, r=r, g=g, h=h, L=L,
                    valid=valid, card_vec=card_vec, tok_i=tok_i)

    def _oracle_value(self, obs, c):
        """§6b critic top: joint attention over [critic-readout ‖ trunk
        outputs ‖ oracle tokens]. Training-only; no path to the policy."""
        B = obs.shape[0]
        o = _take(obs, "oracle")
        H, S, P = ORACLE_HAND_SLOTS, ORACLE_SPLIT_SLOTS, ORACLE_PRIZE_SLOTS
        hand = o[:, :H * 2].view(B, H, 2)
        split = o[:, H * 2:H * 2 + S * 3].view(B, S, 3)
        prize = o[:, H * 2 + S * 3:].view(B, P, 2)
        ids = torch.cat([hand[:, :, 0], split[:, :, 0], prize[:, :, 0]],
                        dim=1).long().clamp(0, N_CARDS - 1)
        feats = torch.zeros(B, N_ORACLE_TOKENS, 6,
                            dtype=o.dtype, device=o.device)
        feats[:, :H, 0] = 1.0
        feats[:, :H, 3] = hand[:, :, 1]
        feats[:, H:H + S, 1] = 1.0
        feats[:, H:H + S, 4] = split[:, :, 1]
        feats[:, H:H + S, 5] = split[:, :, 2]
        feats[:, H + S:, 2] = 1.0
        feats[:, H + S:, 3] = prize[:, :, 1]
        otok = self.oracle_proj(
            torch.cat([c["card_vec"][ids], feats], dim=-1))

        x = torch.cat([self.critic_readout.expand(B, 1, -1),
                       c["h"], otok], dim=1)
        one = torch.ones(B, 1, dtype=torch.bool, device=obs.device)
        keep = torch.cat([one, one, c["valid"][:, :c["L"]], ids > 0],
                         dim=1).view(B, 1, 1, -1)
        for blk in self.critic_blocks:
            x = blk(x, keep)
        return self.oracle_value_mlp(torch.cat([x[:, 0], c["g"]], dim=-1))

    def forward_policy(self, obs):
        """Logits only — skips the critic top (league opponents/eval
        sampling don't need values)."""
        return self._core(obs)["logits"]

    def forward_eval(self, obs, state=None):
        c = self._core(obs)
        if self.critic_mode == "blind":
            v = self.value_mlp(torch.cat([c["r"], c["g"]], dim=-1))
        else:
            v = self._oracle_value(obs, c)
        return c["logits"], v

    def forward(self, obs, state=None):
        return self.forward_eval(obs, state)

    def forward_train(self, obs):
        """Everything the vendored trainer needs: policy logits, both
        value heads, and belief-aux predictions (targets are derived
        from the oracle block of the same obs, trainer-side)."""
        c = self._core(obs)
        rg = torch.cat([c["r"], c["g"]], dim=-1)
        tok_h = c["h"][:, 1:]                       # (B, L, D)
        return dict(
            logits=c["logits"],
            oracle_v=self._oracle_value(obs, c),
            blind_v=self.value_mlp(rg),
            hand_pred=F.softplus(self.aux_hand(rg)),
            prize_logits=self.aux_prize(tok_h).squeeze(-1),   # (B, L)
            unseen_mask=(c["tok_i"][:, :c["L"], 1] == Zone.MY_UNSEEN),
            token_ids=c["tok_i"][:, :c["L"], 0],
        )
