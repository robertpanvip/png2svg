from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from model.targets import NUM_SLOTS, SLOT_DIM, BG_DIM, NUM_CLS, I_BBOX


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(groups, cout)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 16):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.conv1(self.act(self.norm1(x)))
        y = self.conv2(self.act(self.norm2(y)))
        return x + y


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 4, d_model: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_ch, 32, 2, 8),
            ConvBlock(32, 48, 2, 8),
            ConvBlock(48, 96, 2, 16),
            ConvBlock(96, 192, 2, 16),
            ConvBlock(192, d_model, 1, 16),
        )
        self.res1 = ResBlock(d_model)
        self.res2 = ResBlock(d_model)

    def forward(self, x):
        return self.res2(self.res1(self.stem(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, q, kv):
        x = self.n1(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        x = self.n2(q)
        q = q + self.cross_attn(x, kv, kv, need_weights=False)[0]
        q = q + self.ffn(self.n3(q))
        return q


class VectorNet(nn.Module):
    def __init__(self, in_ch: int = 4, d_model: int = 256, n_layers: int = 2,
                 n_heads: int = 4, ffn_dim: int = 512,
                 num_slots: int = NUM_SLOTS, img_size: int = 256):
        super().__init__()
        self.num_slots = num_slots
        self.encoder = Encoder(in_ch, d_model)
        self.grid = img_size // 16
        self.pos_emb = nn.Parameter(torch.zeros(1, d_model, self.grid, self.grid))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.queries = nn.Parameter(torch.zeros(1, num_slots, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.slot_head = nn.Linear(d_model, SLOT_DIM)
        self.cls_head = nn.Linear(d_model, NUM_CLS)
        self.ftype_head = nn.Linear(d_model, 4)
        self.bg_head = nn.Linear(d_model, BG_DIM)
        # 空间锚点先验：强制每个 slot 偏向画布不同区域，打破 bbox 中心坍缩。
        # 加到 bbox 的 cx/cy 原始 logit 上（squash 用 _lin 把 logit 映射到 [-0.2,1.2]）。
        self.spatial_anchor = nn.Parameter(torch.zeros(num_slots, 2))
        C_MIN, C_MAX = -0.20, 1.20
        cols, rows = 4, 2
        with torch.no_grad():
            for k in range(num_slots):
                r, c = divmod(k, cols)
                gx = (c + 0.5) / cols          # 期望中心（canvas 坐标 0..1）
                gy = (r + 0.5) / rows
                ax = math.log((gx - C_MIN) / (C_MAX - gx))   # logit 逆变换
                ay = math.log((gy - C_MIN) / (C_MAX - gy))
                self.spatial_anchor[k, 0] = ax
                self.spatial_anchor[k, 1] = ay
        n_params = sum(p.numel() for p in self.parameters())
        assert 3_000_000 <= n_params <= 8_000_000, f"param budget violated: {n_params}"

    def forward(self, img):
        feats = self.encoder(img)
        b = feats.shape[0]
        pos = self.pos_emb
        if pos.shape[-2:] != feats.shape[-2:]:
            pos = F.interpolate(pos, size=feats.shape[-2:], mode="bilinear",
                                align_corners=False)
        tokens = (feats + pos).flatten(2).transpose(1, 2)
        q = self.queries.expand(b, -1, -1)
        for layer in self.layers:
            q = layer(q, tokens)
        h = self.final_norm(q)
        slots = self.slot_head(h)
        slots[..., I_BBOX:I_BBOX + 2] = slots[..., I_BBOX:I_BBOX + 2] + self.spatial_anchor
        aux = {"cls": self.cls_head(h), "ftype": self.ftype_head(h)}
        bg = self.bg_head(tokens.mean(dim=1))
        return slots, aux, bg


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
