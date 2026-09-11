from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from model.targets import (
    NUM_SLOTS, SLOT_DIM, BG_DIM, I_BBOX, I_SEG, I_FTYPE, I_FRGB, I_SVALID,
    I_OPACITY, I_GROUP, I_END, N_SEG, SEG_DIM,
)
from model.spec import C_MIN, C_MAX, W_MIN, W_MAX


def _lin(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


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


# 分块头宽度（由 §11.6 契约索引推导，勿手写魔数）
_GEOM_W = I_SEG                  # 7  (valid, bbox4, nseg, closed)
_SEG_W = N_SEG * SEG_DIM         # 192
_FILL_W = I_SVALID - I_FTYPE     # 34
_STROKE_W = I_OPACITY - I_SVALID  # 17
_FX_W = I_GROUP - I_OPACITY      # 14
_COMP_W = I_END - I_GROUP        # 11
assert _GEOM_W + _SEG_W + _FILL_W + _STROKE_W + _FX_W + _COMP_W == I_END


def _sine_pos_at(d: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """连续坐标 (x,y)∈[0,1]² 的 2D 正弦位置编码 [B,K,d]。

    与 _sine_pos_2d 使用同一频率基（token 网格用其格点中心坐标），
    使 query 位置编码与 token 位置编码可直接做内积 → attention 无需
    学习即可按 bbox 中心聚焦（HANDOFF §11.11：在线分布下 attention
    学不会"往哪看"，改为确定性空间提示）。
    """
    # x,y: [B,K] in [0,1]
    d_half = d // 2
    div = torch.exp(torch.arange(0, d_half, 2, device=x.device, dtype=x.dtype)
                    * (-math.log(10000.0) / d_half))          # [d_half/2]
    ang_x = x.unsqueeze(-1) * div                          # [B,K,d_half/2]
    ang_y = y.unsqueeze(-1) * div
    pe = torch.zeros(*x.shape, d, device=x.device, dtype=x.dtype)
    pe[..., 0:d_half:2] = ang_x.sin()
    pe[..., 1:d_half:2] = ang_x.cos()
    pe[..., d_half + 0::2] = ang_y.sin()
    pe[..., d_half + 1::2] = ang_y.cos()
    return pe


def _sine_pos_2d(d: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """token 网格位置编码 [1, h*w, d]，格点取格心坐标，与 _sine_pos_at 同基。"""
    assert d % 4 == 0
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")         # [h,w]
    yy = yy.reshape(1, -1).expand(1, h * w).squeeze(0)     # 行优先 flatten
    xx = xx.reshape(1, -1).expand(1, h * w).squeeze(0)
    return _sine_pos_at(d, xx.unsqueeze(0), yy.unsqueeze(0)).reshape(1, h * w, d)


class VectorNet(nn.Module):
    """VectorNet + 高分辨率颜色分支（HANDOFF §11.11 P-Appearance）。

    主干 encoder 16×16 token 会把细小对象的颜色平均掉（实测 bbox/路径
    采样与 GT fill 颜色相关性仅 0.1-0.27），slot 表征 h 因而不含可泛化
    的颜色信息——跨场景 fill 全员收敛到数据集均值灰。此处加一条
    32×32 token（8px 粒度）的高分辨率分支，slot 经 cross-attention
    直接读取对象内部像素颜色；仅 fill/stroke 头消费该特征。
    """

    HR_DIM = 64   # 高分辨率分支 token 维度

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
        # ---- 高分辨率颜色分支（stride 4 → 32×32 token）----
        # 关键：GroupNorm 会抹掉绝对颜色尺度（实测跨场景 test R²<0），
        # 故 attention 的 key/value 拼入【未归一化】的 8px 原始颜色 token，
        # 并行输出确定性 cf_raw（bbox 中心高斯加权原始颜色均值）——
        # 颜色信息"线性在特征里"，跨场景泛化由构造保证。
        self.hr_encoder = nn.Sequential(
            ConvBlock(in_ch, 32, 2, 8),
            ConvBlock(32, 64, 2, 8),
            ConvBlock(64, self.HR_DIM - 4, 2, 4),
        )
        self.hr_grid = img_size // 8
        self.hr_proj_q = nn.Linear(d_model, self.HR_DIM)
        self.hr_n1 = nn.LayerNorm(self.HR_DIM)
        self.hr_n2 = nn.LayerNorm(self.HR_DIM)
        self.hr_kv = self.HR_DIM                               # ctx 60 + raw rgba 4 = 64
        self.hr_attn = nn.MultiheadAttention(self.HR_DIM, 4, batch_first=True)
        self.hr_out = nn.Linear(self.hr_kv, self.HR_DIM)
        self.hr_ffn = nn.Sequential(
            nn.Linear(self.HR_DIM, 128), nn.SiLU(), nn.Linear(128, self.HR_DIM),
        )
        # ---- E 方案：填充掩码解码头（HANDOFF §11.15：先分割再取色）----
        # Mask2Former-lite：slot mask query 经 1 层 cross-attn 精化后与
        # pixel embedding 做点积 → [B,K,32,32] logits。
        # v1（纯点积）在过拟合探针 400-600 步 IoU 平台于 0.41（门 0.60）——
        # 单 query 向量表达不了凹形/复合形状，加查询精化 + 独立 pix embed。
        self.mask_tok = nn.Sequential(
            nn.Conv2d(self.HR_DIM - 4, self.HR_DIM, 1), nn.SiLU(),
            nn.Conv2d(self.HR_DIM, self.HR_DIM, 3, padding=1), nn.SiLU(),
        )
        self.mask_qproj = nn.Linear(d_model, self.HR_DIM)
        self.mask_attn = nn.MultiheadAttention(self.HR_DIM, 4, batch_first=True)
        self.mask_n1 = nn.LayerNorm(self.HR_DIM)
        self.mask_n2 = nn.LayerNorm(self.HR_DIM)
        self.mask_ffn = nn.Sequential(
            nn.Linear(self.HR_DIM, 128), nn.SiLU(), nn.Linear(128, self.HR_DIM),
        )
        self.mask_pix = nn.Sequential(
            nn.Conv2d(self.HR_DIM, self.HR_DIM, 1), nn.SiLU(),
            nn.Conv2d(self.HR_DIM, self.HR_DIM, 1),
        )
        self.mask_bias = nn.Parameter(torch.zeros(num_slots))
        # 分块输出头（§11.6 契约）：段表用宽 MLP，其余块轻量。
        # fill/stroke 头消费 color_in = [h, hrf, cf_raw, cf_mask]
        # （+8 = 两种原始 RGBA 池化各 4 维）。
        self.geom_head = nn.Linear(d_model, _GEOM_W)
        self.seg_head = nn.Sequential(
            nn.Linear(d_model, 512), nn.SiLU(), nn.Linear(512, _SEG_W),
        )
        self.fill_head = nn.Sequential(
            nn.Linear(d_model + self.HR_DIM + 8, 256), nn.SiLU(), nn.Linear(256, _FILL_W),
        )
        # §11.13 色板分类头（N_PAL=48 类 logits）。
        # 输入必须是 color_in（h 本身无颜色信息，§11.11 定案）。
        from model.palette import N_PALETTE, PALETTE_RGB
        self.palette_head = nn.Linear(d_model + self.HR_DIM + 8, N_PALETTE)
        self.register_buffer("palette_rgb",
                             torch.from_numpy(PALETTE_RGB))          # [48,3]
        self.I_FRGB_OFF = I_FRGB - I_FTYPE                            # 槽内偏移
        self.stroke_head = nn.Sequential(
            nn.Linear(d_model + self.HR_DIM + 8, 128), nn.SiLU(), nn.Linear(128, _STROKE_W),
        )
        self.fx_head = nn.Linear(d_model, _FX_W)
        self.comp_head = nn.Linear(d_model, _COMP_W)
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

    def _hr_feat(self, img: torch.Tensor, h: torch.Tensor,
                 cx: torch.Tensor, cy: torch.Tensor,
                 w: torch.Tensor, hgt: torch.Tensor):
        """高分辨率颜色特征：确定性原始颜色池化 + slot cross-attention。

        返回 (hrf [B,K,64], cf_raw [B,K,4], ctx [B,60,gh,gw], rawt [B,T,4])：
        - cf_raw：8px 原始 RGBA token 按 bbox 中心高斯加权均值（零学习，
          泛化由构造保证，corr≈0.42 起步）；
        - hrf：query=proj(h)+bbox 中心位置编码，对 [ctx‖raw] token 做
          attention（学到后可逼近 oracle 内部采样 0.65）；
        - ctx/rawt：供掩码头（E 方案）与掩码内池化复用，避免二次前向。
        """
        b = img.shape[0]
        hr = self.hr_encoder(img)                              # [B,60,gh,gw]
        gh, gw = hr.shape[-2:]
        raw = F.avg_pool2d(img, img.shape[-1] // gw)           # [B,4,gh,gw] 原始颜色
        ctx = hr.flatten(2).transpose(1, 2)                    # [B,gh*gw,60]
        rawt = raw.flatten(2).transpose(1, 2)                  # [B,gh*gw,4]
        tok = torch.cat([ctx, rawt], dim=-1)                   # [B,T,64]
        pe = _sine_pos_2d(self.HR_DIM, gh, gw, tok.device, tok.dtype)
        tokp = tok + pe
        q = self.hr_proj_q(h) + _sine_pos_at(self.HR_DIM, cx, cy)
        x = self.hr_n1(q)
        attn_out = self.hr_attn(x, tokp, tok, need_weights=False)[0]  # [B,K,64]
        hrf = self.hr_out(attn_out)
        hrf = hrf + self.hr_ffn(self.hr_n2(hrf))
        # 确定性原始颜色池化：高斯权 σ = bbox 尺寸/4（8px token 网格）
        gx1 = (torch.arange(gw, device=img.device, dtype=img.dtype) + 0.5) / gw
        gy1 = (torch.arange(gh, device=img.device, dtype=img.dtype) + 0.5) / gh
        sx = (w / 4).clamp(min=1.0 / gw).unsqueeze(-1)         # [B,K,1]
        sy = (hgt / 4).clamp(min=1.0 / gh).unsqueeze(-1)
        gwx = torch.exp(-0.5 * ((gx1.view(1, 1, -1) - cx.unsqueeze(-1)) / sx) ** 2)  # [B,K,gw]
        gwy = torch.exp(-0.5 * ((gy1.view(1, 1, -1) - cy.unsqueeze(-1)) / sy) ** 2)  # [B,K,gh]
        wgt = (gwy.unsqueeze(3) * gwx.unsqueeze(2)).reshape(b, -1, gh * gw)          # [B,K,T]
        wsum = wgt.sum(-1, keepdim=True).clamp(min=1e-6)
        cf_raw = torch.einsum("bkt,btc->bkc", wgt, rawt) / wsum
        return hrf, cf_raw, hr, rawt

    def forward(self, img, anchor_scale: float = 1.0):
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
        # 按契约索引拼装 slot 张量；备用槽（I_END..SLOT_DIM）恒 0。
        geom = self.geom_head(h)
        seg = self.seg_head(h)
        # 高分辨率颜色特征（只供 fill/stroke 头，不动几何通路）。
        # bbox 中心/尺寸取 geom raw + 锚点（与最终 bbox 一致），detach。
        cx = _lin(geom[..., I_BBOX + 0] + self.spatial_anchor[..., 0] * anchor_scale,
                  C_MIN, C_MAX).detach()
        cy = _lin(geom[..., I_BBOX + 1] + self.spatial_anchor[..., 1] * anchor_scale,
                  C_MIN, C_MAX).detach()
        bw = _lin(geom[..., I_BBOX + 2], W_MIN, W_MAX).detach()
        bh = _lin(geom[..., I_BBOX + 3], W_MIN, W_MAX).detach()
        hrf, cf_raw, hr_ctx, rawt = self._hr_feat(img, h, cx, cy, bw, bh)
        # ---- E 方案：掩码预测 + 掩码内池化取色（§11.15）----
        # v2：query cross-attn 精化 + pixel embedding 点积。取色梯度经池化
        # 权重反传回掩码头：取色需求直接塑造分割。
        gh, gw = hr_ctx.shape[-2:]
        mfeat = self.mask_tok(hr_ctx)                          # [B,64,gh,gw]
        T = gh * gw
        mft = mfeat.flatten(2).transpose(1, 2)                 # [B,T,64]
        pe = _sine_pos_2d(self.HR_DIM, gh, gw, mft.device, mft.dtype)
        mft = mft + pe
        pix = self.mask_pix(mfeat).flatten(2).transpose(1, 2)  # [B,T,64]
        q = self.mask_qproj(h)                                 # [B,K,64]
        x = self.mask_n1(q)
        q = q + self.mask_attn(x, mft, mft, need_weights=False)[0]
        q = q + self.mask_ffn(self.mask_n2(q))
        mask_logits = (torch.einsum("bkc,btc->bkt", q, pix)
                       * (self.HR_DIM ** -0.5)
                       + self.mask_bias.view(1, -1, 1))       # bias 对齐 K 维
        mask_logits = mask_logits.reshape(b, self.num_slots, gh, gw)
        # cf_mask 像素粒度池化（E-v3）：掩码上采样到输入分辨率后逐像素
        # 加权原始颜色，再按 8px 块聚合。v2 的 token 粒度（整块 token
        # 均值）被描边/背景污染，探针实测天花板≈灰（0.26）；像素粒度
        # （边界像素低权重=内置腐蚀）实测 0.17-0.20 < 灰 0.23-0.26。
        pm = torch.sigmoid(mask_logits)                            # [B,K,gh,gw]
        blk = img.shape[-1] // gh
        pm_up = F.interpolate(pm, size=img.shape[-2:], mode="bilinear",
                              align_corners=False)                 # [B,K,H,W]
        prod = F.avg_pool2d(
            (pm_up.unsqueeze(2) * img.unsqueeze(1)).flatten(0, 1),
            blk).reshape(b, self.num_slots, 4, gh, gw)             # [B,K,4,gh,gw]
        mass = F.avg_pool2d(pm_up, blk)                            # [B,K,gh,gw]
        cf_mask = (torch.einsum("bkchw,bkhw->bkc", prod, mass)
                   / mass.sum((2, 3)).clamp_min(1e-6).unsqueeze(-1))  # [B,K,4]
        color_in = torch.cat([h, hrf, cf_raw, cf_mask], dim=-1)   # [B,K,328]
        fill = self.fill_head(color_in)
        stroke = self.stroke_head(color_in)
        # §11.13 色板分类：solid fill 颜色改为 palette softmax 加权色（可微），
        # 直接写入契约 rgb 字段——渲染/解码自动一致，CE 提供强分类梯度，
        # 消灭"回归均值灰"的退路。logits 经 aux 返回供损失监督。
        pal_logits = self.palette_head(color_in)               # [B,K,N_PAL]
        pal_soft = torch.softmax(pal_logits, dim=-1) @ self.palette_rgb
        fill[..., self.I_FRGB_OFF:self.I_FRGB_OFF + 3] = pal_soft
        fx = self.fx_head(h)
        comp = self.comp_head(h)
        spare = SLOT_DIM - I_END
        slots = torch.cat([geom, seg, fill, stroke, fx, comp], dim=-1)
        if spare > 0:
            slots = torch.cat([slots, slots.new_zeros(b, self.num_slots, spare)], dim=-1)
        # 空间锚点先验按 anchor_scale 缩放：训练早期=1.0 打破对称坍缩，
        # 后期退火到 0 让对象学到任意连续位置（消除网格偏置）。
        slots[..., I_BBOX:I_BBOX + 2] = (slots[..., I_BBOX:I_BBOX + 2]
                                         + self.spatial_anchor * anchor_scale)
        aux = {"palette": pal_logits, "mask": mask_logits}
        bg = self.bg_head(tokens.mean(dim=1))
        return slots, aux, bg


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
