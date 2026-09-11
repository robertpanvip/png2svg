from __future__ import annotations

import math

import numpy as np
import torch

from model.targets import (
    NUM_SLOTS, N_SEG, SEG_DIM, NUM_STOPS, NUM_DASH, BG_DIM,
    SEG_M, SEG_L, SEG_Q, SEG_C, SEG_A,
    I_VALID, I_CLOSED, I_BBOX, I_NSEG, I_SEG,
    I_FTYPE, I_FRGB, I_FALPHA, I_GP0, I_GP1, I_GRAD, I_STOPS, I_NSTOPS,
    I_SVALID, I_SWIDTH, I_SRGB, I_SALPHA, I_SDASH, I_SCAP, I_SJOIN,
    I_OPACITY, I_BLUR, I_SDX, I_SDY, I_SBLUR, I_ESRGB, I_ESALPHA,
    I_GLOWR, I_EGRGB, I_EGALPHA,
    I_GROUP, I_CLIP, I_CLIP_REF, I_MASK, I_MASK_REF, I_MASK_KIND,
    _slot_template,
)
from svg.scene_graph import FILL_LINEAR, FILL_RADIAL

C_MIN, C_MAX = -0.20, 1.20
W_MIN, W_MAX = 0.02, 1.30
G_MIN, G_MAX = -1.00, 2.00
R_MIN, R_MAX = 0.02, 1.50
SW_MIN, SW_MAX = 0.002, 0.25
E_MIN, E_MAX = 0.0, 0.20        # effects 半径/模糊上限（uv）
SD_MIN, SD_MAX = -0.20, 0.20    # shadow 位移上限（uv）
VALID_SKIP = 0.05
CURVE_SAMPLES = 60


def _lin(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def squash_slots(slots_raw: torch.Tensor) -> dict:
    """网络 raw 输出（logit 空间）→ 各块 [0,1]/物理量，全程可微。

    one-hot 块（段类型/ftype/cap/join/group）保留 raw logits，由渲染侧
    argmax 取硬选择、损失侧 CE 监督。
    """
    v = slots_raw
    shp = v.shape[:-1]
    seg = v[..., I_SEG:I_SEG + N_SEG * SEG_DIM].reshape(*shp, N_SEG, SEG_DIM)
    return {
        "valid": torch.sigmoid(v[..., I_VALID]),
        "closed": torch.sigmoid(v[..., I_CLOSED]),
        "cx": _lin(v[..., I_BBOX + 0], C_MIN, C_MAX),
        "cy": _lin(v[..., I_BBOX + 1], C_MIN, C_MAX),
        "w": _lin(v[..., I_BBOX + 2], W_MIN, W_MAX),
        "h": _lin(v[..., I_BBOX + 3], W_MIN, W_MAX),
        "nseg": N_SEG * torch.sigmoid(v[..., I_NSEG]),
        "seg_type": seg[..., :5],                       # raw logits [.., N_SEG, 5]
        "seg_c": torch.sigmoid(seg[..., 5:]),           # [.., N_SEG, 7]
        "ftype": v[..., I_FTYPE:I_FTYPE + 4],           # raw logits
        "rgb": torch.sigmoid(v[..., I_FRGB:I_FRGB + 3]),
        "fill_alpha": torch.sigmoid(v[..., I_FALPHA]),
        "gp0": _lin(v[..., I_GP0:I_GP0 + 2], G_MIN, G_MAX),
        "gp1": _lin(v[..., I_GP1:I_GP1 + 2], G_MIN, G_MAX),
        "radius": _lin(v[..., I_GRAD], R_MIN, R_MAX),
        "stops": torch.sigmoid(v[..., I_STOPS:I_STOPS + NUM_STOPS * 5]).reshape(*shp, NUM_STOPS, 5),
        "nstops": 2.0 + 2.0 * torch.sigmoid(v[..., I_NSTOPS]),
        "stroke_valid": torch.sigmoid(v[..., I_SVALID]),
        "sw": _lin(v[..., I_SWIDTH], SW_MIN, SW_MAX),
        "stroke_rgb": torch.sigmoid(v[..., I_SRGB:I_SRGB + 3]),
        "stroke_alpha": torch.sigmoid(v[..., I_SALPHA]),
        "dash": torch.sigmoid(v[..., I_SDASH:I_SDASH + NUM_DASH]),
        "ndash": NUM_DASH * torch.sigmoid(v[..., I_SDASH + NUM_DASH]),
        "cap": v[..., I_SCAP:I_SCAP + 3],               # raw logits
        "join": v[..., I_SJOIN:I_SJOIN + 3],            # raw logits
        "opacity": torch.sigmoid(v[..., I_OPACITY]),
        "blur": _lin(v[..., I_BLUR], E_MIN, E_MAX),
        "sdx": _lin(v[..., I_SDX], SD_MIN, SD_MAX),
        "sdy": _lin(v[..., I_SDY], SD_MIN, SD_MAX),
        "sblur": _lin(v[..., I_SBLUR], E_MIN, E_MAX),
        "esrgb": torch.sigmoid(v[..., I_ESRGB:I_ESRGB + 3]),
        "esalpha": torch.sigmoid(v[..., I_ESALPHA]),
        "glowr": _lin(v[..., I_GLOWR], E_MIN, E_MAX),
        "egrgb": torch.sigmoid(v[..., I_EGRGB:I_EGRGB + 3]),
        "egalpha": torch.sigmoid(v[..., I_EGALPHA]),
        "group": v[..., I_GROUP:I_GROUP + 4],           # raw logits
        "clip": torch.sigmoid(v[..., I_CLIP]),
        "clip_ref": v[..., I_CLIP_REF],                 # raw 值（0..7）
        "mask": torch.sigmoid(v[..., I_MASK]),
        "mask_ref": v[..., I_MASK_REF],
        "mask_kind": torch.sigmoid(v[..., I_MASK_KIND]),
    }


def squash_bg(bg_raw: torch.Tensor) -> torch.Tensor:
    valid = torch.sigmoid(bg_raw[..., 0:1])
    return torch.sigmoid(bg_raw[..., 1:5]) * valid


# ---- 可微曲线扁平化（uv → abs 由调用方完成）----

def _quad_points(p0, p1, p2, n, dtype, device):
    t = torch.linspace(0.0, 1.0, n + 1, dtype=dtype, device=device)[1:]
    mt = 1.0 - t
    x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
    y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
    return torch.stack([x, y], dim=-1)


def _cubic_points(p0, p1, p2, p3, n, dtype, device):
    t = torch.linspace(0.0, 1.0, n + 1, dtype=dtype, device=device)[1:]
    mt = 1.0 - t
    x = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
    y = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
    return torch.stack([x, y], dim=-1)


def _arc_points(p0, p1, rx, ry, rot_deg, large, sweep, n, dtype, device):
    """SVG 端点参数化 → 中心参数化采样，对 p0/p1/rx/ry/rot 可微。

    rx/ry 为 abs 画布单位；rot 弧度转角 deg。large/sweep 为硬 flag。
    退化（rx/ry≈0 或端点重合）→ 直线段。
    """
    dx, dy = p0[0] - p1[0], p0[1] - p1[1]
    if float((dx * dx + dy * dy).detach()) < 1e-12 or float(rx.detach()) < 1e-6 or float(ry.detach()) < 1e-6:
        t = torch.linspace(0.0, 1.0, n + 1, dtype=dtype, device=device)[1:].unsqueeze(-1)
        return p0.unsqueeze(0) + t * (p1 - p0).unsqueeze(0)
    phi = math.radians(float(rot_deg.detach()) if torch.is_tensor(rot_deg)
                       else float(rot_deg))
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    x1p = cos_p * dx / 2 + sin_p * dy / 2
    y1p = -sin_p * dx / 2 + cos_p * dy / 2
    lam = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    scale = torch.sqrt(torch.clamp(lam, min=1.0))
    rx_s, ry_s = rx * scale, ry * scale
    num = rx_s * rx_s * ry_s * ry_s - rx_s * rx_s * y1p * y1p - ry_s * ry_s * x1p * x1p
    co = torch.sqrt(torch.clamp(num / (rx_s * rx_s * y1p * y1p + ry_s * ry_s * x1p * x1p), min=0.0))
    if large == sweep:
        co = -co
    cxp = co * rx_s * y1p / ry_s
    cyp = -co * ry_s * x1p / rx_s
    def _angle(ux, uy, vx, vy):
        # atan2(cross, dot)：处处可微（acos 在 ±1 处导数发散，弧起点
        # 落在椭圆 θ=0°/180° 时会触发 NaN 梯度，故弃用）
        cross = ux * vy - uy * vx
        dot = ux * vx + uy * vy
        return torch.atan2(cross, dot)
    theta1 = _angle(torch.tensor(1.0, dtype=dtype, device=device),
                    torch.tensor(0.0, dtype=dtype, device=device),
                    (x1p - cxp) / rx_s, (y1p - cyp) / ry_s)
    dtheta = _angle((x1p - cxp) / rx_s, (y1p - cyp) / ry_s,
                    (-x1p - cxp) / rx_s, (-y1p - cyp) / ry_s)
    if not sweep and float(dtheta.detach()) > 0:
        dtheta = dtheta - 2 * math.pi
    elif sweep and float(dtheta.detach()) < 0:
        dtheta = dtheta + 2 * math.pi
    t = torch.linspace(0.0, 1.0, n + 1, dtype=dtype, device=device)[1:]
    theta = theta1 + dtheta * t
    ex = rx_s * torch.cos(theta)
    ey = ry_s * torch.sin(theta)
    # 中心 = p0/p1 中点 + 旋转 (cxp, cyp)
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    ccx = mx + cos_p * cxp - sin_p * cyp
    ccy = my + sin_p * cxp + cos_p * cyp
    x = ccx + cos_p * ex - sin_p * ey
    y = ccy + sin_p * ex + cos_p * ey
    return torch.stack([x, y], dim=-1)


def _build_geo(f, k: int, canvas: int, pad_px: float, curve_samples: int):
    """slot k 的段表 → ("poly", [(pts[N,2] abs, closed), ...]) + crop。

    段类型 argmax（硬），坐标可微；Q/C/A 扁平化为折线。
    """
    cx, cy, w, h = f["cx"][k], f["cy"][k], f["w"][k], f["h"][k]
    st = f["seg_type"][k]     # [N_SEG, 5] logits
    sc = f["seg_c"][k]        # [N_SEG, 7] in (0,1)
    n_hard = int(np.clip(round(float(f["nseg"][k].detach())), 1, N_SEG))
    closed = bool(float(f["closed"][k].detach()) > 0.5)
    dev, dt = sc.device, sc.dtype

    def ax(u):
        return cx + (u - 0.5) * w

    def ay(v):
        return cy + (v - 0.5) * h

    subs = []
    cur = None  # list of point tensors（不含起点重复）
    start = None

    def flush():
        nonlocal cur, start
        if cur is not None and start is not None and len(cur) > 0:
            pts = torch.stack([start] + cur, dim=0)
            if closed:
                pts = torch.cat([pts, start.unsqueeze(0)], dim=0)
            subs.append(pts)
        cur, start = None, None

    for j in range(n_hard):
        t = int(torch.argmax(st[j].detach()).item())
        c = sc[j]
        if t == SEG_M:
            flush()
            start = torch.stack([ax(c[0]), ay(c[1])])
            cur = []
        elif start is None:
            # 无 M 开头的非法序列：视作从该点开始
            start = torch.stack([ax(c[0]), ay(c[1])])
            cur = []
        elif t == SEG_L:
            cur.append(torch.stack([ax(c[0]), ay(c[1])]))
        elif t == SEG_Q:
            pts = _quad_points(cur[-1] if cur else start,
                               torch.stack([ax(c[0]), ay(c[1])]),
                               torch.stack([ax(c[2]), ay(c[3])]),
                               curve_samples, dt, dev)
            cur.extend(list(pts))
        elif t == SEG_C:
            pts = _cubic_points(cur[-1] if cur else start,
                                torch.stack([ax(c[0]), ay(c[1])]),
                                torch.stack([ax(c[2]), ay(c[3])]),
                                torch.stack([ax(c[4]), ay(c[5])]),
                                curve_samples, dt, dev)
            cur.extend(list(pts))
        else:  # SEG_A
            p0 = cur[-1] if cur else start
            p1 = torch.stack([ax(c[5]), ay(c[6])])
            pts = _arc_points(p0, p1, c[0] * w, c[1] * h,
                              c[2] * 360.0,
                              float(c[3].detach()) > 0.5,
                              float(c[4].detach()) > 0.5,
                              curve_samples, dt, dev)
            cur.extend(list(pts))
    flush()
    subs = [p for p in subs if p.shape[0] >= 2]
    if not subs:
        return None, None

    geo = ("poly", [(p, closed) for p in subs])
    allp = torch.cat(subs, dim=0).detach()
    hw_px = 0.5 * float(f["sw"][k].detach()) * canvas
    pad = (pad_px + hw_px) / canvas
    gx0, gy0 = float(allp[:, 0].min()), float(allp[:, 1].min())
    gx1, gy1 = float(allp[:, 0].max()), float(allp[:, 1].max())
    x0 = max(0, int(math.floor((gx0 - pad) * canvas)))
    y0 = max(0, int(math.floor((gy0 - pad) * canvas)))
    x1 = min(canvas, int(math.ceil((gx1 + pad) * canvas)))
    y1 = min(canvas, int(math.ceil((gy1 + pad) * canvas)))
    return geo, (x0, y0, x1, y1)


def slots_to_objs(slots_raw: torch.Tensor, bg_raw: torch.Tensor,
                  canvas: int, pad_px: float, curve_samples: int = CURVE_SAMPLES):
    """新契约：段表 → 渲染 dict 列表（几何/填充/描边可微）。

    P3b（HANDOFF §11.17）：dash 已接入 soft 渲染路径（GT 与预测一致，
    ndash 硬取整、dash 值可微）。cap/join、effects、clip/mask 仍不在
    soft 渲染路径（辅助损失监督，见 HANDOFF §11.6 渲染器契约注记）。
    """
    f = squash_slots(slots_raw)
    objs = []
    for k in range(NUM_SLOTS):
        valid = f["valid"][k]
        if float(valid.detach()) < VALID_SKIP:
            continue
        geo, crop = _build_geo(f, k, canvas, pad_px, curve_samples)
        if geo is None:
            continue

        ftype = int(torch.argmax(f["ftype"][k].detach()).item())
        fill_spec = None
        if ftype != 0:  # 0=none
            gate = f["fill_alpha"][k] * valid
            if ftype == 1:  # solid
                fill_spec = ("solid", f["rgb"][k], gate)
            else:
                n_s = int(np.clip(round(float(f["nstops"][k].detach())), 2, NUM_STOPS))
                st_used = f["stops"][k][:n_s]
                pos, order = torch.sort(st_used[:, 0])
                grad = {
                    "p0": f["gp0"][k], "p1": f["gp1"][k], "radius": f["radius"][k],
                    "pos": pos, "rgb": st_used[order, 1:4], "alpha": st_used[order, 4],
                }
                kind = FILL_LINEAR if ftype == 2 else FILL_RADIAL
                fill_spec = (kind, grad, gate)

        hw = 0.5 * f["sw"][k]
        # P3b dash（HANDOFF §11.17）：ndash 硬取整（与 ftype argmax 同风格），
        # dash 值保持可微（渲染损失可回传）；nd=0 → 无 dash，与旧行为一致
        dash_t = None
        nd = int(np.clip(round(float(f["ndash"][k].detach())), 0, NUM_DASH))
        if nd > 0 and float(f["stroke_valid"][k].detach()) > 0.5:
            dash_t = f["dash"][k][:nd]
        stroke_spec = (hw, f["stroke_rgb"][k],
                       f["stroke_alpha"][k] * f["stroke_valid"][k] * valid,
                       dash_t)

        opacity = f["opacity"][k] * valid
        objs.append({"geo": geo, "fill": fill_spec, "stroke": stroke_spec,
                     "opacity": opacity, "crop": crop})
    return objs, squash_bg(bg_raw)


def _lg(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def targets_to_raw(slots: np.ndarray) -> torch.Tensor:
    """GT 契约 slot → 网络 raw 空间（squash(raw) ≈ GT）。

    one-hot 块用 ±4 的饱和 logits 近似硬选择。
    """
    v = np.asarray(slots, dtype=np.float64)
    raw = np.zeros_like(v)

    def inv(x, lo, hi):
        return _lg((x - lo) / (hi - lo))

    raw[:, I_VALID] = _lg(v[:, I_VALID])
    raw[:, I_CLOSED] = _lg(v[:, I_CLOSED])
    raw[:, I_BBOX + 0] = inv(v[:, I_BBOX + 0], C_MIN, C_MAX)
    raw[:, I_BBOX + 1] = inv(v[:, I_BBOX + 1], C_MIN, C_MAX)
    raw[:, I_BBOX + 2] = inv(v[:, I_BBOX + 2], W_MIN, W_MAX)
    raw[:, I_BBOX + 3] = inv(v[:, I_BBOX + 3], W_MIN, W_MAX)
    raw[:, I_NSEG] = _lg(v[:, I_NSEG] / N_SEG)

    # 段表：one-hot ±4；坐标 inv-logit
    seg = v[:, I_SEG:I_SEG + N_SEG * SEG_DIM].reshape(-1, N_SEG, SEG_DIM)
    out_seg = np.full((v.shape[0], N_SEG, SEG_DIM), -4.0)
    for j in range(N_SEG):
        hot = np.argmax(seg[:, j, :5], axis=1)
        out_seg[np.arange(v.shape[0]), j, hot] = 4.0
        out_seg[:, j, 5:] = _lg(seg[:, j, 5:])
    raw[:, I_SEG:I_SEG + N_SEG * SEG_DIM] = out_seg.reshape(v.shape[0], -1)

    raw[:, I_FTYPE:I_FTYPE + 4] = -4.0
    ft = np.argmax(v[:, I_FTYPE:I_FTYPE + 4], axis=1)
    raw[np.arange(v.shape[0]), I_FTYPE + ft] = 4.0
    raw[:, I_FRGB:I_FRGB + 3] = _lg(v[:, I_FRGB:I_FRGB + 3])
    raw[:, I_FALPHA] = _lg(v[:, I_FALPHA])
    raw[:, I_GP0:I_GP0 + 2] = inv(v[:, I_GP0:I_GP0 + 2], G_MIN, G_MAX)
    raw[:, I_GP1:I_GP1 + 2] = inv(v[:, I_GP1:I_GP1 + 2], G_MIN, G_MAX)
    raw[:, I_GRAD] = inv(v[:, I_GRAD], R_MIN, R_MAX)
    raw[:, I_STOPS:I_STOPS + NUM_STOPS * 5] = _lg(v[:, I_STOPS:I_STOPS + NUM_STOPS * 5])
    raw[:, I_NSTOPS] = _lg((v[:, I_NSTOPS] - 2.0) / 2.0)
    raw[:, I_SVALID] = _lg(v[:, I_SVALID])
    raw[:, I_SWIDTH] = inv(v[:, I_SWIDTH], SW_MIN, SW_MAX)
    raw[:, I_SRGB:I_SRGB + 3] = _lg(v[:, I_SRGB:I_SRGB + 3])
    raw[:, I_SALPHA] = _lg(v[:, I_SALPHA])
    raw[:, I_SDASH:I_SDASH + NUM_DASH] = _lg(v[:, I_SDASH:I_SDASH + NUM_DASH])
    raw[:, I_SDASH + NUM_DASH] = _lg(v[:, I_SDASH + NUM_DASH] / NUM_DASH)
    raw[:, I_SCAP:I_SCAP + 3] = -4.0
    cap = np.argmax(v[:, I_SCAP:I_SCAP + 3], axis=1)
    raw[np.arange(v.shape[0]), I_SCAP + cap] = 4.0
    raw[:, I_SJOIN:I_SJOIN + 3] = -4.0
    jn = np.argmax(v[:, I_SJOIN:I_SJOIN + 3], axis=1)
    raw[np.arange(v.shape[0]), I_SJOIN + jn] = 4.0
    raw[:, I_OPACITY] = _lg(v[:, I_OPACITY])
    raw[:, I_BLUR] = inv(v[:, I_BLUR], E_MIN, E_MAX)
    raw[:, I_SDX] = inv(v[:, I_SDX], SD_MIN, SD_MAX)
    raw[:, I_SDY] = inv(v[:, I_SDY], SD_MIN, SD_MAX)
    raw[:, I_SBLUR] = inv(v[:, I_SBLUR], E_MIN, E_MAX)
    raw[:, I_ESRGB:I_ESRGB + 3] = _lg(v[:, I_ESRGB:I_ESRGB + 3])
    raw[:, I_ESALPHA] = _lg(v[:, I_ESALPHA])
    raw[:, I_GLOWR] = inv(v[:, I_GLOWR], E_MIN, E_MAX)
    raw[:, I_EGRGB:I_EGRGB + 3] = _lg(v[:, I_EGRGB:I_EGRGB + 3])
    raw[:, I_EGALPHA] = _lg(v[:, I_EGALPHA])
    raw[:, I_GROUP:I_GROUP + 4] = -4.0
    grp = np.argmax(v[:, I_GROUP:I_GROUP + 4], axis=1)
    raw[np.arange(v.shape[0]), I_GROUP + grp] = 4.0
    raw[:, I_CLIP] = _lg(v[:, I_CLIP])
    raw[:, I_CLIP_REF] = v[:, I_CLIP_REF]
    raw[:, I_MASK] = _lg(v[:, I_MASK])
    raw[:, I_MASK_REF] = v[:, I_MASK_REF]
    raw[:, I_MASK_KIND] = _lg(v[:, I_MASK_KIND])
    return torch.tensor(raw, dtype=torch.float32)


def bg_to_raw(bg: np.ndarray) -> torch.Tensor:
    bg = np.asarray(bg, dtype=np.float64)
    out = np.zeros(BG_DIM, dtype=np.float64)
    out[0] = _lg(bg[0])
    out[1:5] = _lg(bg[1:5])
    return torch.tensor(out, dtype=torch.float32)


def predictions_to_targets(slots_raw: torch.Tensor, bg_raw: torch.Tensor):
    """模型 raw 输出 → 硬化契约 slot（供 prune/导出/评估）。"""
    f = squash_slots(slots_raw)
    slots = np.stack([_slot_template() for _ in range(NUM_SLOTS)])
    fdet = {key: val.detach().cpu().numpy() for key, val in f.items()}
    for k in range(NUM_SLOTS):
        if fdet["valid"][k] < 0.5:
            continue
        v = slots[k]
        v[I_VALID] = 1.0
        v[I_CLOSED] = 1.0 if fdet["closed"][k] > 0.5 else 0.0
        v[I_BBOX:I_BBOX + 4] = (fdet["cx"][k], fdet["cy"][k], fdet["w"][k], fdet["h"][k])
        n = int(np.clip(round(float(fdet["nseg"][k])), 1, N_SEG))
        v[I_NSEG] = float(n)
        st = fdet["seg_type"][k]
        sc = fdet["seg_c"][k]
        seg_out = np.zeros((N_SEG, SEG_DIM), dtype=np.float64)
        for j in range(n):
            seg_out[j, int(np.argmax(st[j]))] = 1.0
            seg_out[j, 5:] = sc[j]
        v[I_SEG:I_SEG + N_SEG * SEG_DIM] = seg_out.reshape(-1)
        ft = int(np.argmax(fdet["ftype"][k]))
        v[I_FTYPE:I_FTYPE + 4] = 0.0
        v[I_FTYPE + ft] = 1.0
        if ft == 1:
            v[I_FRGB:I_FRGB + 3] = fdet["rgb"][k]
            v[I_FALPHA] = float(fdet["fill_alpha"][k])
        elif ft in (2, 3):
            n_s = int(np.clip(round(float(fdet["nstops"][k])), 2, NUM_STOPS))
            v[I_GP0:I_GP0 + 2] = fdet["gp0"][k]
            v[I_GP1:I_GP1 + 2] = fdet["gp1"][k]
            v[I_GRAD] = float(fdet["radius"][k])
            stp = fdet["stops"][k]
            order = np.argsort(stp[:n_s, 0])
            for j, jj in enumerate(order):
                base = I_STOPS + j * 5
                v[base:base + 5] = (float(stp[jj, 0]), stp[jj, 1], stp[jj, 2],
                                    stp[jj, 3], stp[jj, 4])
            v[I_NSTOPS] = float(n_s)
        else:
            v[I_FALPHA] = 0.0
        if fdet["stroke_valid"][k] > 0.5:
            v[I_SVALID] = 1.0
            v[I_SWIDTH] = float(fdet["sw"][k])
            v[I_SRGB:I_SRGB + 3] = fdet["stroke_rgb"][k]
            v[I_SALPHA] = float(fdet["stroke_alpha"][k])
            nd = int(np.clip(round(float(fdet["ndash"][k])), 0, NUM_DASH))
            v[I_SDASH:I_SDASH + nd] = fdet["dash"][k][:nd]
            v[I_SDASH + NUM_DASH] = float(nd)
            v[I_SCAP:I_SCAP + 3] = 0.0
            v[I_SCAP + int(np.argmax(fdet["cap"][k]))] = 1.0
            v[I_SJOIN:I_SJOIN + 3] = 0.0
            v[I_SJOIN + int(np.argmax(fdet["join"][k]))] = 1.0
        v[I_OPACITY] = float(fdet["opacity"][k])
        v[I_BLUR] = float(fdet["blur"][k])
        v[I_SDX] = float(fdet["sdx"][k])
        v[I_SDY] = float(fdet["sdy"][k])
        v[I_SBLUR] = float(fdet["sblur"][k])
        v[I_ESRGB:I_ESRGB + 3] = fdet["esrgb"][k]
        v[I_ESALPHA] = float(fdet["esalpha"][k])
        v[I_GLOWR] = float(fdet["glowr"][k])
        v[I_EGRGB:I_EGRGB + 3] = fdet["egrgb"][k]
        v[I_EGALPHA] = float(fdet["egalpha"][k])
        v[I_GROUP:I_GROUP + 4] = 0.0
        v[I_GROUP + int(np.argmax(fdet["group"][k]))] = 1.0
        v[I_CLIP] = 1.0 if fdet["clip"][k] > 0.5 else 0.0
        v[I_CLIP_REF] = float(fdet["clip_ref"][k])
        v[I_MASK] = 1.0 if fdet["mask"][k] > 0.5 else 0.0
        v[I_MASK_REF] = float(fdet["mask_ref"][k])
        v[I_MASK_KIND] = float(fdet["mask_kind"][k])
    v_bg = float(torch.sigmoid(bg_raw[0].detach()))
    rgba = torch.sigmoid(bg_raw[1:5]).detach().cpu().numpy()
    if v_bg > 0.5:
        bg = np.concatenate([[1.0], rgba]).astype(np.float32)
    else:
        bg = np.zeros(BG_DIM, dtype=np.float32)
    return slots.astype(np.float32), bg
