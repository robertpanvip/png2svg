from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from model.targets import (
    NUM_SLOTS, N_SEG, SEG_DIM, NUM_STOPS, NUM_DASH,
    I_VALID, I_CLOSED, I_BBOX, I_NSEG, I_SEG,
    I_FTYPE, I_FRGB, I_FALPHA, I_GP0, I_GP1, I_GRAD, I_STOPS, I_NSTOPS,
    I_SVALID, I_SWIDTH, I_SRGB, I_SALPHA, I_SDASH, I_SCAP, I_SJOIN,
    I_OPACITY, I_BLUR, I_SDX, I_SDY, I_SBLUR, I_ESRGB, I_ESALPHA,
    I_GLOWR, I_EGRGB, I_EGALPHA,
    I_GROUP, I_CLIP, I_CLIP_REF, I_MASK, I_MASK_REF, I_MASK_KIND,
)
from model.spec import squash_slots, squash_bg
from model.matching import match_slots

# 每段类型占用的坐标槽数（M,L,Q,C,A）——GT 未用槽恒 0，损失按类型掩码
_COORD_N = (2, 2, 4, 6, 7)


def _ssim_mean(img_pred: torch.Tensor, img_gt: torch.Tensor, window: int = 7) -> torch.Tensor:
    x = img_pred.unsqueeze(0)
    y = img_gt.unsqueeze(0)
    pad = window // 2
    mx = F.avg_pool2d(x, window, stride=1, padding=pad)
    my = F.avg_pool2d(y, window, stride=1, padding=pad)
    mxx = F.avg_pool2d(x * x, window, stride=1, padding=pad)
    myy = F.avg_pool2d(y * y, window, stride=1, padding=pad)
    mxy = F.avg_pool2d(x * y, window, stride=1, padding=pad)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mx * my + c1) * (2 * mxy + c2)) / \
           ((mx * mx + my * my + c1) * (mxx + myy + c2))
    return ssim.mean()


def render_losses(img_pred: torch.Tensor, img_gt: torch.Tensor,
                  ssim_weight: float = 0.3) -> dict:
    mae = (img_pred - img_gt).abs().mean()
    ssim = 1.0 - _ssim_mean(img_pred, img_gt)
    return {"mae": mae, "ssim": ssim, "total": mae + ssim_weight * ssim}


def _masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
               graph: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return graph.sum() * 0.0
    diff = (pred - gt).abs()
    while diff.dim() > mask.dim():
        mask = mask.unsqueeze(-1).expand_as(diff)
    return diff[mask].mean()


def _appearance_block(f: dict, gt: torch.Tensor) -> torch.Tensor:
    """外观直接监督（匹配对）：fill 颜色/alpha + 描边颜色/alpha + opacity。

    P-Appearance（HANDOFF §11.10）：颜色通路原先埋在几何块均值里
    （约 3/50 的梯度占比），40k 步仍全员卡在数据集均值灰——梯度在
    共享权重上互相抵消。拆出独立权重 w_fill 高强度监督。
    f / gt 均已按匹配对重排。
    """
    device = gt.device
    B = gt.shape[0]
    zero = gt.sum() * 0.0
    out = zero

    ft_gt = gt[:, I_FTYPE:I_FTYPE + 4].argmax(-1)              # [B]
    m_solid = ft_gt == 1
    out = out + _masked_l1(f["rgb"], gt[:, I_FRGB:I_FRGB + 3], m_solid, zero)
    out = out + _masked_l1(f["fill_alpha"], gt[:, I_FALPHA],
                           torch.ones(B, dtype=torch.bool, device=device), zero)
    m_grad = (ft_gt == 2) | (ft_gt == 3)
    n_stops_gt = gt[:, I_NSTOPS]
    j_stops = torch.arange(NUM_STOPS, device=device)
    m_stop = m_grad.unsqueeze(1) & (j_stops.unsqueeze(0) < n_stops_gt.unsqueeze(1))
    stops = f["stops"]                                          # [B, NUM_STOPS, 5]
    gt_stops = gt[:, I_STOPS:I_STOPS + NUM_STOPS * 5].reshape(B, NUM_STOPS, 5)
    # stops rgb/alpha（位置 pos 属几何，留在 geom 块）
    out = out + _masked_l1(stops[..., 1:5], gt_stops[..., 1:5], m_stop, zero)

    # 描边颜色/alpha
    m_stroke = gt[:, I_SVALID] > 0.5
    out = out + _masked_l1(f["stroke_rgb"], gt[:, I_SRGB:I_SRGB + 3], m_stroke, zero)
    out = out + _masked_l1(f["stroke_alpha"], gt[:, I_SALPHA], m_stroke, zero)

    # opacity（合成外观）
    out = out + _masked_l1(f["opacity"], gt[:, I_OPACITY],
                           torch.ones(B, dtype=torch.bool, device=device), zero)
    return out


def _geom_block(f: dict, slots_raw: torch.Tensor, gt: torch.Tensor,
                include_appearance: bool = True) -> torch.Tensor:
    """对匹配对（全部 valid）计算新契约的全块监督。

    f / slots_raw / gt 均已按匹配对重排。
    """
    device = slots_raw.device
    B = gt.shape[0]
    zero = slots_raw.sum() * 0.0
    out = zero

    nseg_gt = gt[:, I_NSEG]
    closed_gt = gt[:, I_CLOSED]
    j_seg = torch.arange(N_SEG, device=device)
    m_seg = j_seg.unsqueeze(0) < nseg_gt.unsqueeze(1)          # [B,16]

    # ---- 段表 ----
    gt_seg = gt[:, I_SEG:I_SEG + N_SEG * SEG_DIM].reshape(B, N_SEG, SEG_DIM)
    gt_type = gt_seg[..., :5].argmax(-1)                       # [B,16]
    gt_coord = gt_seg[..., 5:]                                 # [B,16,7]
    # 段类型 CE（w_cls 权重在调用侧施加）
    logits = f["seg_type"]                                     # [B,16,5] raw
    ce = F.cross_entropy(logits.reshape(-1, 5), gt_type.reshape(-1),
                         reduction="none").reshape(B, N_SEG)
    out = out + (ce * m_seg.float()).sum() / m_seg.float().sum().clamp(min=1.0)
    # 段坐标 masked-L1：坐标槽占用数随类型而变
    coord_n = torch.tensor(_COORD_N, device=device)[gt_type]   # [B,16]
    col = torch.arange(7, device=device)
    m_coord = m_seg.unsqueeze(-1) & (col.unsqueeze(0).unsqueeze(0) < coord_n.unsqueeze(-1))
    out = out + _masked_l1(f["seg_c"], gt_coord, m_coord, slots_raw)
    # nseg 软计数
    out = out + _masked_l1(f["nseg"], nseg_gt, torch.ones(B, dtype=torch.bool, device=device),
                           slots_raw)
    out = out + _masked_l1(f["closed"], closed_gt,
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)

    # ---- bbox / opacity ----
    out = out + _masked_l1(
        torch.stack([f["cx"], f["cy"], f["w"], f["h"]], dim=1),
        gt[:, I_BBOX:I_BBOX + 4],
        torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    if include_appearance:
        out = out + _masked_l1(f["opacity"], gt[:, I_OPACITY],
                               torch.ones(B, dtype=torch.bool, device=device), slots_raw)

    # ---- fill ----
    # 颜色/alpha 项已拆到 _appearance_block（w_fill 独立权重）；
    # 此处保留渐变几何（gp/radius/stops pos/nstops）。
    ft_gt = gt[:, I_FTYPE:I_FTYPE + 4].argmax(-1)              # [B]
    if include_appearance:
        m_solid = ft_gt == 1
        out = out + _masked_l1(f["rgb"], gt[:, I_FRGB:I_FRGB + 3], m_solid, slots_raw)
        out = out + _masked_l1(f["fill_alpha"], gt[:, I_FALPHA],
                               torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    m_grad = (ft_gt == 2) | (ft_gt == 3)
    n_stops_gt = gt[:, I_NSTOPS]
    j_stops = torch.arange(NUM_STOPS, device=device)
    m_stop = m_grad.unsqueeze(1) & (j_stops.unsqueeze(0) < n_stops_gt.unsqueeze(1))
    out = out + _masked_l1(f["gp0"], gt[:, I_GP0:I_GP0 + 2], m_grad, slots_raw)
    out = out + _masked_l1(f["gp1"], gt[:, I_GP1:I_GP1 + 2], m_grad, slots_raw)
    out = out + _masked_l1(f["radius"], gt[:, I_GRAD], m_grad, slots_raw)
    gt_stops_geom = gt[:, I_STOPS:I_STOPS + NUM_STOPS * 5].reshape(B, NUM_STOPS, 5)
    out = out + _masked_l1(f["stops"][..., :1], gt_stops_geom[..., :1], m_stop, slots_raw)
    out = out + _masked_l1(f["nstops"], n_stops_gt, m_grad, slots_raw)

    # ---- stroke ----
    m_stroke = gt[:, I_SVALID] > 0.5
    out = out + _masked_l1(f["sw"], gt[:, I_SWIDTH], m_stroke, slots_raw)
    if include_appearance:
        out = out + _masked_l1(f["stroke_rgb"], gt[:, I_SRGB:I_SRGB + 3], m_stroke, slots_raw)
        out = out + _masked_l1(f["stroke_alpha"], gt[:, I_SALPHA], m_stroke, slots_raw)
    nd_gt = gt[:, I_SDASH + NUM_DASH]
    j_dash = torch.arange(NUM_DASH, device=device)
    m_dash = m_stroke.unsqueeze(1) & (j_dash.unsqueeze(0) < nd_gt.unsqueeze(1))
    out = out + _masked_l1(f["dash"], gt[:, I_SDASH:I_SDASH + NUM_DASH]
                           .reshape(B, NUM_DASH), m_dash, slots_raw)
    out = out + _masked_l1(f["ndash"], nd_gt, m_stroke, slots_raw)
    cap_gt = gt[:, I_SCAP:I_SCAP + 3].argmax(-1)
    jn_gt = gt[:, I_SJOIN:I_SJOIN + 3].argmax(-1)
    if bool(m_stroke.any()):
        out = out + F.cross_entropy(f["cap"][m_stroke], cap_gt[m_stroke])
        out = out + F.cross_entropy(f["join"][m_stroke], jn_gt[m_stroke])

    # ---- effects（GT 无效果处恒 0，L1 推向 0）----
    out = out + _masked_l1(f["blur"], gt[:, I_BLUR],
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    out = out + _masked_l1(torch.stack([f["sdx"], f["sdy"], f["sblur"]], dim=1),
                           gt[:, I_SDX:I_SBLUR + 1],
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    out = out + _masked_l1(f["esrgb"], gt[:, I_ESRGB:I_ESRGB + 3],
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    out = out + _masked_l1(f["esalpha"], gt[:, I_ESALPHA],
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    out = out + _masked_l1(f["glowr"], gt[:, I_GLOWR],
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    out = out + _masked_l1(f["egrgb"], gt[:, I_EGRGB:I_EGRGB + 3],
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    out = out + _masked_l1(f["egalpha"], gt[:, I_EGALPHA],
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)

    # ---- composition ----
    out = out + _masked_l1(f["clip"], gt[:, I_CLIP],
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    m_clip = gt[:, I_CLIP] > 0.5
    out = out + _masked_l1(f["clip_ref"], gt[:, I_CLIP_REF], m_clip, slots_raw)
    out = out + _masked_l1(f["mask"], gt[:, I_MASK],
                           torch.ones(B, dtype=torch.bool, device=device), slots_raw)
    m_mask = gt[:, I_MASK] > 0.5
    out = out + _masked_l1(f["mask_ref"], gt[:, I_MASK_REF], m_mask, slots_raw)
    out = out + _masked_l1(f["mask_kind"], gt[:, I_MASK_KIND], m_mask, slots_raw)
    return out


def matched_auxiliary_losses(slots_raw: torch.Tensor, bg_raw: torch.Tensor,
                             slots_gt, bg_gt,
                             palette_logits: torch.Tensor = None) -> dict:
    """带 Hungarian 匹配的辅助损失（新契约）。

    代价 = bbox L1 + 段类型 NLL（几何签名）。匹配对上施加段表/属性块监督；
    未匹配 slot 推向 invalid。aux 头已并入 slot 向量，参数仅为兼容保留。
    """
    device = slots_raw.device
    gt = torch.as_tensor(np.asarray(slots_gt), dtype=torch.float32, device=device)
    bg_t = torch.as_tensor(np.asarray(bg_gt), dtype=torch.float32, device=device)
    if bg_raw.dim() == 2:
        bg_raw = bg_raw[0]

    f = squash_slots(slots_raw)
    gv = gt[:, I_VALID]
    valid_gt = gv > 0.5

    pred_bbox = torch.stack([f["cx"], f["cy"], f["w"], f["h"]], dim=1).detach()
    gt_seg_all = gt[:, I_SEG:I_SEG + N_SEG * SEG_DIM].reshape(NUM_SLOTS, N_SEG, SEG_DIM)
    gt_type_all = gt_seg_all[..., :5].argmax(-1)               # [G,16]
    nseg_all = gt[:, I_NSEG]

    n = NUM_SLOTS
    with torch.no_grad():
        logp = torch.log_softmax(f["seg_type"].detach(), -1)   # [K,16,5]
        bbox_cost = (pred_bbox.unsqueeze(1) - gt[None, :, I_BBOX:I_BBOX + 4]) \
            .abs().sum(-1).cpu().numpy()                       # [K,G]
        jidx = torch.arange(N_SEG, device=device)
        cost = np.full((n, n), 1e6, dtype=np.float64)
        for g in range(n):
            if not bool(valid_gt[g]):
                continue
            m = jidx < nseg_all[g]
            lp = logp[:, jidx, gt_type_all[g]]                 # [K,16]
            type_nll = (-lp[:, m].mean(-1)).cpu().numpy()      # [K]
            cost[:, g] = bbox_cost[:, g] + type_nll
    assign = match_slots(cost)

    matched_pred = [k for k in assign if k >= 0]
    parts = {}
    valid_target = torch.zeros(n, device=device)
    if matched_pred:
        pred_idx = torch.tensor(matched_pred, device=device)
        gt_idx = torch.tensor([g for g in range(n) if assign[g] >= 0], device=device)
        valid_target[pred_idx] = 1.0
        f_g = {key: val[pred_idx] for key, val in f.items()}
        pred_raw_g = slots_raw[pred_idx]
        gt_g = gt[gt_idx]
        parts["geom"] = _geom_block(f_g, pred_raw_g, gt_g, include_appearance=False)
        parts["appearance"] = _appearance_block(f_g, gt_g)
        # §11.13 色板 CE：GT solid 对象 → 最近色板 id，在匹配对上监督
        if palette_logits is not None:
            from model.palette import rgb_to_id
            if palette_logits.dim() == 3:
                palette_logits = palette_logits[0]
            pal_g = palette_logits[pred_idx]                   # [M,N_PAL]
            ft_g2 = gt_g[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
            m_sol = ft_g2 == 1
            if bool(m_sol.any()):
                gt_rgb = gt_g[m_sol][:, I_FRGB:I_FRGB + 3].detach().cpu().numpy()
                pal_id = torch.as_tensor(rgb_to_id(gt_rgb), device=device,
                                         dtype=torch.long)
                parts["palette"] = F.cross_entropy(pal_g[m_sol], pal_id)
            else:
                parts["palette"] = slots_raw.sum() * 0.0
        parts["bbox"] = F.l1_loss(
            torch.stack([f_g["cx"], f_g["cy"]], dim=1),
            gt_g[:, I_BBOX:I_BBOX + 2])
        # 段类型 CE 单独记账（w_cls 权重）
        gt_seg = gt_g[:, I_SEG:I_SEG + N_SEG * SEG_DIM].reshape(-1, N_SEG, SEG_DIM)
        gt_type = gt_seg[..., :5].argmax(-1)
        nseg_g = gt_g[:, I_NSEG]
        m_seg = torch.arange(N_SEG, device=device).unsqueeze(0) < nseg_g.unsqueeze(1)
        ce = F.cross_entropy(f_g["seg_type"].reshape(-1, 5), gt_type.reshape(-1),
                             reduction="none").reshape(-1, N_SEG)
        parts["cls"] = (ce * m_seg.float()).sum() / m_seg.float().sum().clamp(min=1.0)
        # ftype CE
        ft_gt = gt_g[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
        parts["ftype"] = F.cross_entropy(f_g["ftype"], ft_gt)
        # stroke valid BCE
        parts["svalid"] = F.binary_cross_entropy(
            f_g["stroke_valid"].clamp(1e-6, 1.0 - 1e-6), gt_g[:, I_SVALID])
    else:
        zero = slots_raw.sum() * 0.0
        parts["geom"] = zero
        parts["appearance"] = zero
        parts["palette"] = slots_raw.sum() * 0.0
        parts["bbox"] = zero
        parts["cls"] = zero
        parts["ftype"] = zero
        parts["svalid"] = zero

    parts["valid"] = F.binary_cross_entropy(
        f["valid"].clamp(1e-6, 1.0 - 1e-6), valid_target)

    bg_valid = torch.sigmoid(bg_raw[0]).clamp(1e-6, 1.0 - 1e-6)
    parts["bg"] = F.binary_cross_entropy(bg_valid, bg_t[0]) \
        + (squash_bg(bg_raw) - bg_t[1:5] * bg_t[0]).abs().mean()
    return parts


def spatial_diversity(slots_raw: torch.Tensor) -> torch.Tensor:
    """惩罚所有活跃 slot 的中心过度集中（防定位坍缩的安全网）。"""
    f = squash_slots(slots_raw)
    w = f["valid"]
    wsum = w.sum().clamp(min=1e-6)
    cxm = (f["cx"] * w).sum() / wsum
    cym = (f["cy"] * w).sum() / wsum
    var = (((f["cx"] - cxm) ** 2 + (f["cy"] - cym) ** 2) * w).sum() / wsum
    return -var


def compute_losses(img_pred: torch.Tensor, img_gt: torch.Tensor,
                   slots_raw: torch.Tensor, bg_raw: torch.Tensor,
                   slots_gt, bg_gt,
                   ssim_weight: float = 0.3,
                   w_cls: float = 0.3, w_ftype: float = 0.3,
                   w_valid: float = 0.1, w_svalid: float = 0.1,
                   w_geom: float = 0.7, w_bg: float = 0.2,
                   w_div: float = 0.05, w_bbox: float = 0.5,
                   w_fill: float = 2.0, w_palette: float = 0.5,
                   palette_logits: torch.Tensor = None,
                   cls_balance: bool = True) -> tuple:
    r = render_losses(img_pred, img_gt, ssim_weight)
    a = matched_auxiliary_losses(slots_raw, bg_raw, slots_gt, bg_gt,
                                 palette_logits=palette_logits)
    aux_total = (w_cls * a["cls"] + w_ftype * a["ftype"] + w_valid * a["valid"]
                 + w_svalid * a["svalid"] + w_geom * a["geom"] + w_bg * a["bg"]
                 + w_div * spatial_diversity(slots_raw) + w_bbox * a["bbox"]
                 + w_fill * a["appearance"])
    # §11.13 色板 CE：palette_logits 由调用方经 kwargs 传入
    if "palette" in a:
        aux_total = aux_total + w_palette * a["palette"]
    total = r["total"] + aux_total
    parts = {"mae": float(r["mae"].detach()),
             "ssim": float(r["ssim"].detach()),
             "render": float(r["total"].detach()),
             "cls": float(a["cls"].detach()),
             "ftype": float(a["ftype"].detach()),
             "valid": float(a["valid"].detach()),
             "svalid": float(a["svalid"].detach()),
             "geom": float(a["geom"].detach()),
             "appearance": float(a["appearance"].detach()),
             "palette": float(a["palette"].detach()) if "palette" in a else 0.0,
             "bbox": float(a["bbox"].detach()),
             "bg": float(a["bg"].detach()),
             "div": float(spatial_diversity(slots_raw).detach()),
             "aux": float(aux_total.detach()),
             "total": float(total.detach())}
    return total, parts
