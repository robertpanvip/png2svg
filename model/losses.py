from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from model.targets import (
    NUM_SLOTS, NUM_PTS, NUM_STOPS,
    CLS_BLOB, CLS_POLYGON, CLS_STROKE,
    FILL_SOLID_I, FILL_LINEAR_I, FILL_RADIAL_I,
    I_VALID, I_CLS, I_CLOSED, I_BBOX, I_PTS, I_NPTS, I_FTYPE, I_FRGB, I_FALPHA,
    I_GP0, I_GP1, I_GRAD, I_STOPS, I_NSTOPS, I_SVALID, I_SW, I_SRGB, I_SALPHA,
    I_OPACITY, I_HOLE,
)
from model.spec import squash_slots, squash_bg
from model.matching import match_slots


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


def _geom_block(f: dict, slots_raw: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """对一组"已对齐"的 (pred, gt) slot 计算几何 masked-L1。

    约定：传入的 f / slots_raw / gt 均已按匹配对重排，且全部为 valid。
    """
    device = slots_raw.device
    cls_gt = gt[:, I_CLS].long()
    ft_gt = gt[:, I_FTYPE].long()
    n_pts_gt = gt[:, I_NPTS]
    n_stops_gt = gt[:, I_NSTOPS]
    j_pts = torch.arange(NUM_PTS, device=device)
    j_stops = torch.arange(NUM_STOPS, device=device)

    m_valid = torch.ones(gt.shape[0], dtype=torch.bool, device=device)
    m_pts_cls = m_valid & ((cls_gt == CLS_BLOB) | (cls_gt == CLS_POLYGON)
                           | (cls_gt == CLS_STROKE))
    m_pt = m_pts_cls.unsqueeze(1) & (j_pts.unsqueeze(0) < n_pts_gt.unsqueeze(1))
    geom = _masked_l1(f["pts"], gt[:, I_PTS:I_PTS + 2 * NUM_PTS]
                      .reshape(-1, NUM_PTS, 2), m_pt, slots_raw)
    geom = geom + _masked_l1(f["n_soft"], n_pts_gt, m_pts_cls, slots_raw)
    geom = geom + _masked_l1(f["closed"], gt[:, I_CLOSED], m_pts_cls, slots_raw)
    geom = geom + _masked_l1(
        torch.stack([f["cx"], f["cy"], f["w"], f["h"]], dim=1),
        gt[:, I_BBOX:I_BBOX + 4], m_valid, slots_raw)
    geom = geom + _masked_l1(f["opacity"], gt[:, I_OPACITY], m_valid, slots_raw)

    m_solid = m_valid & (ft_gt == FILL_SOLID_I)
    geom = geom + _masked_l1(f["rgb"], gt[:, I_FRGB:I_FRGB + 3], m_solid, slots_raw)
    geom = geom + _masked_l1(f["fill_alpha"], gt[:, I_FALPHA], m_solid, slots_raw)

    m_grad = m_valid & ((ft_gt == FILL_LINEAR_I) | (ft_gt == FILL_RADIAL_I))
    m_stop = m_grad.unsqueeze(1) & (j_stops.unsqueeze(0) < n_stops_gt.unsqueeze(1))
    geom = geom + _masked_l1(f["gp0"], gt[:, I_GP0:I_GP0 + 2], m_grad, slots_raw)
    geom = geom + _masked_l1(f["gp1"], gt[:, I_GP1:I_GP1 + 2], m_grad, slots_raw)
    geom = geom + _masked_l1(f["radius"], gt[:, I_GRAD], m_grad, slots_raw)
    geom = geom + _masked_l1(f["stops"], gt[:, I_STOPS:I_STOPS + NUM_STOPS * 5]
                             .reshape(-1, NUM_STOPS, 5), m_stop, slots_raw)
    geom = geom + _masked_l1(2.0 + 2.0 * torch.sigmoid(slots_raw[:, I_NSTOPS]),
                             n_stops_gt, m_grad, slots_raw)

    m_stroke = m_valid & (gt[:, I_SVALID] > 0.5)
    geom = geom + _masked_l1(f["sw"], gt[:, I_SW], m_stroke, slots_raw)
    geom = geom + _masked_l1(f["stroke_rgb"], gt[:, I_SRGB:I_SRGB + 3],
                             m_stroke, slots_raw)
    geom = geom + _masked_l1(f["stroke_alpha"], gt[:, I_SALPHA], m_stroke, slots_raw)

    m_hole = m_valid & (cls_gt == CLS_BLOB)
    geom = geom + _masked_l1(f["hole"], gt[:, I_HOLE], m_hole, slots_raw)
    return geom


def matched_auxiliary_losses(slots_raw: torch.Tensor, bg_raw: torch.Tensor, aux: dict,
                             slots_gt, bg_gt) -> dict:
    """带 Hungarian 匹配的辅助损失。

    每一步把 8 个预测 slot 与 8 个 GT 对象做最优一一匹配，再在匹配对上算
    cls/ftype/geom 监督；未匹配的预测 slot 被推向 invalid。这消除了"固定
    slot 顺序对齐 GT"与"渲染损失置换不变"之间的冲突，是定位坍缩的核心修复。
    """
    device = slots_raw.device
    gt = torch.as_tensor(np.asarray(slots_gt), dtype=torch.float32, device=device)
    bg_t = torch.as_tensor(np.asarray(bg_gt), dtype=torch.float32, device=device)
    if bg_raw.dim() == 2:
        bg_raw = bg_raw[0]
    cls_logits = aux["cls"] if aux["cls"].dim() == 3 else aux["cls"].unsqueeze(0)
    ft_logits = aux["ftype"] if aux["ftype"].dim() == 3 else aux["ftype"].unsqueeze(0)
    cls_logits = cls_logits[0]
    ft_logits = ft_logits[0]

    f = squash_slots(slots_raw)
    gv = gt[:, I_VALID]
    valid_gt = gv > 0.5
    cls_gt = gt[:, I_CLS].long()
    ft_gt = gt[:, I_FTYPE].long()

    # ---- 代价矩阵（仅 valid GT 参与匹配）----
    pred_cx = f["cx"].detach(); pred_cy = f["cy"].detach()
    pred_w = f["w"].detach(); pred_h = f["h"].detach()
    pred_cls = torch.softmax(cls_logits, -1).detach()
    n = NUM_SLOTS
    cost = np.full((n, n), 1e6, dtype=np.float64)
    for k in range(n):
        for g in range(n):
            if not bool(valid_gt[g]):
                continue
            bbox_l1 = (abs(pred_cx[k] - gt[g, I_BBOX + 0])
                       + abs(pred_cy[k] - gt[g, I_BBOX + 1])
                       + abs(pred_w[k] - gt[g, I_BBOX + 2])
                       + abs(pred_h[k] - gt[g, I_BBOX + 3])).item()
            cls_nll = -(pred_cls[k, int(cls_gt[g])].clamp(1e-4, 1).log()).item()
            cost[k, g] = bbox_l1 + cls_nll
    assign = match_slots(cost)             # assign[g] = 预测 slot 下标 或 -1

    matched_pred = [k for k in assign if k >= 0]
    parts = {}
    valid_target = torch.zeros(n, device=device)
    if matched_pred:
        pred_idx = torch.tensor(matched_pred, device=device)
        gt_idx = torch.tensor([g for g in range(n) if assign[g] >= 0], device=device)
        valid_target[pred_idx] = 1.0
        parts["cls"] = F.cross_entropy(cls_logits[pred_idx], cls_gt[gt_idx])
        parts["ftype"] = F.cross_entropy(ft_logits[pred_idx], ft_gt[gt_idx])
        f_g = {key: val[pred_idx] for key, val in f.items()}
        pred_raw_g = slots_raw[pred_idx]
        gt_g = gt[gt_idx]
        geom = _geom_block(f_g, pred_raw_g, gt_g)
        # cx/cy 专项 L1：匹配对上把中心拉向 GT 中心，直接提升 bbox 位置精度
        # （geom 已含 cx/cy/w/h 联合 L1，此处额外加权中心以抑制网格偏置残留）
        parts["bbox"] = F.l1_loss(
            torch.stack([f_g["cx"], f_g["cy"]], dim=1),
            gt_g[:, I_BBOX:I_BBOX + 2])
    else:
        parts["cls"] = slots_raw.sum() * 0.0
        parts["ftype"] = slots_raw.sum() * 0.0
        geom = slots_raw.sum() * 0.0
        parts["bbox"] = slots_raw.sum() * 0.0

    parts["valid"] = F.binary_cross_entropy(
        f["valid"].clamp(1e-6, 1.0 - 1e-6), valid_target)

    # stroke_valid BCE：在匹配对上比较
    if matched_pred:
        pred_idx = torch.tensor(matched_pred, device=device)
        gt_idx = torch.tensor([g for g in range(n) if assign[g] >= 0], device=device)
        parts["svalid"] = F.binary_cross_entropy(
            f["stroke_valid"][pred_idx].clamp(1e-6, 1.0 - 1e-6),
            gt[gt_idx, I_SVALID])
    else:
        parts["svalid"] = slots_raw.sum() * 0.0

    parts["geom"] = geom

    bg_valid = torch.sigmoid(bg_raw[0]).clamp(1e-6, 1.0 - 1e-6)
    parts["bg"] = F.binary_cross_entropy(bg_valid, bg_t[0]) \
        + (squash_bg(bg_raw) - bg_t[1:5] * bg_t[0]).abs().mean()
    return parts


def auxiliary_losses(slots_raw: torch.Tensor, bg_raw: torch.Tensor, aux: dict,
                     slots_gt, bg_gt) -> dict:
    """兼容旧接口：直接转发到匹配版。"""
    return matched_auxiliary_losses(slots_raw, bg_raw, aux, slots_gt, bg_gt)


def spatial_diversity(slots_raw: torch.Tensor) -> torch.Tensor:
    """惩罚所有活跃 slot 的中心过度集中（防定位坍缩的安全网）。

    返回负空间方差——最小化它即最大化活跃 slot 中心分布。
    """
    f = squash_slots(slots_raw)
    w = f["valid"]
    wsum = w.sum().clamp(min=1e-6)
    cxm = (f["cx"] * w).sum() / wsum
    cym = (f["cy"] * w).sum() / wsum
    var = (((f["cx"] - cxm) ** 2 + (f["cy"] - cym) ** 2) * w).sum() / wsum
    return -var


def compute_losses(img_pred: torch.Tensor, img_gt: torch.Tensor,
                   slots_raw: torch.Tensor, bg_raw: torch.Tensor, aux: dict,
                   slots_gt, bg_gt,
                   ssim_weight: float = 0.3,
                   w_cls: float = 0.3, w_ftype: float = 0.3,
                   w_valid: float = 0.1, w_svalid: float = 0.1,
                   w_geom: float = 0.7, w_bg: float = 0.2,
                   w_div: float = 0.05, w_bbox: float = 0.5) -> tuple:
    r = render_losses(img_pred, img_gt, ssim_weight)
    a = matched_auxiliary_losses(slots_raw, bg_raw, aux, slots_gt, bg_gt)
    aux_total = (w_cls * a["cls"] + w_ftype * a["ftype"] + w_valid * a["valid"]
                 + w_svalid * a["svalid"] + w_geom * a["geom"] + w_bg * a["bg"]
                 + w_div * spatial_diversity(slots_raw) + w_bbox * a["bbox"])
    total = r["total"] + aux_total
    parts = {"mae": float(r["mae"].detach()),
             "ssim": float(r["ssim"].detach()),
             "render": float(r["total"].detach()),
             "cls": float(a["cls"].detach()),
             "ftype": float(a["ftype"].detach()),
             "valid": float(a["valid"].detach()),
             "svalid": float(a["svalid"].detach()),
             "geom": float(a["geom"].detach()),
             "bbox": float(a["bbox"].detach()),
             "bg": float(a["bg"].detach()),
             "div": float(spatial_diversity(slots_raw).detach()),
             "aux": float(aux_total.detach()),
             "total": float(total.detach())}
    return total, parts
