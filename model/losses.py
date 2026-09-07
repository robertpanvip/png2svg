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


def auxiliary_losses(slots_raw: torch.Tensor, bg_raw: torch.Tensor, aux: dict,
                     slots_gt, bg_gt) -> dict:
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
    m_valid = gv > 0.5
    cls_gt = gt[:, I_CLS].long()
    ft_gt = gt[:, I_FTYPE].long()
    n_pts_gt = gt[:, I_NPTS]
    n_stops_gt = gt[:, I_NSTOPS]
    j_pts = torch.arange(NUM_PTS, device=device)
    j_stops = torch.arange(NUM_STOPS, device=device)

    parts = {}
    parts["valid"] = F.binary_cross_entropy(
        f["valid"].clamp(1e-6, 1.0 - 1e-6), gv)

    if bool(m_valid.any()):
        parts["cls"] = F.cross_entropy(cls_logits[m_valid], cls_gt[m_valid])
        parts["ftype"] = F.cross_entropy(ft_logits[m_valid], ft_gt[m_valid])
    else:
        parts["cls"] = slots_raw.sum() * 0.0
        parts["ftype"] = slots_raw.sum() * 0.0

    m_pts_cls = m_valid & ((cls_gt == CLS_BLOB) | (cls_gt == CLS_POLYGON)
                           | (cls_gt == CLS_STROKE))
    m_pt = m_pts_cls.unsqueeze(1) & (j_pts.unsqueeze(0) < n_pts_gt.unsqueeze(1))
    geom = _masked_l1(f["pts"], gt[:, I_PTS:I_PTS + 2 * NUM_PTS]
                      .reshape(NUM_SLOTS, NUM_PTS, 2), m_pt, slots_raw)
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
                             .reshape(NUM_SLOTS, NUM_STOPS, 5), m_stop, slots_raw)
    geom = geom + _masked_l1(2.0 + 2.0 * torch.sigmoid(slots_raw[:, I_NSTOPS]),
                             n_stops_gt, m_grad, slots_raw)

    parts["svalid"] = F.binary_cross_entropy(
        f["stroke_valid"].clamp(1e-6, 1.0 - 1e-6), gt[:, I_SVALID])
    m_stroke = m_valid & (gt[:, I_SVALID] > 0.5)
    geom = geom + _masked_l1(f["sw"], gt[:, I_SW], m_stroke, slots_raw)
    geom = geom + _masked_l1(f["stroke_rgb"], gt[:, I_SRGB:I_SRGB + 3],
                             m_stroke, slots_raw)
    geom = geom + _masked_l1(f["stroke_alpha"], gt[:, I_SALPHA], m_stroke, slots_raw)

    m_hole = m_valid & (cls_gt == CLS_BLOB)
    geom = geom + _masked_l1(f["hole"], gt[:, I_HOLE], m_hole, slots_raw)

    parts["geom"] = geom

    bg_valid = torch.sigmoid(bg_raw[0]).clamp(1e-6, 1.0 - 1e-6)
    parts["bg"] = F.binary_cross_entropy(bg_valid, bg_t[0]) \
        + (squash_bg(bg_raw) - bg_t[1:5] * bg_t[0]).abs().mean()
    return parts


def compute_losses(img_pred: torch.Tensor, img_gt: torch.Tensor,
                   slots_raw: torch.Tensor, bg_raw: torch.Tensor, aux: dict,
                   slots_gt, bg_gt,
                   ssim_weight: float = 0.3,
                   w_cls: float = 0.05, w_ftype: float = 0.05,
                   w_valid: float = 0.1, w_svalid: float = 0.1,
                   w_geom: float = 0.5, w_bg: float = 0.2) -> tuple:
    r = render_losses(img_pred, img_gt, ssim_weight)
    a = auxiliary_losses(slots_raw, bg_raw, aux, slots_gt, bg_gt)
    aux_total = (w_cls * a["cls"] + w_ftype * a["ftype"] + w_valid * a["valid"]
                 + w_svalid * a["svalid"] + w_geom * a["geom"] + w_bg * a["bg"])
    total = r["total"] + aux_total
    parts = {"mae": float(r["mae"].detach()),
             "ssim": float(r["ssim"].detach()),
             "render": float(r["total"].detach()),
             "cls": float(a["cls"].detach()),
             "ftype": float(a["ftype"].detach()),
             "valid": float(a["valid"].detach()),
             "svalid": float(a["svalid"].detach()),
             "geom": float(a["geom"].detach()),
             "bg": float(a["bg"].detach()),
             "aux": float(aux_total.detach()),
             "total": float(total.detach())}
    return total, parts
