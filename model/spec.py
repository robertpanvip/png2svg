from __future__ import annotations

import math

import numpy as np
import torch

from model.targets import (
    NUM_SLOTS, NUM_PTS, NUM_STOPS, NUM_CLS, BG_DIM,
    CLS_BLOB, CLS_POLYGON, CLS_ELLIPSE, CLS_RECT, CLS_STROKE,
    FILL_NONE_I, FILL_SOLID_I, FILL_LINEAR_I, FILL_RADIAL_I,
    I_VALID, I_CLS, I_CLOSED, I_BBOX, I_PTS, I_NPTS, I_FTYPE, I_FRGB, I_FALPHA,
    I_GP0, I_GP1, I_GRAD, I_STOPS, I_NSTOPS, I_SVALID, I_SW, I_SRGB, I_SALPHA,
    I_OPACITY, I_HOLE,
    _slot_template,
)
from svg.scene_graph import FILL_LINEAR, FILL_RADIAL
from dataset.renderer import _ellipse_subpath

C_MIN, C_MAX = -0.20, 1.20
W_MIN, W_MAX = 0.02, 1.30
G_MIN, G_MAX = -1.00, 2.00
R_MIN, R_MAX = 0.02, 1.50
SW_MIN, SW_MAX = 0.002, 0.25
HOLE_MAX = 0.95
HOLE_MIN = 0.02
VALID_SKIP = 0.05
SOFT_N_K = 12.0
CURVE_SAMPLES = 60
ELLIPSE_SAMPLES = 60

_ELLIPSE_UV = torch.tensor(
    np.asarray(_ellipse_subpath(None, ELLIPSE_SAMPLES), dtype=np.float64),
    dtype=torch.float64)


def _lin(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def squash_slots(slots_raw: torch.Tensor) -> dict:
    v = slots_raw
    shp = v.shape[:-1]
    return {
        "valid": torch.sigmoid(v[..., I_VALID]),
        "closed": torch.sigmoid(v[..., I_CLOSED]),
        "cx": _lin(v[..., I_BBOX + 0], C_MIN, C_MAX),
        "cy": _lin(v[..., I_BBOX + 1], C_MIN, C_MAX),
        "w": _lin(v[..., I_BBOX + 2], W_MIN, W_MAX),
        "h": _lin(v[..., I_BBOX + 3], W_MIN, W_MAX),
        "pts": torch.sigmoid(v[..., I_PTS:I_PTS + 2 * NUM_PTS]).reshape(*shp, NUM_PTS, 2),
        "n_soft": 2.0 + (NUM_PTS - 2.0) * torch.sigmoid(v[..., I_NPTS]),
        "rgb": torch.sigmoid(v[..., I_FRGB:I_FRGB + 3]),
        "fill_alpha": torch.sigmoid(v[..., I_FALPHA]),
        "gp0": _lin(v[..., I_GP0:I_GP0 + 2], G_MIN, G_MAX),
        "gp1": _lin(v[..., I_GP1:I_GP1 + 2], G_MIN, G_MAX),
        "radius": _lin(v[..., I_GRAD], R_MIN, R_MAX),
        "stops": torch.sigmoid(v[..., I_STOPS:I_STOPS + NUM_STOPS * 5]).reshape(*shp, NUM_STOPS, 5),
        "stroke_valid": torch.sigmoid(v[..., I_SVALID]),
        "sw": _lin(v[..., I_SW], SW_MIN, SW_MAX),
        "stroke_rgb": torch.sigmoid(v[..., I_SRGB:I_SRGB + 3]),
        "stroke_alpha": torch.sigmoid(v[..., I_SALPHA]),
        "opacity": torch.sigmoid(v[..., I_OPACITY]),
        "hole": HOLE_MAX * torch.sigmoid(v[..., I_HOLE]),
    }


def squash_bg(bg_raw: torch.Tensor) -> torch.Tensor:
    valid = torch.sigmoid(bg_raw[..., 0:1])
    return torch.sigmoid(bg_raw[..., 1:5]) * valid


def _collapse_pts(pts: torch.Tensor, n_soft: torch.Tensor) -> torch.Tensor:
    n = pts.shape[-2]
    j = torch.arange(n, device=pts.device, dtype=pts.dtype)
    m = torch.sigmoid(SOFT_N_K * (n_soft.unsqueeze(-1) - j - 0.5))
    prev = pts[..., 0, :]
    outs = [prev]
    for s in range(1, n):
        prev = prev + m[..., s].unsqueeze(-1) * (pts[..., s, :] - prev)
        outs.append(prev)
    return torch.stack(outs, dim=-2)


def _flatten_ring(ctrl: torch.Tensor, samples: int) -> torch.Tensor:
    n = ctrl.shape[0]
    idx = torch.arange(n, device=ctrl.device)
    p0 = ctrl[(idx - 1) % n]
    p1 = ctrl[idx]
    p2 = ctrl[(idx + 1) % n]
    p3 = ctrl[(idx + 2) % n]
    c1 = p1 + (p2 - p0) / 6.0
    c2 = p2 - (p3 - p1) / 6.0
    t = torch.linspace(0.0, 1.0, samples + 1, device=ctrl.device,
                       dtype=ctrl.dtype)[:-1].view(1, samples, 1)
    mt = 1.0 - t
    b = ((mt ** 3) * p1.unsqueeze(1) + (3 * mt ** 2 * t) * c1.unsqueeze(1)
         + (3 * mt * t ** 2) * c2.unsqueeze(1) + (t ** 3) * p2.unsqueeze(1))
    return b.reshape(n * samples, 2)


def _to_abs(uv: torch.Tensor, cx, cy, w, h) -> torch.Tensor:
    return torch.stack([cx + (uv[..., 0] - 0.5) * w,
                        cy + (uv[..., 1] - 0.5) * h], dim=-1)


def slots_to_objs(slots_raw: torch.Tensor, cls_ids, ftype_ids, bg_raw: torch.Tensor,
                  canvas: int, pad_px: float, curve_samples: int = CURVE_SAMPLES):
    f = squash_slots(slots_raw)
    objs = []
    for k in range(NUM_SLOTS):
        valid = f["valid"][k]
        if float(valid.detach()) < VALID_SKIP:
            continue
        cls = int(cls_ids[k]) % NUM_CLS
        ftype = int(ftype_ids[k]) % 4
        cx, cy, w, h = f["cx"][k], f["cy"][k], f["w"][k], f["h"][k]

        hw = 0.5 * f["sw"][k]
        stroke_spec = (hw, f["stroke_rgb"][k],
                       f["stroke_alpha"][k] * f["stroke_valid"][k] * valid)

        if cls == CLS_ELLIPSE:
            uv = _ELLIPSE_UV.to(dtype=slots_raw.dtype, device=slots_raw.device)
            geo = ("poly", [(_to_abs(uv, cx, cy, w, h), True)])
        elif cls == CLS_RECT:
            geo = ("rect", (cx, cy, w, h))
        else:
            ctrl_uv = _collapse_pts(f["pts"][k], f["n_soft"][k])
            ctrl = _to_abs(ctrl_uv, cx, cy, w, h)
            if cls == CLS_BLOB:
                subs = [(_flatten_ring(ctrl, curve_samples), True)]
                hole = f["hole"][k]
                if float(hole.detach()) > HOLE_MIN:
                    inner_uv = 0.5 + (ctrl_uv - 0.5) * hole
                    inner = _to_abs(inner_uv, cx, cy, w, h)
                    subs.append((_flatten_ring(inner, curve_samples), True))
                geo = ("poly", subs)
            elif cls == CLS_POLYGON:
                geo = ("poly", [(ctrl, True)])
            else:
                geo = ("poly", [(ctrl, False)])

        fill_spec = None
        if cls != CLS_STROKE and ftype != FILL_NONE_I:
            gate = f["fill_alpha"][k] * valid
            if ftype == FILL_SOLID_I:
                fill_spec = ("solid", f["rgb"][k], gate)
            else:
                n_s = int(round(float(torch.sigmoid(slots_raw[k, I_NSTOPS].detach())) * 2.0 + 2.0))
                n_s = max(2, min(NUM_STOPS, n_s))
                st_used = f["stops"][k][:n_s]
                pos, order = torch.sort(st_used[:, 0])
                grad = {
                    "p0": f["gp0"][k], "p1": f["gp1"][k], "radius": f["radius"][k],
                    "pos": pos, "rgb": st_used[order, 1:4], "alpha": st_used[order, 4],
                }
                kind = FILL_LINEAR if ftype == FILL_LINEAR_I else FILL_RADIAL
                fill_spec = (kind, grad, gate)

        opacity = f["opacity"][k] * valid
        hw_px = float(hw.detach()) * canvas
        pad = (pad_px + hw_px) / canvas
        if geo[0] == "rect":
            fcx, fcy, fw, fh = (float(cx.detach()), float(cy.detach()),
                                float(w.detach()), float(h.detach()))
            gx0, gy0, gx1, gy1 = fcx - fw / 2, fcy - fh / 2, fcx + fw / 2, fcy + fh / 2
        else:
            allp = torch.cat([p for p, _ in geo[1]], dim=0).detach()
            gx0, gy0 = float(allp[:, 0].min()), float(allp[:, 1].min())
            gx1, gy1 = float(allp[:, 0].max()), float(allp[:, 1].max())
        x0 = max(0, int(math.floor((gx0 - pad) * canvas)))
        y0 = max(0, int(math.floor((gy0 - pad) * canvas)))
        x1 = min(canvas, int(math.ceil((gx1 + pad) * canvas)))
        y1 = min(canvas, int(math.ceil((gy1 + pad) * canvas)))
        objs.append({"geo": geo, "fill": fill_spec, "stroke": stroke_spec,
                     "opacity": opacity, "crop": (x0, y0, x1, y1)})
    return objs, squash_bg(bg_raw)


def _lg(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def targets_to_raw(slots: np.ndarray) -> torch.Tensor:
    v = np.asarray(slots, dtype=np.float64)
    raw = np.zeros_like(v)

    def inv(x, lo, hi):
        return _lg((x - lo) / (hi - lo))

    raw[:, I_VALID] = _lg(v[:, I_VALID])
    raw[:, I_CLS] = v[:, I_CLS]
    raw[:, I_CLOSED] = _lg(v[:, I_CLOSED])
    raw[:, I_BBOX + 0] = inv(v[:, I_BBOX + 0], C_MIN, C_MAX)
    raw[:, I_BBOX + 1] = inv(v[:, I_BBOX + 1], C_MIN, C_MAX)
    raw[:, I_BBOX + 2] = inv(v[:, I_BBOX + 2], W_MIN, W_MAX)
    raw[:, I_BBOX + 3] = inv(v[:, I_BBOX + 3], W_MIN, W_MAX)
    raw[:, I_PTS:I_PTS + 2 * NUM_PTS] = _lg(v[:, I_PTS:I_PTS + 2 * NUM_PTS])
    raw[:, I_NPTS] = _lg((v[:, I_NPTS] - 2.0) / (NUM_PTS - 2.0))
    raw[:, I_FTYPE] = v[:, I_FTYPE]
    raw[:, I_FRGB:I_FRGB + 3] = _lg(v[:, I_FRGB:I_FRGB + 3])
    raw[:, I_FALPHA] = _lg(v[:, I_FALPHA])
    raw[:, I_GP0:I_GP0 + 2] = inv(v[:, I_GP0:I_GP0 + 2], G_MIN, G_MAX)
    raw[:, I_GP1:I_GP1 + 2] = inv(v[:, I_GP1:I_GP1 + 2], G_MIN, G_MAX)
    raw[:, I_GRAD] = inv(v[:, I_GRAD], R_MIN, R_MAX)
    raw[:, I_STOPS:I_STOPS + NUM_STOPS * 5] = _lg(v[:, I_STOPS:I_STOPS + NUM_STOPS * 5])
    raw[:, I_NSTOPS] = _lg((v[:, I_NSTOPS] - 2.0) / 2.0)
    raw[:, I_SVALID] = _lg(v[:, I_SVALID])
    raw[:, I_SW] = inv(v[:, I_SW], SW_MIN, SW_MAX)
    raw[:, I_SRGB:I_SRGB + 3] = _lg(v[:, I_SRGB:I_SRGB + 3])
    raw[:, I_SALPHA] = _lg(v[:, I_SALPHA])
    raw[:, I_OPACITY] = _lg(v[:, I_OPACITY])
    raw[:, I_HOLE] = _lg(v[:, I_HOLE] / HOLE_MAX)
    return torch.tensor(raw, dtype=torch.float32)


def bg_to_raw(bg: np.ndarray) -> torch.Tensor:
    bg = np.asarray(bg, dtype=np.float64)
    out = np.zeros(BG_DIM, dtype=np.float64)
    out[0] = _lg(bg[0])
    out[1:5] = _lg(bg[1:5])
    return torch.tensor(out, dtype=torch.float32)


def predictions_to_targets(slots_raw: torch.Tensor, cls_ids, ftype_ids,
                           bg_raw: torch.Tensor):
    f = squash_slots(slots_raw)
    slots = np.stack([_slot_template() for _ in range(NUM_SLOTS)])
    fdet = {key: val.detach().cpu().numpy() for key, val in f.items()}
    for k in range(NUM_SLOTS):
        if fdet["valid"][k] < 0.5:
            continue
        v = slots[k]
        v[I_VALID] = 1.0
        v[I_CLS] = int(cls_ids[k]) % NUM_CLS
        v[I_CLOSED] = 1.0 if fdet["closed"][k] > 0.5 else 0.0
        v[I_BBOX:I_BBOX + 4] = (fdet["cx"][k], fdet["cy"][k], fdet["w"][k], fdet["h"][k])
        n = int(np.clip(round(float(fdet["n_soft"][k])), 2, NUM_PTS))
        pts = fdet["pts"][k].reshape(NUM_PTS, 2)
        v[I_PTS:I_PTS + 2 * n] = pts[:n].reshape(-1)
        v[I_NPTS] = float(n)
        ft = int(ftype_ids[k]) % 4
        if ft == FILL_NONE_I or int(cls_ids[k]) % NUM_CLS == CLS_STROKE:
            v[I_FTYPE] = FILL_NONE_I
            v[I_FALPHA] = 0.0
        elif ft == FILL_SOLID_I:
            v[I_FTYPE] = FILL_SOLID_I
            v[I_FRGB:I_FRGB + 3] = fdet["rgb"][k]
            v[I_FALPHA] = float(fdet["fill_alpha"][k])
        else:
            v[I_FTYPE] = ft
            st = fdet["stops"][k]
            n_s = int(round(float(torch.sigmoid(slots_raw[k, I_NSTOPS].detach())) * 2.0 + 2.0))
            n_s = max(2, min(NUM_STOPS, n_s))
            v[I_GP0:I_GP0 + 2] = fdet["gp0"][k]
            v[I_GP1:I_GP1 + 2] = fdet["gp1"][k]
            v[I_GRAD] = float(fdet["radius"][k])
            order = np.argsort(st[:n_s, 0])
            for j in range(n_s):
                jj = int(order[j])
                base = I_STOPS + j * 5
                v[base:base + 5] = (float(st[jj, 0]), st[jj, 1], st[jj, 2], st[jj, 3], st[jj, 4])
            v[I_NSTOPS] = float(n_s)
        if fdet["stroke_valid"][k] > 0.5:
            v[I_SVALID] = 1.0
            v[I_SW] = float(fdet["sw"][k])
            v[I_SRGB:I_SRGB + 3] = fdet["stroke_rgb"][k]
            v[I_SALPHA] = float(fdet["stroke_alpha"][k])
        v[I_OPACITY] = float(fdet["opacity"][k])
        v[I_HOLE] = float(fdet["hole"][k])
    v_bg = float(torch.sigmoid(bg_raw[0].detach()))
    rgba = torch.sigmoid(bg_raw[1:5]).detach().cpu().numpy()
    if v_bg > 0.5:
        bg = np.concatenate([[1.0], rgba]).astype(np.float32)
    else:
        bg = np.zeros(BG_DIM, dtype=np.float32)
    return slots.astype(np.float32), bg
