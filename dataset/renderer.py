from __future__ import annotations

import math

import numpy as np
import torch
import torch.utils.checkpoint

from svg.scene_graph import (
    Scene, PathGeom, CircleGeom, EllipseGeom, RectGeom, PolygonGeom,
    FILL_NONE, FILL_SOLID, FILL_LINEAR, FILL_RADIAL,
    CMD_M, CMD_L, CMD_Q, CMD_C, CMD_Z,
)

_KAPPA = 0.5522847498
_EPS = 1e-9


def _tt(v, dtype, device):
    if torch.is_tensor(v):
        return v.to(dtype=dtype, device=device)
    return torch.tensor(v, dtype=dtype, device=device)


def _flatten_cubic(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    mt = 1.0 - t
    return (mt ** 3) * p0 + 3 * (mt ** 2) * t * p1 + 3 * mt * (t ** 2) * p2 + (t ** 3) * p3


def _flatten_quadratic(p0, p1, p2, n):
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    mt = 1.0 - t
    return (mt ** 2) * p0 + 2 * mt * t * p1 + (t ** 2) * p2


def _dedupe(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep[1:] = d > 1e-9
    return pts[keep]


def _path_subpaths(path: PathGeom, samples_per_curve: int):
    subpaths = []
    cur = None
    for seg in path.segments:
        pts = np.asarray(seg.pts, dtype=np.float64).reshape(-1, 2)
        if seg.cmd == CMD_M:
            if cur is not None:
                subpaths.append(cur)
            cur = {"pts": [pts[0]], "closed": False}
        elif seg.cmd == CMD_L:
            if cur is not None:
                cur["pts"].append(pts[0])
        elif seg.cmd == CMD_Q:
            if cur is not None:
                p0 = np.asarray(cur["pts"][-1])
                cur["pts"].extend(_flatten_quadratic(p0, pts[0], pts[1], samples_per_curve))
                cur["pts"].append(pts[1])
        elif seg.cmd == CMD_C:
            if cur is not None:
                p0 = np.asarray(cur["pts"][-1])
                cur["pts"].extend(_flatten_cubic(p0, pts[0], pts[1], pts[2], samples_per_curve))
                cur["pts"].append(pts[2])
        elif seg.cmd == CMD_Z:
            if cur is not None:
                cur["closed"] = True
                subpaths.append(cur)
                cur = None
    if cur is not None:
        subpaths.append(cur)
    out = []
    for sp in subpaths:
        arr = _dedupe(np.array(sp["pts"], dtype=np.float64))
        if len(arr) < 2:
            continue
        if sp["closed"] and np.allclose(arr[0], arr[-1], atol=1e-9):
            arr = arr[:-1]
        if len(arr) < 3 and sp["closed"]:
            continue
        out.append((arr, sp["closed"]))
    return out


def _ellipse_subpath(bbox, samples_per_curve: int):
    cx, cy = 0.5, 0.5
    rx, ry = 0.5, 0.5
    quads = [
        ((1, 0), (1, _KAPPA), (_KAPPA, 1), (0, 1)),
        ((0, 1), (-_KAPPA, 1), (-1, _KAPPA), (-1, 0)),
        ((-1, 0), (-1, -_KAPPA), (-_KAPPA, -1), (0, -1)),
        ((0, -1), (_KAPPA, -1), (1, -_KAPPA), (1, 0)),
    ]
    pts = [np.array([cx + rx, cy])]
    for p0n, p1n, p2n, p3n in quads:
        p0 = np.array([cx + p0n[0] * rx, cy + p0n[1] * ry])
        p1 = np.array([cx + p1n[0] * rx, cy + p1n[1] * ry])
        p2 = np.array([cx + p2n[0] * rx, cy + p2n[1] * ry])
        p3 = np.array([cx + p3n[0] * rx, cy + p3n[1] * ry])
        pts.extend(_flatten_cubic(p0, p1, p2, p3, samples_per_curve))
        pts.append(p3)
    return _dedupe(np.array(pts))


def _polygon_subpath(geom: PolygonGeom):
    pts = np.array(geom.points, dtype=np.float64)
    return _dedupe(np.vstack([pts, pts[:1]]))


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orient(pts: np.ndarray) -> np.ndarray:
    return pts[::-1].copy() if _shoelace(pts) < 0 else pts


def _to_abs(pts: np.ndarray, bbox) -> np.ndarray:
    return np.stack([bbox.cx + (pts[:, 0] - 0.5) * bbox.w,
                     bbox.cy + (pts[:, 1] - 0.5) * bbox.h], axis=1)


class SoftSVGRenderer:
    def __init__(self, out_size: int = 256, device: str = "cpu", dtype=torch.float32,
                 gamma_px: float = 0.005, zeta_px: float = 0.1,
                 samples_per_curve: int = 60, pad_px: float = 5.0,
                 sub_px: int = 4, chunk_points: int = 20000,
                 grad_checkpoint: bool = False):
        self.S = int(out_size)
        self.device = device
        self.dtype = dtype
        self.gamma = float(gamma_px / self.S)
        self.zeta = float(zeta_px / self.S)
        self.samples_per_curve = int(samples_per_curve)
        self.pad_px = float(pad_px)
        self.sub_px = max(1, int(sub_px))
        self.chunk_points = int(chunk_points)
        self.grad_checkpoint = bool(grad_checkpoint)

    def _prepare(self, scene: Scene):
        S = self.S
        objs = []
        for obj in scene.objects:
            g = obj.geometry
            if isinstance(g, PathGeom):
                subs = [(_orient(_to_abs(pts, g.bbox)), closed)
                        for pts, closed in _path_subpaths(g, self.samples_per_curve)]
                geo = ("poly", subs)
            elif isinstance(g, CircleGeom):
                geo = ("circle", (g.bbox.cx, g.bbox.cy, 0.5 * min(g.bbox.w, g.bbox.h)))
            elif isinstance(g, RectGeom):
                geo = ("rect", (g.bbox.cx, g.bbox.cy, g.bbox.w, g.bbox.h))
            elif isinstance(g, EllipseGeom):
                geo = ("poly", [(_orient(_to_abs(_ellipse_subpath(g, self.samples_per_curve), g.bbox)), True)])
            elif isinstance(g, PolygonGeom):
                geo = ("poly", [(_orient(_to_abs(_polygon_subpath(g), g.bbox)), True)])
            else:
                raise ValueError(f"unsupported geometry {type(g)!r}")

            fill = obj.fill
            if fill.type == FILL_NONE:
                fill_spec = None
            elif fill.type == FILL_SOLID:
                fill_spec = ("solid",
                             torch.tensor(fill.color, dtype=self.dtype, device=self.device),
                             torch.tensor(float(fill.alpha), dtype=self.dtype, device=self.device))
            elif fill.type in (FILL_LINEAR, FILL_RADIAL):
                gt = self._grad_tensors(fill.gradient)
                fill_spec = (FILL_LINEAR if fill.type == FILL_LINEAR else FILL_RADIAL,
                             gt, torch.tensor(float(fill.alpha), dtype=self.dtype, device=self.device))
            else:
                raise ValueError(f"bad fill type {fill.type!r}")

            stroke_spec = None
            if obj.stroke is not None:
                stroke_spec = (
                    torch.tensor(max(float(obj.stroke.width), 0.0) * 0.5, dtype=self.dtype, device=self.device),
                    torch.tensor(obj.stroke.color, dtype=self.dtype, device=self.device),
                    torch.tensor(float(obj.stroke.alpha), dtype=self.dtype, device=self.device),
                )

            bb = g.bbox
            hw_px = float(stroke_spec[0]) * S if stroke_spec is not None else 0.0
            pad = (self.pad_px + hw_px) / S
            if geo[0] == "poly":
                all_pts = np.vstack([p for p, _ in geo[1]])
                gx0, gy0 = all_pts[:, 0].min(), all_pts[:, 1].min()
                gx1, gy1 = all_pts[:, 0].max(), all_pts[:, 1].max()
            else:
                gx0, gy0 = bb.cx - bb.w / 2, bb.cy - bb.h / 2
                gx1, gy1 = bb.cx + bb.w / 2, bb.cy + bb.h / 2
            x0 = max(0, int(math.floor((gx0 - pad) * S)))
            y0 = max(0, int(math.floor((gy0 - pad) * S)))
            x1 = min(S, int(math.ceil((gx1 + pad) * S)))
            y1 = min(S, int(math.ceil((gy1 + pad) * S)))

            objs.append({
                "geo": geo, "fill": fill_spec, "stroke": stroke_spec,
                "opacity": torch.tensor(float(obj.opacity), dtype=self.dtype, device=self.device),
                "crop": (x0, y0, x1, y1),
            })
        return objs

    def _grad_tensors(self, g):
        return {
            "p0": torch.tensor(g.p0, dtype=self.dtype, device=self.device),
            "p1": torch.tensor(g.p1, dtype=self.dtype, device=self.device),
            "radius": torch.tensor(max(float(g.radius), 1e-6), dtype=self.dtype, device=self.device),
            "pos": torch.tensor([s.position for s in g.stops], dtype=self.dtype, device=self.device),
            "rgb": torch.tensor([s.rgb for s in g.stops], dtype=self.dtype, device=self.device),
            "alpha": torch.tensor([s.alpha for s in g.stops], dtype=self.dtype, device=self.device),
        }

    def _pixel_grid(self, x0, y0, x1, y1):
        k = self.sub_px
        Sk = self.S * k
        xs = (torch.arange(x0 * k, x1 * k, device=self.device, dtype=self.dtype) + 0.5) / Sk
        ys = (torch.arange(y0 * k, y1 * k, device=self.device, dtype=self.dtype) + 0.5) / Sk
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(-1, 2)

    def _chunks(self, n: int):
        step = self.chunk_points
        for i in range(0, n, step):
            yield i, min(i + step, n)

    def _softmin_true(self, p: torch.Tensor, pts: torch.Tensor, closed: bool = True):
        zeta = self.zeta
        a_full = pts if closed else pts[:-1]
        b_full = torch.roll(pts, -1, dims=0) if closed else pts[1:]
        a = a_full.unsqueeze(0)
        b = b_full.unsqueeze(0)
        ab = b - a
        len2 = (ab * ab).sum(-1).clamp_min(_EPS)
        out = []
        for i, j in self._chunks(p.shape[0]):
            pc = p[i:j]
            ap = pc.unsqueeze(1) - a
            t = ((ap * ab).sum(-1) / len2).clamp(0.0, 1.0)
            proj = a + ab * t.unsqueeze(-1)
            d = (pc.unsqueeze(1) - proj).norm(dim=-1)
            d_min = d.min(dim=1).values
            agg = torch.log(torch.exp(-(d - d_min.unsqueeze(1)) / zeta).clamp_min(1e-30).sum(dim=1))
            out.append(d_min - zeta * agg)
        return torch.cat(out)

    def _parity_coverage(self, p: torch.Tensor, pts_list):
        gamma = self.gamma
        outs = []
        for i, j in self._chunks(p.shape[0]):
            x, y = p[i:j, 0], p[i:j, 1]
            crossings = torch.zeros_like(x)
            for pts_src in pts_list:
                pts = pts_src if torch.is_tensor(pts_src) else torch.tensor(pts_src, dtype=p.dtype, device=p.device)
                a = pts
                b = torch.roll(pts, -1, dims=0)
                ya, yb = a[:, 1], b[:, 1]
                xa, xb = a[:, 0], b[:, 0]
                s1 = torch.sigmoid((y.unsqueeze(1) - ya.unsqueeze(0)) / gamma)
                s2 = torch.sigmoid((y.unsqueeze(1) - yb.unsqueeze(0)) / gamma)
                t_str = s1 - s2
                dy = yb - ya
                dy_safe = torch.where(dy.abs() < 1e-9, torch.full_like(dy, 1e-9), dy)
                x_int = xa.unsqueeze(0) + (y.unsqueeze(1) - ya.unsqueeze(0)) * (xb - xa).unsqueeze(0) / dy_safe.unsqueeze(0)
                K = (xb - xa).abs() / dy_safe
                delta = 3.0 * gamma * K + 2.0 / self.S
                lo = torch.minimum(xa, xb).unsqueeze(0) - delta.unsqueeze(0)
                hi = torch.maximum(xa, xb).unsqueeze(0) + delta.unsqueeze(0)
                x_int = x_int.clamp(lo, hi)
                right = torch.sigmoid((x_int - x.unsqueeze(1)) / gamma)
                crossings = crossings + (t_str * right).sum(dim=1)
            outs.append(0.5 * (1.0 - torch.cos(torch.pi * crossings.abs())))
        return torch.cat(outs)

    def _object_coverage(self, geo, p: torch.Tensor):
        kind, data = geo
        gamma = self.gamma
        if kind == "circle":
            cx, cy, r = data
            c = torch.stack([_tt(cx, p.dtype, p.device), _tt(cy, p.dtype, p.device)])
            d = (p - c).norm(dim=1) - _tt(r, p.dtype, p.device)
            return [torch.sigmoid(-d / gamma)], [d.abs()]
        if kind == "rect":
            cx, cy, w, h = data
            c = torch.stack([_tt(cx, p.dtype, p.device), _tt(cy, p.dtype, p.device)])
            half = torch.stack([_tt(w, p.dtype, p.device), _tt(h, p.dtype, p.device)]) / 2
            q = (p - c).abs() - half
            d = q.clamp_min(0.0).norm(dim=1) + q.clamp(max=0.0).max(dim=1).values
            return [torch.sigmoid(-d / gamma)], [d.abs()]
        fill_pts = [pts for pts, closed in data if closed and len(pts) >= 3]
        if fill_pts:
            fill_cov = [self._parity_coverage(p, fill_pts)]
        else:
            fill_cov = []
        dists = []
        for pts, closed in data:
            pts = pts if torch.is_tensor(pts) else torch.tensor(pts, dtype=p.dtype, device=p.device)
            dists.append(self._softmin_true(p, pts, closed=closed))
        return fill_cov, dists

    def _eval_gradient(self, spec, p: torch.Tensor):
        tensors = spec[1]
        p0, p1 = tensors["p0"], tensors["p1"]
        if spec[0] == FILL_LINEAR:
            d = p1 - p0
            denom = (d * d).sum().clamp_min(_EPS)
            t = ((p - p0) * d).sum(dim=1) / denom
        else:
            t = (p - p0).norm(dim=1) / tensors["radius"].clamp_min(_EPS)
        t = t.clamp(0.0, 1.0)
        pos, rgb, alpha = tensors["pos"], tensors["rgb"], tensors["alpha"]
        k = pos.numel()
        idx = torch.searchsorted(pos[1:-1].contiguous(), t, right=True).clamp(0, k - 2)
        t0, t1 = pos[idx], pos[idx + 1]
        w = ((t - t0) / (t1 - t0).clamp_min(_EPS)).clamp(0.0, 1.0).unsqueeze(-1)
        c = rgb[idx] * (1 - w) + rgb[idx + 1] * w
        a = alpha[idx] * (1 - w.squeeze(-1)) + alpha[idx + 1] * w.squeeze(-1)
        return c, a * spec[2]

    def render_object_layers(self, scene: Scene):
        return self._render(scene, keep_layers=True)

    def render_scene(self, scene: Scene) -> torch.Tensor:
        return self._render(scene, keep_layers=False)

    def _render(self, scene: Scene, keep_layers: bool):
        objs = self._prepare(scene)
        return self._render_objs(objs, scene.background, keep_layers=keep_layers)

    def _object_layer(self, item):
        x0, y0, x1, y1 = item["crop"]
        p = self._pixel_grid(x0, y0, x1, y1)
        fills, dists = self._object_coverage(item["geo"], p)

        fill_cov = None
        for a_sub in fills:
            fill_cov = a_sub if fill_cov is None else (fill_cov + a_sub - 2.0 * fill_cov * a_sub)

        stroke_cov = None
        if item["stroke"] is not None:
            hw = item["stroke"][0]
            for d in dists:
                band = torch.sigmoid((hw - d) / self.gamma)
                stroke_cov = band if stroke_cov is None else torch.maximum(stroke_cov, band)

        pm_rgb = None
        pm_a = None
        if fill_cov is not None and item["fill"] is not None:
            spec = item["fill"]
            if spec[0] == FILL_SOLID:
                color = spec[1].unsqueeze(0).expand(p.shape[0], 3)
                alpha = spec[2].unsqueeze(0).expand(p.shape[0])
            else:
                color, alpha = self._eval_gradient(spec, p)
            a = fill_cov * alpha
            pm_rgb = color * a.unsqueeze(-1)
            pm_a = a.unsqueeze(-1)
        if stroke_cov is not None:
            _, scolor, salpha = item["stroke"]
            a = (stroke_cov * salpha).unsqueeze(-1)
            if pm_rgb is None:
                pm_rgb = scolor.unsqueeze(0) * a
                pm_a = a
            else:
                pm_rgb = scolor.unsqueeze(0) * a + pm_rgb * (1 - a)
                pm_a = a + pm_a * (1 - a)
        if pm_rgb is None:
            rgba_p = torch.zeros(p.shape[0], 4, dtype=self.dtype, device=self.device)
        else:
            rgba_p = torch.cat([pm_rgb, pm_a], dim=1)

        k = self.sub_px
        layer = (rgba_p * item["opacity"]).reshape(y1 - y0, k, x1 - x0, k, 4)
        layer = layer.permute(0, 2, 4, 1, 3).reshape(y1 - y0, x1 - x0, 4, k * k)
        layer = layer.mean(dim=-1)
        layer = layer.permute(2, 0, 1)
        return layer

    def _render_objs(self, objs, background, keep_layers: bool = False):
        S = self.S
        if background is not None:
            bg = _tt(background, self.dtype, self.device)
            canvas = torch.cat([bg[:3].reshape(3, 1, 1).expand(3, S, S),
                                bg[3].reshape(1, 1, 1).expand(1, S, S)])
        else:
            canvas = torch.zeros(4, S, S, dtype=self.dtype, device=self.device)
        layers = []
        for item in objs:
            x0, y0, x1, y1 = item["crop"]
            if x1 <= x0 or y1 <= y0:
                continue
            if self.grad_checkpoint and torch.is_grad_enabled():
                layer = torch.utils.checkpoint.checkpoint(
                    self._object_layer, item, use_reentrant=False)
            else:
                layer = self._object_layer(item)
            if keep_layers:
                full = torch.cat([
                    torch.zeros(4, y0, S, dtype=self.dtype, device=self.device),
                    torch.cat([
                        torch.zeros(4, y1 - y0, x0, dtype=self.dtype, device=self.device),
                        layer,
                        torch.zeros(4, y1 - y0, S - x1, dtype=self.dtype, device=self.device),
                    ], dim=2),
                    torch.zeros(4, S - y1, S, dtype=self.dtype, device=self.device),
                ], dim=1)
                layers.append(full)

            pm = canvas[:, y0:y1, x0:x1]
            blended = layer + pm * (1 - layer[3:4])
            canvas = torch.cat([
                canvas[:, :y0],
                torch.cat([canvas[:, y0:y1, :x0], blended, canvas[:, y0:y1, x1:]], dim=2),
                canvas[:, y1:],
            ], dim=1)

        a_raw = canvas[3:4]
        rgb_s = (canvas[:3] / a_raw.clamp_min(1e-6)).clamp(0.0, 1.0)
        rgb_s = torch.where(a_raw > 1e-6, rgb_s, torch.zeros_like(rgb_s))
        straight = torch.cat([rgb_s, a_raw.clamp(0.0, 1.0)], dim=0)
        if keep_layers:
            return layers, straight
        return straight

    def to_uint8(self, img: torch.Tensor) -> np.ndarray:
        arr = (img.clamp(0.0, 1.0) * 255.0).add(0.5).floor().to(torch.uint8)
        return arr.permute(1, 2, 0).cpu().numpy()
