from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

from svg.scene_graph import (
    Scene, ShapeObject, BBox, Segment, PathGeom, PolygonGeom, EllipseGeom, RectGeom,
    Stop, Gradient, Fill, Stroke,
    FILL_NONE, FILL_SOLID, FILL_LINEAR, FILL_RADIAL,
    SHAPE_PATH, SHAPE_ELLIPSE, SHAPE_RECT, SHAPE_POLYGON,
    CMD_M, CMD_C, CMD_L, CMD_Z,
)

_SHAPES = ("blob", "polygon", "ellipse", "rect", "stroke")


@dataclass
class GeneratorConfig:
    canvas: int = 256
    min_objects: int = 2
    max_objects: int = 6
    margin: float = 0.06
    min_bbox: float = 0.12
    max_bbox: float = 0.55
    overlap_prob: float = 0.7
    alpha_prob: float = 0.35
    stroke_prob: float = 0.25
    gradient_prob: float = 0.5
    hole_prob: float = 0.15
    transparent_bg_prob: float = 0.3
    max_stops: int = 4
    shape_weights: dict = field(default_factory=lambda: {
        "blob": 0.35, "polygon": 0.20, "ellipse": 0.15, "rect": 0.15, "stroke": 0.15,
    })


def _hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(float(h) % 1.0, float(s), float(v))
    return (r, g, b)


def _catmull_rom_to_bezier(points: np.ndarray) -> list:
    n = len(points)
    segs = [Segment(CMD_M, [float(points[0, 0]), float(points[0, 1])])]
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        segs.append(Segment(CMD_C, [float(c1[0]), float(c1[1]), float(c2[0]), float(c2[1]),
                                    float(p2[0]), float(p2[1])]))
    segs.append(Segment(CMD_Z, []))
    return segs


def _bbox_of(points: np.ndarray, pad: float = 0.05) -> BBox:
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w = max((x1 - x0) * (1.0 + pad), 1e-3)
    h = max((y1 - y0) * (1.0 + pad), 1e-3)
    return BBox(float(cx), float(cy), float(w), float(h))


def _radial_points(rng, k: int) -> np.ndarray:
    angles = (np.arange(k) / k) * 2.0 * np.pi + rng.uniform(0.0, 2.0 * np.pi / k)
    radii = rng.uniform(0.45, 1.0, size=k)
    return np.stack([0.5 + radii * np.cos(angles) * 0.5,
                     0.5 + radii * np.sin(angles) * 0.5], axis=1)


def _scaled(points: np.ndarray, factor: float) -> np.ndarray:
    return 0.5 + (points - 0.5) * factor


def _sample_shape_bbox(rng, cfg: GeneratorConfig, placed: list) -> BBox:
    for _ in range(24):
        w = rng.uniform(cfg.min_bbox, cfg.max_bbox)
        h = rng.uniform(cfg.min_bbox, cfg.max_bbox)
        cx = rng.uniform(cfg.margin + w / 2, 1.0 - cfg.margin - w / 2)
        cy = rng.uniform(cfg.margin + h / 2, 1.0 - cfg.margin - h / 2)
        bbox = BBox(float(cx), float(cy), float(w), float(h))
        if not placed:
            return bbox
        overlap_ok = all(_iou(bbox, b) <= 1e-6 for b in placed)
        if overlap_ok:
            return bbox
        if rng.random() < cfg.overlap_prob:
            return bbox
    w = rng.uniform(cfg.min_bbox, cfg.max_bbox)
    h = rng.uniform(cfg.min_bbox, cfg.max_bbox)
    return BBox(float(rng.uniform(cfg.margin + w / 2, 1.0 - cfg.margin - w / 2)),
                float(rng.uniform(cfg.margin + h / 2, 1.0 - cfg.margin - h / 2)),
                float(w), float(h))


def _iou(a: BBox, b: BBox) -> float:
    ax0, ay0 = a.cx - a.w / 2, a.cy - a.h / 2
    ax1, ay1 = a.cx + a.w / 2, a.cy + a.h / 2
    bx0, by0 = b.cx - b.w / 2, b.cy - b.h / 2
    bx1, by1 = b.cx + b.w / 2, b.cy + b.h / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(union, 1e-9)


def _sample_color(rng, vmin: float = 0.25, vmax: float = 1.0) -> tuple:
    return _hsv_to_rgb(rng.uniform(0.0, 1.0), rng.uniform(0.35, 1.0),
                       rng.uniform(vmin, vmax))


def _sample_gradient(rng, cfg: GeneratorConfig, bbox: BBox) -> Gradient:
    kind = FILL_LINEAR if rng.random() < 0.55 else FILL_RADIAL
    n_stops = int(rng.integers(2, cfg.max_stops + 1))
    base_hue = rng.uniform(0.0, 1.0)
    positions = np.sort(rng.uniform(0.0, 1.0, size=n_stops))
    positions[0] = 0.0
    positions[-1] = 1.0
    stops = []
    for i, p in enumerate(positions):
        hue = (base_hue + rng.uniform(-0.18, 0.18)) % 1.0
        rgb = _hsv_to_rgb(hue, rng.uniform(0.3, 1.0), rng.uniform(0.35, 1.0))
        alpha = float(rng.uniform(0.55, 1.0)) if rng.random() < 0.3 else 1.0
        stops.append(Stop(float(p), rgb, alpha))
    spread = rng.uniform(0.8, 1.6)
    if kind == FILL_LINEAR:
        theta = rng.uniform(0.0, 2.0 * np.pi)
        dx, dy = np.cos(theta) * 0.5 * spread, np.sin(theta) * 0.5 * spread
        p0 = (float(bbox.cx - dx), float(bbox.cy - dy))
        p1 = (float(bbox.cx + dx), float(bbox.cy + dy))
        return Gradient(kind=kind, p0=p0, p1=p1, stops=stops)
    p0 = (float(bbox.cx + rng.uniform(-0.1, 0.1)), float(bbox.cy + rng.uniform(-0.1, 0.1)))
    p1 = (float(bbox.cx + rng.uniform(-0.1, 0.1)), float(bbox.cy + rng.uniform(-0.1, 0.1)))
    radius = float(0.5 * max(bbox.w, bbox.h) * spread)
    return Gradient(kind=kind, p0=p0, p1=p1, radius=radius, stops=stops)


def _sample_fill(rng, cfg: GeneratorConfig, bbox: BBox) -> Fill:
    if rng.random() < cfg.gradient_prob:
        return _gradient_fill(rng, cfg, bbox)
    alpha = float(rng.uniform(0.35, 1.0)) if rng.random() < cfg.alpha_prob else 1.0
    return Fill(type=FILL_SOLID, color=_sample_color(rng), alpha=alpha)


def _gradient_fill(rng, cfg: GeneratorConfig, bbox: BBox) -> Fill:
    g = _sample_gradient(rng, cfg, bbox)
    return Fill(type=g.kind, color=(0.0, 0.0, 0.0), alpha=1.0, gradient=g)


def _make_blob(rng, cfg: GeneratorConfig, bbox: BBox, hole: bool):
    k = int(rng.integers(6, 11))
    pts = _radial_points(rng, k)
    segs = _catmull_rom_to_bezier(pts)
    if hole:
        inner = _catmull_rom_to_bezier(_scaled(pts, rng.uniform(0.35, 0.55)))
        segs.extend(inner)
    geom = PathGeom(bbox=bbox, segments=segs, closed=True)
    return ShapeObject(shape=SHAPE_PATH, geometry=geom)


def _make_polygon(rng, cfg: GeneratorConfig, bbox: BBox):
    k = int(rng.integers(5, 9))
    pts = _radial_points(rng, k)
    poly = PolygonGeom(bbox=bbox, points=[(float(u), float(v)) for u, v in pts])
    return ShapeObject(shape=SHAPE_POLYGON, geometry=poly)


def _make_ellipse(rng, cfg: GeneratorConfig, bbox: BBox):
    return ShapeObject(shape=SHAPE_ELLIPSE, geometry=EllipseGeom(bbox=bbox))


def _make_rect(rng, cfg: GeneratorConfig, bbox: BBox):
    if rng.random() < 0.5:
        theta = rng.uniform(0.0, np.pi)
        ca, sa = np.cos(theta) * 0.5, np.sin(theta) * 0.5
        corners = np.array([
            [-ca - sa, -sa + ca],
            [ca - sa, sa + ca],
            [ca + sa, sa - ca],
            [-ca + sa, -sa - ca],
        ])
        pts = np.stack([0.5 + corners[:, 0], 0.5 + corners[:, 1]], axis=1)
        poly = PolygonGeom(bbox=bbox, points=[(float(u), float(v)) for u, v in pts])
        return ShapeObject(shape=SHAPE_POLYGON, geometry=poly)
    return ShapeObject(shape=SHAPE_RECT, geometry=RectGeom(bbox=bbox))


def _make_stroke_path(rng, cfg: GeneratorConfig, bbox: BBox):
    n = int(rng.integers(4, 9))
    kind = rng.random()
    if kind < 0.5:
        t = np.linspace(0.0, 1.0, n)
        amp = rng.uniform(0.25, 0.45)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq = rng.uniform(1.0, 2.5) * np.pi
        u = t
        v = 0.5 + amp * np.sin(freq * t + phase) * (1.0 if rng.random() < 0.5 else t * 0.5 + 0.5)
        pts = np.stack([u, v], axis=1)
        segs = [Segment(CMD_M, [float(pts[0, 0]), float(pts[0, 1])])]
        for i in range(1, n):
            segs.append(Segment(CMD_L, [float(pts[i, 0]), float(pts[i, 1])]))
    else:
        turns = rng.uniform(1.0, 2.2) * 2.0 * np.pi
        t = np.linspace(0.0, 1.0, n)
        r = np.linspace(0.15, 0.5, n)
        ang = turns * t + rng.uniform(0.0, 2.0 * np.pi)
        pts = np.stack([0.5 + r * np.cos(ang), 0.5 + r * np.sin(ang)], axis=1)
        segs = [Segment(CMD_M, [float(pts[0, 0]), float(pts[0, 1])])]
        for i in range(1, n):
            segs.append(Segment(CMD_L, [float(pts[i, 0]), float(pts[i, 1])]))
    geom = PathGeom(bbox=bbox, segments=segs, closed=False)
    color = _sample_color(rng)
    return ShapeObject(
        shape=SHAPE_PATH, geometry=geom,
        fill=Fill(type=FILL_NONE),
        stroke=Stroke(width=float(rng.uniform(0.015, 0.05)), color=color,
                      alpha=float(rng.uniform(0.5, 1.0)) if rng.random() < 0.3 else 1.0),
    )


class SceneGenerator:
    def __init__(self, config: GeneratorConfig = None, seed: int = 0):
        self.cfg = config or GeneratorConfig()
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def sample(self) -> Scene:
        rng = self._rng
        cfg = self.cfg
        n_objects = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
        weights = np.array([cfg.shape_weights.get(s, 0.0) for s in _SHAPES], dtype=np.float64)
        weights = weights / weights.sum()
        placed: list = []
        objects: list = []
        for _ in range(n_objects):
            shape = _SHAPES[int(rng.choice(len(_SHAPES), p=weights))]
            bbox = _sample_shape_bbox(rng, cfg, placed)
            placed.append(bbox)
            if shape == "blob":
                hole = rng.random() < cfg.hole_prob
                obj = _make_blob(rng, cfg, bbox, hole)
                obj.fill = _sample_fill(rng, cfg, bbox)
                if rng.random() < cfg.stroke_prob and not hole:
                    obj.stroke = Stroke(width=float(rng.uniform(0.008, 0.03)),
                                        color=_sample_color(rng), alpha=1.0)
            elif shape == "polygon":
                obj = _make_polygon(rng, cfg, bbox)
                obj.fill = _sample_fill(rng, cfg, bbox)
                if rng.random() < cfg.stroke_prob:
                    obj.stroke = Stroke(width=float(rng.uniform(0.008, 0.03)),
                                        color=_sample_color(rng), alpha=1.0)
            elif shape == "ellipse":
                obj = _make_ellipse(rng, cfg, bbox)
                obj.fill = _sample_fill(rng, cfg, bbox)
                if rng.random() < cfg.stroke_prob:
                    obj.stroke = Stroke(width=float(rng.uniform(0.008, 0.03)),
                                        color=_sample_color(rng), alpha=1.0)
            elif shape == "rect":
                obj = _make_rect(rng, cfg, bbox)
                obj.fill = _sample_fill(rng, cfg, bbox)
                if rng.random() < cfg.stroke_prob:
                    obj.stroke = Stroke(width=float(rng.uniform(0.008, 0.03)),
                                        color=_sample_color(rng), alpha=1.0)
            else:
                obj = _make_stroke_path(rng, cfg, bbox)
            if obj.fill.type != FILL_NONE and obj.fill.type != FILL_SOLID and rng.random() < cfg.alpha_prob:
                obj.opacity = float(rng.uniform(0.55, 1.0))
            objects.append(obj)
        if rng.random() < cfg.transparent_bg_prob:
            background = None
        else:
            if rng.random() < 0.5:
                v = rng.uniform(0.12, 0.35)
                background = (float(v), float(v), float(v * rng.uniform(1.0, 1.5)), 1.0)
            else:
                background = (*_hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.0, 0.25), rng.uniform(0.85, 1.0)), 1.0)
        return Scene(width=cfg.canvas, height=cfg.canvas, background=background, objects=objects)
