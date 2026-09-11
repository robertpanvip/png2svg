from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

from svg.scene_graph import (
    Scene, ShapeObject, BBox, Segment, PathGeom, Stop, Gradient, Fill, Stroke,
    Effects, Clip, Mask,
    FILL_NONE, FILL_SOLID, FILL_LINEAR, FILL_RADIAL,
    SHAPE_PATH,
    CMD_M, CMD_L, CMD_Q, CMD_C, CMD_A, CMD_Z,
)

# 路径子类型分布（取代旧 5 粗类；blob 不再是"类"，而是 cubic 闭合路径）
_PATH_KINDS = ("line", "poly", "quad", "cubic", "arc", "compound")


@dataclass
class GeneratorConfig:
    canvas: int = 256
    min_objects: int = 2
    max_objects: int = 6
    margin: float = 0.06
    # §11.12 A 方案（颜色减压分布）：2026-09-10 诊断证明旧分布下颜色不可提取
    # （bbox 中位面积 8.7%、open 细条多、bbox 池化 corr 仅 0.42）。收紧后重测。
    min_bbox: float = 0.18
    max_bbox: float = 0.60
    overlap_prob: float = 0.45
    alpha_prob: float = 0.25
    stroke_prob: float = 0.30
    gradient_prob: float = 0.35
    hole_prob: float = 0.18
    transparent_bg_prob: float = 0.3
    max_stops: int = 6
    # 路径子类型权重（P1 几何；A 方案：压低纯 open line，closed 形状为主）
    path_weights: dict = field(default_factory=lambda: {
        "line": 0.08, "poly": 0.18, "quad": 0.18, "cubic": 0.28, "arc": 0.13, "compound": 0.15,
    })
    # P3 stroke 样式
    stroke_attr_prob: float = 0.5
    # P4 effects
    effects_prob: float = 0.35
    # P5 composition
    group_prob: float = 0.3
    clip_prob: float = 0.18
    mask_prob: float = 0.15


def _hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(float(h) % 1.0, float(s), float(v))
    return (r, g, b)


def _catmull_rom_closed(points: np.ndarray) -> list:
    """闭合 Catmull-Rom → cubic 段序列（uv 局部坐标）。"""
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


def _catmull_rom_open(points: np.ndarray) -> list:
    """开放 Catmull-Rom → cubic 段序列（uv 局部坐标）。"""
    n = len(points)
    segs = [Segment(CMD_M, [float(points[0, 0]), float(points[0, 1])])]
    for i in range(n - 1):
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, n - 1)]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        segs.append(Segment(CMD_C, [float(c1[0]), float(c1[1]), float(c2[0]), float(c2[1]),
                                    float(p2[0]), float(p2[1])]))
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


def _rand_uv(rng, k: int, pad: float = 0.12) -> np.ndarray:
    return rng.uniform(pad, 1.0 - pad, size=(k, 2))


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
        g = _sample_gradient(rng, cfg, bbox)
        return Fill(type=g.kind, color=(0.0, 0.0, 0.0), alpha=1.0, gradient=g)
    alpha = float(rng.uniform(0.35, 1.0)) if rng.random() < cfg.alpha_prob else 1.0
    # §11.13：solid 颜色从固定色板采样（分类监督，消灭均值退路）
    from model.palette import sample_palette_rgb
    return Fill(type=FILL_SOLID, color=sample_palette_rgb(rng), alpha=alpha)


# ---- 路径几何工厂（段坐标均为 uv 局部帧 [0,1]，由 bbox 映射）----


def _make_line_path(rng, bbox: BBox, closed: bool):
    k = int(rng.integers(3, 7))
    pts = _rand_uv(rng, k)
    segs = [Segment(CMD_M, [float(pts[0, 0]), float(pts[0, 1])])]
    for i in range(1, k):
        segs.append(Segment(CMD_L, [float(pts[i, 0]), float(pts[i, 1])]))
    if closed:
        segs.append(Segment(CMD_Z, []))
    return ShapeObject(shape=SHAPE_PATH, geometry=PathGeom(bbox=bbox, segments=segs, closed=closed))


def _make_quad_path(rng, bbox: BBox, closed: bool):
    k = int(rng.integers(3, 6))
    pts = _radial_points(rng, k) if closed else _rand_uv(rng, k)
    segs = [Segment(CMD_M, [float(pts[0, 0]), float(pts[0, 1])])]
    for i in range(1, k):
        p0 = pts[i - 1]
        p1 = pts[i]
        mid = (p0 + p1) / 2.0
        ctrl = mid + rng.uniform(-0.18, 0.18, size=2)
        segs.append(Segment(CMD_Q, [float(ctrl[0]), float(ctrl[1]),
                                    float(p1[0]), float(p1[1])]))
    if closed:
        segs.append(Segment(CMD_Z, []))
    return ShapeObject(shape=SHAPE_PATH, geometry=PathGeom(bbox=bbox, segments=segs, closed=closed))


def _make_cubic_path(rng, bbox: BBox, closed: bool):
    k = int(rng.integers(5, 10))
    pts = _radial_points(rng, k) if closed else _rand_uv(rng, k, pad=0.1)
    segs = _catmull_rom_closed(pts) if closed else _catmull_rom_open(pts)
    return ShapeObject(shape=SHAPE_PATH, geometry=PathGeom(bbox=bbox, segments=segs, closed=closed))


def _make_arc_path(rng, bbox: BBox, closed: bool):
    r = float(rng.uniform(0.30, 0.46))
    if closed:
        # 完整椭圆：4 段 90° 弧
        angs = np.linspace(0.0, 2.0 * np.pi, 5)
        ring = np.stack([0.5 + r * np.cos(angs), 0.5 + r * np.sin(angs)], axis=1)
        segs = [Segment(CMD_M, [float(ring[0, 0]), float(ring[0, 1])])]
        for i in range(1, 4):
            segs.append(Segment(CMD_A, [r, r, 0.0, 0.0, 1.0,
                                        float(ring[i, 0]), float(ring[i, 1])]))
        segs.append(Segment(CMD_A, [r, r, 0.0, 0.0, 1.0,
                                    float(ring[4, 0]), float(ring[4, 1])]))
        segs.append(Segment(CMD_Z, []))
        return ShapeObject(shape=SHAPE_PATH, geometry=PathGeom(bbox=bbox, segments=segs, closed=True))
    # 开放弧
    a0 = rng.uniform(0.0, 2.0 * np.pi)
    span = float(rng.uniform(np.pi / 3, 5.0 * np.pi / 3))
    a1 = a0 + span
    p0 = (0.5 + r * np.cos(a0), 0.5 + r * np.sin(a0))
    p1 = (0.5 + r * np.cos(a1), 0.5 + r * np.sin(a1))
    large = 1.0 if span > np.pi else 0.0
    segs = [Segment(CMD_M, [float(p0[0]), float(p0[1])]),
            Segment(CMD_A, [r, r, 0.0, large, 1.0, float(p1[0]), float(p1[1])])]
    return ShapeObject(shape=SHAPE_PATH, geometry=PathGeom(bbox=bbox, segments=segs, closed=False))


def _make_compound_path(rng, bbox: BBox):
    # 外环（cubic 闭合）+ 内孔（反向 winding 的小椭圆），构成带孔闭合路径
    k = int(rng.integers(6, 9))
    outer = _catmull_rom_closed(_radial_points(rng, k))
    r = float(rng.uniform(0.18, 0.30))
    angs = np.linspace(0.0, 2.0 * np.pi, 5)
    ring = np.stack([0.5 + r * np.cos(angs), 0.5 + r * np.sin(angs)], axis=1)
    inner = [Segment(CMD_M, [float(ring[0, 0]), float(ring[0, 1])])]
    for i in range(1, 4):
        inner.append(Segment(CMD_A, [r, r, 0.0, 0.0, 0.0,
                                     float(ring[i, 0]), float(ring[i, 1])]))
    inner.append(Segment(CMD_A, [r, r, 0.0, 0.0, 0.0,
                                 float(ring[4, 0]), float(ring[4, 1])]))
    inner.append(Segment(CMD_Z, []))
    return ShapeObject(shape=SHAPE_PATH, geometry=PathGeom(bbox=bbox, segments=outer + inner, closed=True))


def _make_stroke(rng, cfg: GeneratorConfig):
    width = float(rng.uniform(0.008, 0.05))
    color = _sample_color(rng)
    alpha = float(rng.uniform(0.5, 1.0)) if rng.random() < 0.3 else 1.0
    if rng.random() < cfg.stroke_attr_prob:
        nd = int(rng.integers(1, 4))
        dash = tuple(float(x) for x in rng.uniform(0.02, 0.12, size=nd))
        linecap = rng.choice(["butt", "round", "square"])
        linejoin = rng.choice(["miter", "round", "bevel"])
    else:
        dash, linecap, linejoin = (), "butt", "miter"
    return Stroke(width=width, color=color, alpha=alpha, dash=dash,
                  linecap=linecap, linejoin=linejoin)


def _sample_effects(rng, cfg: GeneratorConfig):
    if rng.random() >= cfg.effects_prob:
        return None
    kind = rng.choice(["blur", "shadow", "glow"])
    if kind == "blur":
        return Effects(blur_radius=float(rng.uniform(0.02, 0.08)))
    if kind == "shadow":
        return Effects(shadow_dx=float(rng.uniform(-0.06, 0.06)),
                       shadow_dy=float(rng.uniform(-0.06, 0.06)),
                       shadow_blur=float(rng.uniform(0.02, 0.08)),
                       shadow_rgb=_sample_color(rng),
                       shadow_alpha=float(rng.uniform(0.3, 0.8)))
    return Effects(glow_radius=float(rng.uniform(0.03, 0.12)),
                   glow_rgb=(1.0, 1.0, 1.0),
                   glow_alpha=float(rng.uniform(0.3, 0.8)))


class SceneGenerator:
    def __init__(self, config: GeneratorConfig = None, seed: int = 0):
        self.cfg = config or GeneratorConfig()
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def sample(self) -> Scene:
        rng = self._rng
        cfg = self.cfg
        n_objects = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
        weights = np.array([cfg.path_weights.get(s, 0.0) for s in _PATH_KINDS], dtype=np.float64)
        weights = weights / weights.sum()
        placed: list = []
        objects: list = []
        for _ in range(n_objects):
            kind = _PATH_KINDS[int(rng.choice(len(_PATH_KINDS), p=weights))]
            bbox = _sample_shape_bbox(rng, cfg, placed)
            placed.append(bbox)
            if kind == "line":
                obj = _make_line_path(rng, bbox, closed=False)
            elif kind == "poly":
                obj = _make_line_path(rng, bbox, closed=True)
            elif kind == "quad":
                obj = _make_quad_path(rng, bbox, closed=rng.random() < 0.8)
            elif kind == "cubic":
                obj = _make_cubic_path(rng, bbox, closed=rng.random() < 0.85)
            elif kind == "arc":
                obj = _make_arc_path(rng, bbox, closed=rng.random() < 0.7)
            else:
                obj = _make_compound_path(rng, bbox)
            obj.fill = _sample_fill(rng, cfg, bbox)
            if rng.random() < cfg.stroke_prob:
                obj.stroke = _make_stroke(rng, cfg)
            if obj.fill.type != FILL_NONE and rng.random() < cfg.alpha_prob:
                obj.opacity = float(rng.uniform(0.55, 1.0))
            obj.effects = _sample_effects(rng, cfg)
            objects.append(obj)

        # P5 composition：分组 / 裁切 / 遮罩（引用更早 object 下标）
        for i, obj in enumerate(objects):
            if i > 0 and rng.random() < cfg.group_prob:
                obj.group = int(rng.integers(1, 4))
            if i > 0 and rng.random() < cfg.clip_prob:
                obj.clip = Clip(ref=int(rng.integers(0, i)))
            if i > 0 and rng.random() < cfg.mask_prob:
                obj.mask = Mask(ref=int(rng.integers(0, i)), kind=rng.choice(["luminance", "alpha"]))

        if rng.random() < cfg.transparent_bg_prob:
            background = None
        else:
            if rng.random() < 0.5:
                v = rng.uniform(0.12, 0.35)
                background = (float(v), float(v), float(v * rng.uniform(1.0, 1.5)), 1.0)
            else:
                background = (*_hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.0, 0.25), rng.uniform(0.85, 1.0)), 1.0)
        return Scene(width=cfg.canvas, height=cfg.canvas, background=background, objects=objects)
