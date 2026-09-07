from __future__ import annotations

import numpy as np

from svg.scene_graph import (
    Scene, ShapeObject, BBox, Segment, PathGeom, PolygonGeom, EllipseGeom, RectGeom,
    Stop, Gradient, Fill, Stroke,
    FILL_NONE, FILL_SOLID, FILL_LINEAR, FILL_RADIAL,
    SHAPE_PATH, SHAPE_ELLIPSE, SHAPE_RECT, SHAPE_POLYGON,
    CMD_M, CMD_L, CMD_C, CMD_Z,
)

NUM_SLOTS = 8
NUM_PTS = 10
NUM_STOPS = 4
NUM_CLS = 5
CLS_BLOB, CLS_POLYGON, CLS_ELLIPSE, CLS_RECT, CLS_STROKE = range(NUM_CLS)
FILL_NONE_I, FILL_SOLID_I, FILL_LINEAR_I, FILL_RADIAL_I = range(4)

SLOT_DIM = 67
BG_DIM = 5

I_VALID = 0
I_CLS = 1
I_CLOSED = 2
I_BBOX = 3
I_PTS = 7
I_NPTS = 27
I_FTYPE = 28
I_FRGB = 29
I_FALPHA = 32
I_GP0 = 33
I_GP1 = 35
I_GRAD = 37
I_STOPS = 38
I_NSTOPS = 58
I_SVALID = 59
I_SW = 60
I_SRGB = 61
I_SALPHA = 64
I_OPACITY = 65
I_HOLE = 66


def _slot_template() -> np.ndarray:
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    v[I_FALPHA] = 1.0
    v[I_OPACITY] = 1.0
    v[I_GRAD] = 0.5
    v[I_SW] = 0.02
    return v


def _path_control_points(segments: list) -> np.ndarray:
    pts = []
    for seg in segments:
        if seg.cmd == CMD_M:
            pts.append(seg.pts[0:2])
        elif seg.cmd == CMD_L:
            pts.append(seg.pts[0:2])
        elif seg.cmd == CMD_C:
            pts.append(seg.pts[4:6])
    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) > 1 and np.allclose(arr[0], arr[-1], atol=1e-9):
        arr = arr[:-1]
    return arr


def _hole_scale(outer: np.ndarray, inner: np.ndarray) -> float:
    d = outer - 0.5
    mask = np.abs(d) > 1e-6
    if not mask.any():
        return 0.0
    ratios = (inner[mask] - 0.5) / d[mask]
    return float(np.clip(np.median(ratios), 0.05, 0.95))


def encode_scene(scene: Scene) -> tuple:
    K, M = NUM_SLOTS, NUM_PTS
    slots = np.stack([_slot_template() for _ in range(K)])
    objs = list(scene.objects)
    objs.sort(key=lambda o: (-(o.geometry.bbox.w * o.geometry.bbox.h),
                             o.geometry.bbox.cy, o.geometry.bbox.cx))
    for i, obj in enumerate(objs[:K]):
        v = slots[i]
        g = obj.geometry
        bb = g.bbox
        v[I_VALID] = 1.0
        v[I_BBOX:I_BBOX + 4] = (bb.cx, bb.cy, bb.w, bb.h)
        if isinstance(g, PolygonGeom):
            v[I_CLS] = CLS_POLYGON
            raw = np.asarray(g.points, dtype=np.float64)
            abs_pts = np.column_stack([g.bbox.cx + (raw[:, 0] - 0.5) * g.bbox.w,
                                       g.bbox.cy + (raw[:, 1] - 0.5) * g.bbox.h])
            x0, y0 = abs_pts.min(axis=0)
            x1, y1 = abs_pts.max(axis=0)
            nb = BBox(float((x0 + x1) / 2.0), float((y0 + y1) / 2.0),
                      max(float(x1 - x0), 1e-3), max(float(y1 - y0), 1e-3))
            v[I_BBOX:I_BBOX + 4] = (nb.cx, nb.cy, nb.w, nb.h)
            pts = (abs_pts - np.array([nb.cx, nb.cy])) / np.array([nb.w, nb.h]) + 0.5
        elif isinstance(g, EllipseGeom):
            v[I_CLS] = CLS_ELLIPSE
            pts = np.zeros((0, 2))
        elif isinstance(g, RectGeom):
            v[I_CLS] = CLS_RECT
            pts = np.zeros((0, 2))
        elif isinstance(g, PathGeom):
            subs = []
            cur = None
            for seg in g.segments:
                if seg.cmd == CMD_M:
                    cur = []
                    subs.append(cur)
                elif seg.cmd in (CMD_L, CMD_C):
                    if cur is not None:
                        cur.append(seg)
            fill_none = obj.fill.type == FILL_NONE
            if fill_none and obj.stroke is not None:
                v[I_CLS] = CLS_STROKE
                v[I_CLOSED] = 0.0
                pts = _path_control_points(subs[0])
            else:
                v[I_CLS] = CLS_BLOB
                v[I_CLOSED] = 1.0
                pts = _path_control_points(subs[0])
                if len(subs) > 1:
                    inner_pts = _path_control_points(subs[1])
                    if len(pts) == len(inner_pts):
                        v[I_HOLE] = _hole_scale(pts, inner_pts)
        else:
            continue
        n = min(len(pts), M)
        if n > 0:
            v[I_PTS:I_PTS + 2 * n] = pts[:n].reshape(-1)
        v[I_NPTS] = float(n)

        f = obj.fill
        if f.type == FILL_SOLID:
            v[I_FTYPE] = FILL_SOLID_I
            v[I_FRGB:I_FRGB + 3] = f.color
            v[I_FALPHA] = f.alpha
        elif f.type in (FILL_LINEAR, FILL_RADIAL):
            v[I_FTYPE] = FILL_LINEAR_I if f.type == FILL_LINEAR else FILL_RADIAL_I
            grad = f.gradient
            v[I_GP0:I_GP0 + 2] = grad.p0
            v[I_GP1:I_GP1 + 2] = grad.p1
            v[I_GRAD] = grad.radius if grad.radius > 0 else 0.5
            stops = sorted(grad.stops, key=lambda s: s.position)[:NUM_STOPS]
            for j, s in enumerate(stops):
                base = I_STOPS + j * 5
                v[base:base + 5] = (s.position, *s.rgb, s.alpha)
            v[I_NSTOPS] = float(len(stops))
        else:
            v[I_FTYPE] = FILL_NONE_I
            v[I_FALPHA] = 0.0

        if obj.stroke is not None:
            v[I_SVALID] = 1.0
            v[I_SW] = obj.stroke.width
            v[I_SRGB:I_SRGB + 3] = obj.stroke.color
            v[I_SALPHA] = obj.stroke.alpha
        v[I_OPACITY] = obj.opacity

    if scene.background is not None:
        bg = np.array([1.0, *scene.background], dtype=np.float32)
    else:
        bg = np.zeros(BG_DIM, dtype=np.float32)
    return slots.astype(np.float32), bg


def decode_scene(slots: np.ndarray, bg: np.ndarray, canvas: int = 256) -> Scene:
    slots = np.asarray(slots, dtype=np.float64)
    objects = []
    for i in range(min(NUM_SLOTS, slots.shape[0])):
        v = slots[i]
        if v[I_VALID] < 0.5:
            continue
        cls = int(round(float(v[I_CLS]))) % NUM_CLS
        cx, cy, w, h = (float(x) for x in v[I_BBOX:I_BBOX + 4])
        w = min(max(w, 0.02), 1.3)
        h = min(max(h, 0.02), 1.3)
        cx = min(max(cx, -0.2), 1.2)
        cy = min(max(cy, -0.2), 1.2)
        bbox = BBox(cx, cy, w, h)
        n = int(np.clip(round(float(v[I_NPTS])), 2, NUM_PTS))
        pts_uv = v[I_PTS:I_PTS + 2 * NUM_PTS].reshape(NUM_PTS, 2)[:n]
        pts_uv = np.clip(pts_uv, 0.0, 1.0)

        fill = _decode_fill(v)
        stroke = None
        if v[I_SVALID] > 0.5:
            sw = float(np.clip(v[I_SW], 0.002, 0.25))
            stroke = Stroke(sw, tuple(float(c) for c in v[I_SRGB:I_SRGB + 3]),
                            float(np.clip(v[I_SALPHA], 0.0, 1.0)))
        opacity = float(np.clip(v[I_OPACITY], 0.0, 1.0))

        if cls == CLS_BLOB:
            ctrl = [(float(u), float(vv)) for u, vv in pts_uv]
            segments = [Segment(CMD_M, [ctrl[0][0], ctrl[0][1]])]
            segments.extend(_bezier_ring(ctrl))
            hole = float(v[I_HOLE])
            if hole > 0.02:
                inner_uv = 0.5 + (pts_uv - 0.5) * hole
                inner = [(float(u), float(vv)) for u, vv in inner_uv]
                segments.append(Segment(CMD_M, [inner[0][0], inner[0][1]]))
                segments.extend(_bezier_ring(inner))
            geom = PathGeom(bbox=bbox, segments=segments, closed=True)
        elif cls == CLS_POLYGON:
            geom = PolygonGeom(bbox=bbox,
                               points=[tuple(map(float, p)) for p in pts_uv])
        elif cls == CLS_ELLIPSE:
            geom = EllipseGeom(bbox=bbox)
        elif cls == CLS_RECT:
            geom = RectGeom(bbox=bbox)
        else:
            poly = [(float(u), float(vv)) for u, vv in pts_uv]
            segments = [Segment(CMD_M, [poly[0][0], poly[0][1]])]
            for p in poly[1:]:
                segments.append(Segment(CMD_L, [p[0], p[1]]))
            geom = PathGeom(bbox=bbox, segments=segments, closed=False)
            fill = Fill(type=FILL_NONE)
            if stroke is None:
                continue

        obj = ShapeObject(shape=SHAPE_PATH if cls in (CLS_BLOB, CLS_STROKE) else
                          (SHAPE_POLYGON if cls == CLS_POLYGON else
                           (SHAPE_ELLIPSE if cls == CLS_ELLIPSE else SHAPE_RECT)),
                          geometry=geom, fill=fill, stroke=stroke, opacity=opacity)
        objects.append(obj)

    background = None
    if bg[0] > 0.5:
        background = tuple(float(np.clip(c, 0.0, 1.0)) for c in bg[1:5])
    return Scene(width=canvas, height=canvas, background=background, objects=objects)


def _bezier_ring(points: list) -> list:
    n = len(points)
    segs = []
    arr = np.asarray(points, dtype=np.float64)
    for i in range(n):
        p0 = arr[(i - 1) % n]
        p1 = arr[i]
        p2 = arr[(i + 1) % n]
        p3 = arr[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        segs.append(Segment(CMD_C, [float(c1[0]), float(c1[1]),
                                    float(c2[0]), float(c2[1]),
                                    float(p2[0]), float(p2[1])]))
    segs.append(Segment(CMD_Z, []))
    return segs


def _decode_fill(v: np.ndarray) -> Fill:
    ftype = int(round(float(v[I_FTYPE]))) % 4
    if ftype == FILL_NONE_I:
        return Fill(type=FILL_NONE)
    if ftype == FILL_SOLID_I:
        rgb = tuple(float(np.clip(c, 0.0, 1.0)) for c in v[I_FRGB:I_FRGB + 3])
        return Fill(type=FILL_SOLID, color=rgb,
                    alpha=float(np.clip(v[I_FALPHA], 0.0, 1.0)))
    kind = FILL_LINEAR if ftype == FILL_LINEAR_I else FILL_RADIAL
    n_stops = int(np.clip(round(float(v[I_NSTOPS])), 2, NUM_STOPS))
    raw = v[I_STOPS:I_STOPS + NUM_STOPS * 5].reshape(NUM_STOPS, 5)[:n_stops]
    order = np.argsort(raw[:, 0])
    stops = [Stop(float(np.clip(raw[j, 0], 0.0, 1.0)),
                  tuple(float(np.clip(c, 0.0, 1.0)) for c in raw[j, 1:4]),
                  float(np.clip(raw[j, 4], 0.0, 1.0))) for j in order]
    p0 = tuple(float(np.clip(c, -1.0, 2.0)) for c in v[I_GP0:I_GP0 + 2])
    p1 = tuple(float(np.clip(c, -1.0, 2.0)) for c in v[I_GP1:I_GP1 + 2])
    radius = float(np.clip(v[I_GRAD], 0.02, 1.5))
    grad = Gradient(kind=kind, p0=p0, p1=p1, radius=radius, stops=stops)
    return Fill(type=kind, color=(0.0, 0.0, 0.0), alpha=1.0, gradient=grad)
