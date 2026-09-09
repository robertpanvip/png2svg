from __future__ import annotations

import numpy as np

from svg.scene_graph import (
    Scene, ShapeObject, BBox, Segment, PathGeom, CircleGeom, EllipseGeom, RectGeom,
    PolygonGeom, Stop, Gradient, Fill, Stroke, Effects, Clip, Mask,
    FILL_NONE, FILL_SOLID, FILL_LINEAR, FILL_RADIAL,
    SHAPE_PATH,
    CMD_M, CMD_L, CMD_Q, CMD_C, CMD_A, CMD_Z,
)

# ---- 容量常量（HANDOFF §11.6 契约）----
NUM_SLOTS = 8
N_SEG = 16          # 每 slot 最大段数（实测 generator 单对象最大 16 段）
SEG_DIM = 12        # 每段 = type one-hot(5) + 坐标(7)
NUM_STOPS = 4       # 渐变 stop 数（generator 可产 2..6，编码截断到 4）
NUM_DASH = 4
SLOT_DIM = 280      # 实际使用 275，275..279 备用
BG_DIM = 5

# 段类型（无 Z：Z 由 closed 标志重建——closed=True 时每个 subpath 末尾补 Z，
# 与 generator 产出约定一致：所有 subpath 要么全闭合要么全开放）
SEG_M, SEG_L, SEG_Q, SEG_C, SEG_A = range(5)
_NSEG_TYPES = 5

# ---- 几何基 0..6 ----
I_VALID = 0
I_BBOX = 1          # cx,cy,w,h → 1..4
I_NSEG = 5
I_CLOSED = 6
# ---- 段表 7..198（16×12）----
I_SEG = 7
# ---- fill 块 199..232（34 维）----
I_FTYPE = 199       # one-hot {none,solid,linear,radial} → 199..202
I_FRGB = 203        # 3 → 203..205
I_FALPHA = 206
I_GP0 = 207         # 2 → 207..208
I_GP1 = 209         # 2 → 209..210
I_GRAD = 211        # radius
I_STOPS = 212       # 4×5 (pos,rgb,alpha) → 212..231
I_NSTOPS = 232
# ---- stroke 块 233..249（17 维）----
I_SVALID = 233
I_SWIDTH = 234
I_SRGB = 235        # 3 → 235..237
I_SALPHA = 238
I_SDASH = 239       # 4 值 + offset → 239..243（offset 恒 0，备用）
I_SCAP = 244        # one-hot {butt,round,square} → 244..246
I_SJOIN = 247       # one-hot {miter,round,bevel} → 247..249
# ---- effects 块 250..263（14 维）----
I_OPACITY = 250
I_BLUR = 251
I_SDX = 252
I_SDY = 253
I_SBLUR = 254
I_ESRGB = 255       # shadow rgb → 255..257
I_ESALPHA = 258
I_GLOWR = 259
I_EGRGB = 260       # glow rgb → 260..262
I_EGALPHA = 263
# ---- composition 块 264..274（11 维）----
I_GROUP = 264       # one-hot group id {0,1,2,3} → 264..267
I_CLIP = 268        # has_clip
I_CLIP_REF = 269
I_MASK = 270        # has_mask
I_MASK_REF = 271
I_MASK_KIND = 272   # luminance=0 / alpha=1
I_END = 275         # 实际使用结束（不含备用槽）；SLOT_DIM=280，275..279 备用
# 273..274 备用

_CAP_ON, _CAP_ROUND, _CAP_SQUARE = range(3)
_JN_MITER, _JN_ROUND, _JN_BEVEL = range(3)
_FT_NONE, _FT_SOLID, _FT_LINEAR, _FT_RADIAL = range(4)

_FILL_I = {FILL_NONE: _FT_NONE, FILL_SOLID: _FT_SOLID,
           FILL_LINEAR: _FT_LINEAR, FILL_RADIAL: _FT_RADIAL}
_CAP_I = {"butt": _CAP_ON, "round": _CAP_ROUND, "square": _CAP_SQUARE}
_JN_I = {"miter": _JN_MITER, "round": _JN_ROUND, "bevel": _JN_BEVEL}


def _slot_template() -> np.ndarray:
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    v[I_CLOSED] = 1.0
    v[I_FALPHA] = 1.0
    v[I_OPACITY] = 1.0
    v[I_SWIDTH] = 0.02
    v[I_FTYPE + _FT_NONE] = 1.0  # 默认无填充
    return v


def _onehot(v: np.ndarray, base: int, n: int, idx: int):
    v[base:base + n] = 0.0
    v[base + idx] = 1.0


def _canonical_segments(obj: ShapeObject) -> tuple:
    """任意几何 → (segments(无 Z), closed)。PathGeom 原样；其余转规范路径。"""
    g = obj.geometry
    if isinstance(g, PathGeom):
        segs = [s for s in g.segments if s.cmd != CMD_Z]
        return segs, bool(g.closed)
    if isinstance(g, RectGeom):
        return ([Segment(CMD_M, [0.0, 0.0]), Segment(CMD_L, [1.0, 0.0]),
                 Segment(CMD_L, [1.0, 1.0]), Segment(CMD_L, [0.0, 1.0])]), True
    if isinstance(g, (CircleGeom, EllipseGeom)):
        # uv 帧 4 段 90° 弧椭圆：r=0.5
        ring = [(1.0, 0.5), (0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 0.5)]
        segs = [Segment(CMD_M, list(ring[0]))]
        for k in range(1, 5):
            segs.append(Segment(CMD_A, [0.5, 0.5, 0.0, 0.0, 1.0,
                                        ring[k][0], ring[k][1]]))
        return segs, True
    if isinstance(g, PolygonGeom):
        pts = [tuple(map(float, p)) for p in g.points]
        segs = [Segment(CMD_M, list(pts[0]))]
        segs += [Segment(CMD_L, list(p)) for p in pts[1:]]
        return segs, True
    raise TypeError(f"unsupported geometry {type(g)!r}")


def _encode_segment(v: np.ndarray, base: int, seg: Segment):
    v[base:base + SEG_DIM] = 0.0
    p = seg.pts
    _c = lambda x: float(np.clip(x, 0.0, 1.0))
    if seg.cmd == CMD_M:
        _onehot(v, base, _NSEG_TYPES, SEG_M)
        v[base + 5:base + 7] = (_c(p[0]), _c(p[1]))
    elif seg.cmd == CMD_L:
        _onehot(v, base, _NSEG_TYPES, SEG_L)
        v[base + 5:base + 7] = (_c(p[0]), _c(p[1]))
    elif seg.cmd == CMD_Q:
        _onehot(v, base, _NSEG_TYPES, SEG_Q)
        v[base + 5:base + 9] = (_c(p[0]), _c(p[1]), _c(p[2]), _c(p[3]))
    elif seg.cmd == CMD_C:
        _onehot(v, base, _NSEG_TYPES, SEG_C)
        v[base + 5:base + 11] = (_c(p[0]), _c(p[1]), _c(p[2]), _c(p[3]),
                                 _c(p[4]), _c(p[5]))
    elif seg.cmd == CMD_A:
        _onehot(v, base, _NSEG_TYPES, SEG_A)
        # A: [rx, ry, rot(→/360), largearc, sweep, ex, ey]
        v[base + 5:base + 12] = (
            np.clip(p[0], 0.0, 1.0), np.clip(p[1], 0.0, 1.0),
            np.clip(p[2] / 360.0, 0.0, 1.0), float(p[3]), float(p[4]),
            np.clip(p[5], 0.0, 1.0), np.clip(p[6], 0.0, 1.0),
        )
    else:
        raise ValueError(f"bad segment cmd {seg.cmd!r}")


def encode_scene(scene: Scene) -> tuple:
    """Scene → (slots [K, SLOT_DIM], bg [BG_DIM])。

    对象按**场景原始顺序**编码（不再按面积排序）——composition 引用
    （clip.ref / mask.ref）指向原始下标，且 z-order 本身有语义。
    """
    K = NUM_SLOTS
    slots = np.stack([_slot_template() for _ in range(K)])
    for i, obj in enumerate(list(scene.objects)[:K]):
        v = slots[i]
        g = obj.geometry
        bb = g.bbox
        v[I_VALID] = 1.0
        v[I_BBOX:I_BBOX + 4] = (bb.cx, bb.cy, bb.w, bb.h)

        segs, closed = _canonical_segments(obj)
        v[I_CLOSED] = 1.0 if closed else 0.0
        n = min(len(segs), N_SEG)
        v[I_NSEG] = float(n)
        for j in range(n):
            _encode_segment(v, I_SEG + j * SEG_DIM, segs[j])

        # fill
        f = obj.fill
        _onehot(v, I_FTYPE, 4, _FILL_I[f.type])
        if f.type == FILL_SOLID:
            v[I_FRGB:I_FRGB + 3] = f.color
            v[I_FALPHA] = f.alpha
        elif f.type in (FILL_LINEAR, FILL_RADIAL):
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
            v[I_FALPHA] = 0.0

        # stroke
        st = obj.stroke
        if st is not None:
            v[I_SVALID] = 1.0
            v[I_SWIDTH] = st.width
            v[I_SRGB:I_SRGB + 3] = st.color
            v[I_SALPHA] = st.alpha
            dash = list(st.dash)[:NUM_DASH]
            v[I_SDASH:I_SDASH + len(dash)] = dash
            v[I_SDASH + NUM_DASH] = float(len(dash))  # offset 槽复用为 dash 数量
            _onehot(v, I_SCAP, 3, _CAP_I.get(st.linecap, _CAP_ON))
            _onehot(v, I_SJOIN, 3, _JN_I.get(st.linejoin, _JN_MITER))

        # effects
        v[I_OPACITY] = obj.opacity
        e = obj.effects
        if e is not None:
            v[I_BLUR] = e.blur_radius
            v[I_SDX] = e.shadow_dx
            v[I_SDY] = e.shadow_dy
            v[I_SBLUR] = e.shadow_blur
            v[I_ESRGB:I_ESRGB + 3] = e.shadow_rgb
            v[I_ESALPHA] = e.shadow_alpha
            v[I_GLOWR] = e.glow_radius
            v[I_EGRGB:I_EGRGB + 3] = e.glow_rgb
            v[I_EGALPHA] = e.glow_alpha

        # composition
        grp = int(np.clip(int(obj.group), 0, 3))
        _onehot(v, I_GROUP, 4, grp)
        if obj.clip is not None:
            v[I_CLIP] = 1.0
            v[I_CLIP_REF] = float(np.clip(obj.clip.ref, 0, K - 1))
        if obj.mask is not None:
            v[I_MASK] = 1.0
            v[I_MASK_REF] = float(np.clip(obj.mask.ref, 0, K - 1))
            v[I_MASK_KIND] = 1.0 if obj.mask.kind == "alpha" else 0.0

    if scene.background is not None:
        bg = np.array([1.0, *scene.background], dtype=np.float32)
    else:
        bg = np.zeros(BG_DIM, dtype=np.float32)
    return slots.astype(np.float32), bg


# ---- decode ----

def _decode_segment(v: np.ndarray, base: int) -> Segment:
    t = int(np.argmax(v[base:base + _NSEG_TYPES]))
    c = v[base + 5:base + 12]
    if t == SEG_M:
        return Segment(CMD_M, [float(c[0]), float(c[1])])
    if t == SEG_L:
        return Segment(CMD_L, [float(c[0]), float(c[1])])
    if t == SEG_Q:
        return Segment(CMD_Q, [float(c[0]), float(c[1]), float(c[2]), float(c[3])])
    if t == SEG_C:
        return Segment(CMD_C, [float(c[0]), float(c[1]), float(c[2]),
                               float(c[3]), float(c[4]), float(c[5])])
    return Segment(CMD_A, [float(max(c[0], 0.01)), float(max(c[1], 0.01)),
                           float(c[2]) * 360.0,
                           1.0 if c[3] > 0.5 else 0.0,
                           1.0 if c[4] > 0.5 else 0.0,
                           float(c[5]), float(c[6])])


def _clamp01_pair(x, y):
    return (float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0)))


def _decode_fill(v: np.ndarray) -> Fill:
    ftype = int(np.argmax(v[I_FTYPE:I_FTYPE + 4]))
    if ftype == _FT_NONE:
        return Fill(type=FILL_NONE)
    if ftype == _FT_SOLID:
        rgb = tuple(float(np.clip(c, 0.0, 1.0)) for c in v[I_FRGB:I_FRGB + 3])
        return Fill(type=FILL_SOLID, color=rgb,
                    alpha=float(np.clip(v[I_FALPHA], 0.0, 1.0)))
    kind = FILL_LINEAR if ftype == _FT_LINEAR else FILL_RADIAL
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


def _decode_stroke(v: np.ndarray):
    if v[I_SVALID] <= 0.5:
        return None
    sw = float(np.clip(v[I_SWIDTH], 0.002, 0.25))
    rgb = tuple(float(np.clip(c, 0.0, 1.0)) for c in v[I_SRGB:I_SRGB + 3])
    alpha = float(np.clip(v[I_SALPHA], 0.0, 1.0))
    nd = int(np.clip(round(float(v[I_SDASH + NUM_DASH])), 0, NUM_DASH))
    dash = tuple(float(np.clip(d, 0.005, 1.0)) for d in v[I_SDASH:I_SDASH + nd])
    cap_i = int(np.argmax(v[I_SCAP:I_SCAP + 3]))
    jn_i = int(np.argmax(v[I_SJOIN:I_SJOIN + 3]))
    cap = ("butt", "round", "square")[cap_i]
    jn = ("miter", "round", "bevel")[jn_i]
    return Stroke(width=sw, color=rgb, alpha=alpha, dash=dash,
                  linecap=cap, linejoin=jn)


def decode_scene(slots: np.ndarray, bg: np.ndarray, canvas: int = 256) -> Scene:
    slots = np.asarray(slots, dtype=np.float64)
    objects = []
    for i in range(min(NUM_SLOTS, slots.shape[0])):
        v = slots[i]
        if v[I_VALID] < 0.5:
            continue
        cx, cy, w, h = (float(x) for x in v[I_BBOX:I_BBOX + 4])
        w = min(max(w, 0.02), 1.3)
        h = min(max(h, 0.02), 1.3)
        cx = min(max(cx, -0.2), 1.2)
        cy = min(max(cy, -0.2), 1.2)
        bbox = BBox(cx, cy, w, h)

        n = int(np.clip(round(float(v[I_NSEG])), 0, N_SEG))
        closed = float(v[I_CLOSED]) > 0.5
        segments = []
        for j in range(n):
            seg = _decode_segment(v, I_SEG + j * SEG_DIM)
            if seg.cmd == CMD_M and j > 0 and closed:
                segments.append(Segment(CMD_Z, []))  # 上一个 subpath 收口
            if seg.cmd == CMD_M:
                p = _clamp01_pair(*seg.pts)
                segments.append(Segment(CMD_M, [p[0], p[1]]))
            elif seg.cmd in (CMD_L, CMD_Q, CMD_C):
                pts = [_clamp01_pair(seg.pts[k], seg.pts[k + 1])
                       for k in range(0, len(seg.pts), 2)]
                flat = [c for pt in pts for c in pt]
                segments.append(Segment(seg.cmd, flat))
            else:  # CMD_A
                ex, ey = _clamp01_pair(seg.pts[5], seg.pts[6])
                segments.append(Segment(CMD_A, [seg.pts[0], seg.pts[1],
                                                seg.pts[2], seg.pts[3],
                                                seg.pts[4], ex, ey]))
        if closed and n > 0 and segments[-1].cmd != CMD_Z:
            segments.append(Segment(CMD_Z, []))
        if not segments:
            continue

        fill = _decode_fill(v)
        stroke = _decode_stroke(v)
        opacity = float(np.clip(v[I_OPACITY], 0.0, 1.0))

        e = None
        if v[I_BLUR] > 0.005 or v[I_SBLUR] > 0.005 or v[I_GLOWR] > 0.005:
            e = Effects(
                blur_radius=float(np.clip(v[I_BLUR], 0.0, 0.2)),
                shadow_dx=float(np.clip(v[I_SDX], -0.2, 0.2)),
                shadow_dy=float(np.clip(v[I_SDY], -0.2, 0.2)),
                shadow_blur=float(np.clip(v[I_SBLUR], 0.0, 0.2)),
                shadow_rgb=tuple(float(np.clip(c, 0.0, 1.0)) for c in v[I_ESRGB:I_ESRGB + 3]),
                shadow_alpha=float(np.clip(v[I_ESALPHA], 0.0, 1.0)),
                glow_radius=float(np.clip(v[I_GLOWR], 0.0, 0.2)),
                glow_rgb=tuple(float(np.clip(c, 0.0, 1.0)) for c in v[I_EGRGB:I_EGRGB + 3]),
                glow_alpha=float(np.clip(v[I_EGALPHA], 0.0, 1.0)),
            )

        grp = int(np.argmax(v[I_GROUP:I_GROUP + 4]))
        clip = None
        if v[I_CLIP] > 0.5:
            clip = Clip(ref=int(np.clip(round(float(v[I_CLIP_REF])), 0, NUM_SLOTS - 1)))
        mask = None
        if v[I_MASK] > 0.5:
            kind = "alpha" if v[I_MASK_KIND] > 0.5 else "luminance"
            mask = Mask(ref=int(np.clip(round(float(v[I_MASK_REF])), 0, NUM_SLOTS - 1)),
                        kind=kind)

        objects.append(ShapeObject(
            shape=SHAPE_PATH,
            geometry=PathGeom(bbox=bbox, segments=segments, closed=closed),
            fill=fill, stroke=stroke, opacity=opacity, effects=e,
            group=grp, clip=clip, mask=mask,
        ))

    background = None
    if bg[0] > 0.5:
        background = tuple(float(np.clip(c, 0.0, 1.0)) for c in bg[1:5])
    return Scene(width=canvas, height=canvas, background=background, objects=objects)
