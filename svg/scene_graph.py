from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

FILL_NONE = "none"
FILL_SOLID = "solid"
FILL_LINEAR = "linear"
FILL_RADIAL = "radial"

SHAPE_PATH = "path"
SHAPE_CIRCLE = "circle"
SHAPE_ELLIPSE = "ellipse"
SHAPE_RECT = "rect"
SHAPE_POLYGON = "polygon"

CMD_M = "M"
CMD_L = "L"
CMD_Q = "Q"
CMD_C = "C"
CMD_A = "A"
CMD_Z = "Z"

# Total coordinate count per command (flat list in Segment.pts).
#   M=[sx,sy]  L=[ex,ey]  Q=[cx,cy,ex,ey]  C=[c1x,c1y,c2x,c2y,ex,ey]
#   A=[rx,ry,rot,largearc,sweep,ex,ey]  Z=[]
_CMD_POINTS = {CMD_M: 2, CMD_L: 2, CMD_Q: 4, CMD_C: 6, CMD_A: 7, CMD_Z: 0}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class BBox:
    cx: float
    cy: float
    w: float
    h: float

    def map(self, u: float, v: float):
        return (self.cx + (u - 0.5) * self.w, self.cy + (v - 0.5) * self.h)

    def unmap(self, x: float, y: float):
        return ((x - self.cx) / self.w + 0.5, (y - self.cy) / self.h + 0.5)


@dataclass
class Segment:
    cmd: str
    pts: list

    def __post_init__(self):
        n = _CMD_POINTS[self.cmd]
        if len(self.pts) != n:
            raise ValueError(f"cmd {self.cmd} expects {n} coords, got {len(self.pts)}")


@dataclass
class PathGeom:
    bbox: BBox
    segments: list
    closed: bool = True


@dataclass
class CircleGeom:
    bbox: BBox


@dataclass
class EllipseGeom:
    bbox: BBox


@dataclass
class RectGeom:
    bbox: BBox


@dataclass
class PolygonGeom:
    bbox: BBox
    points: list


@dataclass
class Stop:
    position: float
    rgb: tuple
    alpha: float = 1.0


@dataclass
class Gradient:
    kind: str
    p0: tuple
    p1: tuple
    radius: float = 0.0
    stops: list = field(default_factory=list)


@dataclass
class Fill:
    type: str = FILL_SOLID
    color: tuple = (0.0, 0.0, 0.0)
    alpha: float = 1.0
    gradient: Optional[Gradient] = None


@dataclass
class Stroke:
    width: float
    color: tuple
    alpha: float = 1.0
    dash: tuple = ()          # 虚线数组（用户单位，uv 归一化），空=实线
    linecap: str = "butt"     # butt / round / square
    linejoin: str = "miter"   # miter / round / bevel


@dataclass
class Effects:
    shadow_dx: float = 0.0
    shadow_dy: float = 0.0
    shadow_blur: float = 0.0
    shadow_rgb: tuple = (0.0, 0.0, 0.0)
    shadow_alpha: float = 0.0
    glow_radius: float = 0.0
    glow_rgb: tuple = (1.0, 1.0, 1.0)
    glow_alpha: float = 0.0
    blur_radius: float = 0.0


@dataclass
class Clip:
    ref: int = -1                      # 引用另一 object 索引作为裁切路径（-1=未用）
    path: Optional["PathGeom"] = None  # 或内联裁切路径


@dataclass
class Mask:
    ref: int = -1                      # 引用另一 object 索引作为亮度遮罩
    kind: str = "luminance"            # luminance / alpha


@dataclass
class ShapeObject:
    shape: str
    geometry: object
    fill: Fill = field(default_factory=Fill)
    stroke: Optional[Stroke] = None
    opacity: float = 1.0
    effects: Optional[Effects] = None
    group: int = 0                     # 组合分组 id（0=无分组）
    clip: Optional[Clip] = None        # 裁切
    mask: Optional[Mask] = None        # 遮罩


@dataclass
class Scene:
    width: int = 256
    height: int = 256
    background: Optional[tuple] = None
    objects: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "background": list(self.background) if self.background is not None else None,
            "objects": [_object_to_dict(o) for o in self.objects],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        bg = d.get("background")
        return Scene(
            width=int(d["width"]),
            height=int(d["height"]),
            background=tuple(bg) if bg is not None else None,
            objects=[_object_from_dict(o) for o in d.get("objects", [])],
        )

    @staticmethod
    def from_json(s: str) -> "Scene":
        return Scene.from_dict(json.loads(s))

    def validate(self):
        for name in ("width", "height"):
            v = getattr(self, name)
            if not (8 <= v <= 4096):
                raise ValueError(f"scene {name} out of range: {v}")
        if self.background is not None:
            _check_rgba(self.background, "background")
        for i, obj in enumerate(self.objects):
            _validate_object(obj, i)
        return self


def _object_to_dict(o: ShapeObject) -> dict:
    g = o.geometry
    if isinstance(g, PathGeom):
        geom = {
            "kind": SHAPE_PATH,
            "bbox": _bbox_to_dict(g.bbox),
            "segments": [{"cmd": s.cmd, "pts": list(s.pts)} for s in g.segments],
            "closed": g.closed,
        }
    elif isinstance(g, CircleGeom):
        geom = {"kind": SHAPE_CIRCLE, "bbox": _bbox_to_dict(g.bbox)}
    elif isinstance(g, EllipseGeom):
        geom = {"kind": SHAPE_ELLIPSE, "bbox": _bbox_to_dict(g.bbox)}
    elif isinstance(g, RectGeom):
        geom = {"kind": SHAPE_RECT, "bbox": _bbox_to_dict(g.bbox)}
    elif isinstance(g, PolygonGeom):
        geom = {"kind": SHAPE_POLYGON, "bbox": _bbox_to_dict(g.bbox), "points": [list(p) for p in g.points]}
    else:
        raise TypeError(f"unknown geometry {type(g)!r}")
    return {
        "shape": o.shape,
        "geometry": geom,
        "fill": _fill_to_dict(o.fill),
        "stroke": None if o.stroke is None else {
            "width": o.stroke.width, "color": list(o.stroke.color), "alpha": o.stroke.alpha,
        },
        "opacity": o.opacity,
        "effects": None if o.effects is None else vars(o.effects).copy(),
        "group": o.group,
        "clip": None if o.clip is None else {
            "ref": o.clip.ref,
            "path": None if o.clip.path is None else {
                "bbox": _bbox_to_dict(o.clip.path.bbox),
                "segments": [{"cmd": s.cmd, "pts": list(s.pts)} for s in o.clip.path.segments],
                "closed": o.clip.path.closed,
            },
        },
        "mask": None if o.mask is None else {"ref": o.mask.ref, "kind": o.mask.kind},
    }


def _bbox_to_dict(b: BBox) -> dict:
    return {"cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h}


def _fill_to_dict(f: Fill) -> dict:
    return {
        "type": f.type,
        "color": list(f.color),
        "alpha": f.alpha,
        "gradient": None if f.gradient is None else {
            "kind": f.gradient.kind,
            "p0": list(f.gradient.p0),
            "p1": list(f.gradient.p1),
            "radius": f.gradient.radius,
            "stops": [{"position": s.position, "rgb": list(s.rgb), "alpha": s.alpha} for s in f.gradient.stops],
        },
    }


def _object_from_dict(d: dict) -> ShapeObject:
    g = d["geometry"]
    kind = g["kind"]
    bb = g["bbox"]
    bbox = BBox(bb["cx"], bb["cy"], bb["w"], bb["h"])
    if kind == SHAPE_PATH:
        geom = PathGeom(
            bbox=bbox,
            segments=[Segment(s["cmd"], list(s["pts"])) for s in g["segments"]],
            closed=bool(g.get("closed", True)),
        )
    elif kind == SHAPE_CIRCLE:
        geom = CircleGeom(bbox=bbox)
    elif kind == SHAPE_ELLIPSE:
        geom = EllipseGeom(bbox=bbox)
    elif kind == SHAPE_RECT:
        geom = RectGeom(bbox=bbox)
    elif kind == SHAPE_POLYGON:
        geom = PolygonGeom(bbox=bbox, points=[tuple(p) for p in g["points"]])
    else:
        raise ValueError(f"unknown geometry kind {kind!r}")

    fd = d.get("fill") or {}
    gd = fd.get("gradient")
    fill = Fill(
        type=fd.get("type", FILL_SOLID),
        color=tuple(fd.get("color", (0.0, 0.0, 0.0))),
        alpha=float(fd.get("alpha", 1.0)),
        gradient=None if gd is None else Gradient(
            kind=gd["kind"], p0=tuple(gd["p0"]), p1=tuple(gd["p1"]),
            radius=float(gd.get("radius", 0.0)),
            stops=[Stop(s["position"], tuple(s["rgb"]), float(s["alpha"])) for s in gd["stops"]],
        ),
    )
    sd = d.get("stroke")
    stroke = None if sd is None else Stroke(
        float(sd["width"]), tuple(sd["color"]), float(sd.get("alpha", 1.0)),
        dash=tuple(sd.get("dash", ()) or ()),
        linecap=str(sd.get("linecap", "butt")),
        linejoin=str(sd.get("linejoin", "miter")),
    )
    ed = d.get("effects")
    effects = None if ed is None else Effects(**ed)
    cd = d.get("clip")
    clip = None if cd is None else Clip(
        ref=int(cd.get("ref", -1)),
        path=None if cd.get("path") is None else PathGeom(
            bbox=BBox(**cd["path"]["bbox"]),
            segments=[Segment(s["cmd"], list(s["pts"])) for s in cd["path"]["segments"]],
            closed=bool(cd["path"].get("closed", True)),
        ),
    )
    mdd = d.get("mask")
    mask = None if mdd is None else Mask(ref=int(mdd.get("ref", -1)), kind=str(mdd.get("kind", "luminance")))
    return ShapeObject(
        shape=d["shape"], geometry=geom, fill=fill, stroke=stroke,
        opacity=float(d.get("opacity", 1.0)), effects=effects,
        group=int(d.get("group", 0)), clip=clip, mask=mask,
    )


def _check_rgba(c, name):
    if len(c) != 4:
        raise ValueError(f"{name} must be rgba")
    for v in c:
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"{name} component out of range: {v}")


def _validate_object(obj: ShapeObject, i: int):
    if obj.shape not in (SHAPE_PATH, SHAPE_CIRCLE, SHAPE_ELLIPSE, SHAPE_RECT, SHAPE_POLYGON):
        raise ValueError(f"object {i}: bad shape {obj.shape!r}")
    if not (0.0 <= obj.opacity <= 1.0):
        raise ValueError(f"object {i}: opacity out of range")
    if obj.shape == SHAPE_PATH and not obj.geometry.segments:
        raise ValueError(f"object {i}: empty path")
    if obj.shape == SHAPE_POLYGON and len(obj.geometry.points) < 3:
        raise ValueError(f"object {i}: polygon needs >= 3 points")
    for bb in (obj.geometry.bbox,):
        for attr in ("cx", "cy", "w", "h"):
            v = getattr(bb, attr)
            if not (-0.25 <= v <= 1.25) or (attr in ("w", "h") and v <= 0):
                raise ValueError(f"object {i}: bbox {attr} out of range: {v}")
    f = obj.fill
    if f.type not in (FILL_NONE, FILL_SOLID, FILL_LINEAR, FILL_RADIAL):
        raise ValueError(f"object {i}: bad fill type {f.type!r}")
    if f.type in (FILL_LINEAR, FILL_RADIAL) and f.gradient is None:
        raise ValueError(f"object {i}: gradient fill without gradient")
    if f.gradient is not None:
        if len(f.gradient.stops) < 2:
            raise ValueError(f"object {i}: gradient needs >= 2 stops")
        prev = -1e-9
        for s in f.gradient.stops:
            if not (prev - 1e-9 <= s.position <= 1.0 + 1e-9):
                raise ValueError(f"object {i}: stop positions must be non-decreasing")
            prev = s.position
    if obj.stroke is not None:
        if not (0.0 < obj.stroke.width <= 0.25):
            raise ValueError(f"object {i}: stroke width out of range")
        if obj.stroke.linecap not in ("butt", "round", "square"):
            raise ValueError(f"object {i}: bad linecap {obj.stroke.linecap!r}")
        if obj.stroke.linejoin not in ("miter", "round", "bevel"):
            raise ValueError(f"object {i}: bad linejoin {obj.stroke.linejoin!r}")
        if any(not (0.0 <= d <= 1.0) for d in obj.stroke.dash):
            raise ValueError(f"object {i}: dash value out of range")
    if obj.clip is not None:
        # clip.ref 引用另一 object 下标，由 generator/encoder 侧保证合法，此处仅做符号校验
        if obj.clip.ref < -1:
            raise ValueError(f"object {i}: bad clip ref {obj.clip.ref}")
    if obj.mask is not None and obj.mask.kind not in ("luminance", "alpha"):
        raise ValueError(f"object {i}: bad mask kind {obj.mask.kind!r}")
