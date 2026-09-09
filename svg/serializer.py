from __future__ import annotations

import math

from .scene_graph import (
    Scene, ShapeObject, PathGeom, CircleGeom, EllipseGeom, RectGeom, PolygonGeom,
    FILL_NONE, FILL_LINEAR, FILL_RADIAL,
    CMD_M, CMD_L, CMD_Q, CMD_C, CMD_A, CMD_Z,
)

_GRAD_MAX = 4096.0


def _f(x: float) -> str:
    s = f"{float(x):.2f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s


def _rgb255(c) -> tuple:
    return tuple(max(0, min(255, int(round(float(v) * 255.0)))) for v in c)


def _rgba_css(c, a: float) -> str:
    r, g, b = _rgb255(c)
    return f"rgba({r},{g},{b},{_f(max(0.0, min(1.0, float(a))))})"


def _hex(c) -> str:
    return "#{:02x}{:02x}{:02x}".format(*_rgb255(c))


def _path_d(path: PathGeom, size) -> str:
    w, h = size
    parts = []
    for seg in path.segments:
        pts = seg.pts
        if seg.cmd == CMD_Z:
            parts.append("Z")
            continue
        if seg.cmd == CMD_A:
            # A 段 7 值: (rx, ry, rot, largearc, sweep, endx, endy)
            # rx/ry 在 bbox uv 帧 -> 乘边长；rot/flag 绝对；终点 bbox.map
            rx, ry, rot, la, sw, eux, euy = seg.pts
            ex, ey = path.bbox.map(eux, euy)
            parts.append(
                f"A{_f(rx * w)} {_f(ry * h)} {_f(rot)} {int(la)} {int(sw)} "
                f"{_f(ex * w)} {_f(ey * h)}"
            )
            continue
        mapped = []
        for i in range(0, len(pts), 2):
            x, y = path.bbox.map(pts[i], pts[i + 1])
            mapped.append((_f(x * w), _f(y * h)))
        if seg.cmd == CMD_M:
            parts.append(f"M{mapped[0][0]} {mapped[0][1]}")
        elif seg.cmd == CMD_L:
            parts.append(f"L{mapped[0][0]} {mapped[0][1]}")
        elif seg.cmd == CMD_Q:
            parts.append(f"Q{mapped[0][0]} {mapped[0][1]} {mapped[1][0]} {mapped[1][1]}")
        elif seg.cmd == CMD_C:
            parts.append(
                f"C{mapped[0][0]} {mapped[0][1]} {mapped[1][0]} {mapped[1][1]} {mapped[2][0]} {mapped[2][1]}"
            )
        else:
            raise ValueError(f"bad segment cmd {seg.cmd!r}")
    return "".join(parts)


def _gradient_stops_xml(gradient, indent: str) -> str:
    out = []
    for s in gradient.stops:
        out.append(
            f'{indent}<stop offset="{_f(s.position)}" stop-color="{_hex(s.rgb)}" stop-opacity="{_f(s.alpha)}"/>'
        )
    return "\n".join(out)


def _fill_attrs(obj: ShapeObject, size, defs: list) -> str:
    fill = obj.fill
    if fill.type == FILL_NONE:
        return 'fill="none"'
    if fill.type in (FILL_LINEAR, FILL_RADIAL):
        g = fill.gradient
        w, h = size
        gid = f"grad{len(defs)}"
        x0, y0 = g.p0[0] * w, g.p0[1] * h
        x1, y1 = g.p1[0] * w, g.p1[1] * h
        if g.kind == FILL_LINEAR:
            elem = (
                f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
                f'x1="{_f(x0)}" y1="{_f(y0)}" x2="{_f(x1)}" y2="{_f(y1)}">\n'
                f"{_gradient_stops_xml(g, '  ')}\n</linearGradient>"
            )
        else:
            r = max(0.0, g.radius) * max(w, h)
            elem = (
                f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" '
                f'cx="{_f(x0)}" cy="{_f(y0)}" r="{_f(r)}">\n'
                f"{_gradient_stops_xml(g, '  ')}\n</radialGradient>"
            )
        defs.append(elem)
        attrs = f'fill="url(#{gid})"'
        if fill.alpha < 1.0:
            attrs += f' fill-opacity="{_f(fill.alpha)}"'
        return attrs
    return f'fill="{_rgba_css(fill.color, fill.alpha)}"'


def _shape_element(obj: ShapeObject, size, defs: list) -> str:
    w, h = size
    g = obj.geometry
    attrs = [_fill_attrs(obj, size, defs)]
    if obj.stroke is not None:
        sw = max(0.0, obj.stroke.width * max(w, h))
        attrs.append(f'stroke="{_rgba_css(obj.stroke.color, obj.stroke.alpha)}"')
        attrs.append(f'stroke-width="{_f(sw)}"')
        attrs.append('stroke-linecap="round"')
        attrs.append('stroke-linejoin="round"')
    if obj.opacity < 1.0:
        attrs.append(f'opacity="{_f(obj.opacity)}"')
    attrs_str = " ".join(attrs)

    if obj.shape == "path" or isinstance(g, PathGeom):
        return f'<path d="{_path_d(g, size)}" fill-rule="evenodd" {attrs_str}/>'
    if obj.shape == "circle" or isinstance(g, CircleGeom):
        cx, cy = g.bbox.cx * w, g.bbox.cy * h
        r = 0.5 * min(g.bbox.w, g.bbox.h) * min(w, h)
        return f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" {attrs_str}/>'
    if obj.shape == "ellipse" or isinstance(g, EllipseGeom):
        cx, cy = g.bbox.cx * w, g.bbox.cy * h
        rx, ry = 0.5 * g.bbox.w * w, 0.5 * g.bbox.h * h
        return f'<ellipse cx="{_f(cx)}" cy="{_f(cy)}" rx="{_f(rx)}" ry="{_f(ry)}" {attrs_str}/>'
    if obj.shape == "rect" or isinstance(g, RectGeom):
        x = (g.bbox.cx - 0.5 * g.bbox.w) * w
        y = (g.bbox.cy - 0.5 * g.bbox.h) * h
        return (
            f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(g.bbox.w * w)}" '
            f'height="{_f(g.bbox.h * h)}" {attrs_str}/>'
        )
    if obj.shape == "polygon" or isinstance(g, PolygonGeom):
        pts = []
        for u, v in g.points:
            x, y = g.bbox.map(u, v)
            pts.append(f"{_f(x * w)},{_f(y * h)}")
        return f'<polygon points="{" ".join(pts)}" {attrs_str}/>'
    raise ValueError(f"unsupported shape {obj.shape!r}")


def serialize(scene: Scene) -> str:
    scene.validate()
    w, h = int(scene.width), int(scene.height)
    defs: list = []
    body: list = []
    if scene.background is not None:
        body.append(
            f'<rect x="0" y="0" width="{w}" height="{h}" '
            f'fill="{_rgba_css(scene.background[:3], scene.background[3])}"/>'
        )
    for obj in scene.objects:
        body.append(_shape_element(obj, (w, h), defs))
    defs_xml = f"<defs>{''.join(defs)}</defs>" if defs else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">{defs_xml}{"".join(body)}</svg>'
    )
