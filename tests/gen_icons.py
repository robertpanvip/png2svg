#!/usr/bin/env python3
"""生成一批“设计感”扁平 icon PNG（透明背景、抗锯齿），用于 png2svg 还原测试。

覆盖各类特征，逼出 png2svg 的边界：
  ic_app      圆角方块 + 线性渐变 + 白色圆（渐变 + 圆角 + 纯色圆）
  ic_ring     圆环（外圆 - 内孔，应识别为两弧/圆环区域）
  ic_star     5 角星（尖角，不应被圆化）
  ic_gear     齿轮（多小弧 + 中心孔，压力测试）
  ic_heart    心形（平滑闭合曲线，应被贝塞尔/平滑重建）
  ic_bubble   对话气泡（圆角矩形 + 三角尾，圆角应保留）
  ic_octagon  八边形 stop sign（多边形，不应被整圆化）
  ic_play     圆 + 播放三角（圆 + 多边形）
  ic_check    圆角方块 + 对勾描边（圆角 + 折线）
  ic_cloud    云（多个椭圆合并，平滑 blob）
  ic_ellipse  纯椭圆（测 fit_rotated_ellipse 分支）
  ic_capsule  胶囊/体育场（2 半圆 + 2 直边，测弧-线-弧衔接）
  ic_rhex     圆角六边形（直边 + 圆角，圆角多边形分支）
  ic_pentagon 尖角正五边形（应锐利，不被圆化）
  ic_drop     水滴（平滑曲线 + 1 尖点）
  ic_chevron  双雪佛龙箭头（尖角多边形）
  ic_rings3   三同心环

绘制在 4x 超采样画布上，再 LANCZOS 下采样到 200x200 得到干净抗锯齿边缘。
依赖：Pillow + numpy（通过 PYLIBS 注入）。
"""
import math
import os
import sys

_PYLIBS = os.environ.get("PYLIBS", "")
if _PYLIBS and os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)

import numpy as np
from PIL import Image, ImageDraw

S = 4          # 超采样倍率
N = 200        # 输出尺寸
W = N * S      # 超采样画布尺寸
OUT = os.path.join(os.path.dirname(__file__), "gen_icons", "in")
os.makedirs(OUT, exist_ok=True)


def q(x):
    """逻辑坐标 -> 超采样坐标。"""
    return x * S


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def vgrad(c1, c2):
    """垂直线性渐变 (W,W,3) uint8。"""
    arr = np.empty((W, W, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, W)
    for i in range(3):
        arr[:, :, i] = (c1[i] + (c2[i] - c1[i]) * ys)[:, None]
    return arr


def rgrad(c1, c2, cx, cy, rmax):
    """径向渐变 (W,W,3) uint8，中心 c1 -> 边缘 c2。"""
    arr = np.empty((W, W, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:W, 0:W].astype(np.float64)
    d = np.sqrt((xs - q(cx)) ** 2 + (ys - q(cy)) ** 2) / q(rmax)
    d = np.clip(d, 0, 1)
    for i in range(3):
        arr[:, :, i] = (c1[i] + (c2[i] - c1[i]) * d)
    return arr.astype(np.uint8)


def canvas():
    return Image.new("RGBA", (W, W), (0, 0, 0, 0))


def add(canv, draw_fn, fill):
    """在 canv 上叠加一个区域：draw_fn(draw) 把该区域画成白色 mask，fill 为
    (r,g,b,a) 纯色或 (W,W,3) 渐变数组。返回新画布。"""
    mask = Image.new("L", (W, W), 0)
    draw_fn(ImageDraw.Draw(mask))
    if isinstance(fill, np.ndarray):
        region = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        region.paste(Image.fromarray(fill, "RGB"), (0, 0), mask)
    else:
        r, g, b, a = fill
        solid = Image.new("RGB", (W, W), (r, g, b))
        region = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        region.paste(solid, (0, 0), mask)
        if a != 255:
            m = np.asarray(region.split()[3], dtype=np.float64)
            m = (m * (a / 255.0)).astype(np.uint8)
            region.putalpha(Image.fromarray(m, "L"))
    return Image.alpha_composite(canv, region)


# ---- 各 icon 的 mask 绘制函数 ----

def _star(d, cx, cy, r_out, r_in, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / n
        pts.append((q(cx + r * math.cos(a)), q(cy + r * math.sin(a))))
    d.polygon(pts, fill=255)


def _roundrect(d, x, y, w, h, rad, fill=255):
    d.rounded_rectangle([q(x), q(y), q(x + w), q(y + h)], radius=q(rad), fill=fill)


def _gear(d, cx, cy, r_out, r_in, teeth=8):
    pts = []
    for i in range(teeth * 2):
        a = -math.pi / 2 + i * math.pi / teeth
        r = r_out if i % 2 == 0 else r_in
        pts.append((q(cx + r * math.cos(a)), q(cy + r * math.sin(a))))
    d.polygon(pts, fill=255)


def _heart(d, cx, cy, scale):
    pts = []
    for t in [i * 2 * math.pi / 240 for i in range(241)]:
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((q(cx + x * scale), q(cy - y * scale)))
    d.polygon(pts, fill=255)


def _cloud(d, cx, cy):
    ells = [
        (cx, cy + 18, 55, 40),
        (cx - 38, cy + 13, 35, 28),
        (cx + 38, cy + 15, 38, 30),
        (cx, cy - 8, 42, 38),
    ]
    for (ex, ey, rx, ry) in ells:
        d.ellipse([q(ex - rx), q(ey - ry), q(ex + rx), q(ey + ry)], fill=255)


def _ellipse(d, cx, cy, rx, ry):
    d.ellipse([q(cx - rx), q(cy - ry), q(cx + rx), q(cy + ry)], fill=255)


def _capsule(d, cx, cy, w, h):
    # 胶囊/体育场形：圆角矩形，圆角半径 = 高/2（两端半圆 + 中间直边）。
    d.rounded_rectangle([q(cx - w / 2), q(cy - h / 2), q(cx + w / 2), q(cy + h / 2)],
                        radius=q(h / 2), fill=255)


def _roundpoly(d, cx, cy, r_out, n, cr, rot=-math.pi / 2):
    """圆角正 n 边形：每条边是直线，每个顶点用半径 cr 的外凸圆弧连接。"""
    verts = []
    for i in range(n):
        a = rot + i * 2 * math.pi / n
        verts.append((cx + r_out * math.cos(a), cy + r_out * math.sin(a)))
    m = len(verts)
    pts = []
    for i in range(m):
        p_prev = verts[(i - 1) % m]
        p_cur = verts[i]
        p_next = verts[(i + 1) % m]

        def tpt(a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            f = min(cr, L / 2 - 0.5)
            return (a[0] + dx / L * f, a[1] + dy / L * f)

        t1 = tpt(p_cur, p_prev)
        t2 = tpt(p_cur, p_next)
        a1 = math.atan2(t1[1] - p_cur[1], t1[0] - p_cur[0])
        a2 = math.atan2(t2[1] - p_cur[1], t2[0] - p_cur[0])
        da = a2 - a1
        while da > math.pi:
            da -= 2 * math.pi
        while da < -math.pi:
            da += 2 * math.pi
        steps = max(2, int(abs(da) / 0.25) + 1)
        for s in range(steps + 1):
            a = a1 + da * s / steps
            pts.append((p_cur[0] + cr * math.cos(a), p_cur[1] + cr * math.sin(a)))
    d.polygon([(q(x), q(y)) for (x, y) in pts], fill=255)


def _pentagon(d, cx, cy, r, rot=-math.pi / 2):
    pts = []
    for i in range(5):
        a = rot + i * 2 * math.pi / 5
        pts.append((q(cx + r * math.cos(a)), q(cy + r * math.sin(a))))
    d.polygon(pts, fill=255)


def _drop(d, cx, cy, r):
    # 水滴：下半圆 + 上方尖角，平滑闭合曲线（1 个尖点）。
    d.ellipse([q(cx - r), q(cy - r), q(cx + r), q(cy + r)], fill=255)
    d.polygon([(q(cx - r * 0.5), q(cy - r * 0.2)),
               (q(cx), q(cy - r * 1.8)),
               (q(cx + r * 0.5), q(cy - r * 0.2))], fill=255)


def _chevron(d, cx, cy, s):
    # 双雪佛龙箭头（尖角多边形，应保持锐利不被圆化）。
    d.polygon([(q(cx - s), q(cy - s * 0.5)), (q(cx), q(cy - s)),
               (q(cx + s), q(cy - s * 0.5)), (q(cx + s * 0.4), q(cy)),
               (q(cx + s), q(cy + s * 0.5)), (q(cx), q(cy + s)),
               (q(cx - s), q(cy + s * 0.5)), (q(cx - s * 0.4), q(cy))], fill=255)


def _rings(d, cx, cy, rs):
    # 同心环：每环 = 白盘 - 内黑盘。
    for r in rs:
        d.ellipse([q(cx - r), q(cy - r), q(cx + r), q(cy + r)], fill=255)
        d.ellipse([q(cx - r + 10), q(cy - r + 10), q(cx + r - 10), q(cy + r - 10)], fill=0)


def build():
    imgs = {}

    # 1) ic_app
    c = canvas()
    c = add(c, lambda d: _roundrect(d, 12, 12, 176, 176, 42), vgrad((88, 101, 242), (173, 109, 244)))
    c = add(c, lambda d: d.ellipse([q(58), q(58), q(142), q(142)], fill=255), (255, 255, 255, 255))
    imgs["ic_app"] = c

    # 2) ic_ring
    c = canvas()
    def ring(d):
        d.ellipse([q(20), q(20), q(180), q(180)], fill=255)
        d.ellipse([q(66), q(66), q(134), q(134)], fill=0)
    c = add(c, ring, (34, 182, 160, 255))
    imgs["ic_ring"] = c

    # 3) ic_star
    c = canvas()
    c = add(c, lambda d: _star(d, 100, 100, 86, 34), (241, 196, 15, 255))
    imgs["ic_star"] = c

    # 4) ic_gear
    c = canvas()
    def gear(d):
        _gear(d, 100, 100, 80, 60, teeth=8)
        d.ellipse([q(74), q(74), q(126), q(126)], fill=0)
    c = add(c, gear, (90, 99, 120, 255))
    imgs["ic_gear"] = c

    # 5) ic_heart
    c = canvas()
    c = add(c, lambda d: _heart(d, 100, 108, 5.0), (231, 76, 60, 255))
    imgs["ic_heart"] = c

    # 6) ic_bubble
    c = canvas()
    def bubble(d):
        _roundrect(d, 28, 32, 144, 96, 26)
        d.polygon([(q(58), q(124)), (q(58), q(150)), (q(92), q(124))], fill=255)
    c = add(c, bubble, (52, 152, 219, 255))
    imgs["ic_bubble"] = c

    # 7) ic_octagon
    c = canvas()
    def oct(d):
        pts = []
        for i in range(8):
            a = math.pi / 8 + i * math.pi / 4
            pts.append((q(100 + 82 * math.cos(a)), q(100 + 82 * math.sin(a))))
        d.polygon(pts, fill=255)
    c = add(c, oct, (231, 76, 60, 255))
    imgs["ic_octagon"] = c

    # 8) ic_play
    c = canvas()
    c = add(c, lambda d: d.ellipse([q(36), q(36), q(164), q(164)], fill=255), (44, 62, 80, 255))
    c = add(c, lambda d: d.polygon([(q(84), q(74)), (q(84), q(126)), (q(130), q(100))], fill=255),
            (255, 255, 255, 255))
    imgs["ic_play"] = c

    # 9) ic_check
    c = canvas()
    c = add(c, lambda d: _roundrect(d, 28, 28, 144, 144, 34), (46, 204, 113, 255))
    def check(d):
        d.line([(q(70), q(100)), (q(92), q(122)), (q(134), q(78))], fill=255, width=q(14), joint="curve")
    c = add(c, check, (255, 255, 255, 255))
    imgs["ic_check"] = c

    # 10) ic_cloud
    c = canvas()
    c = add(c, lambda d: _cloud(d, 100, 100), (236, 240, 241, 255))
    imgs["ic_cloud"] = c

    # 11) ic_ellipse —— 纯椭圆（无旋转，测 fit_rotated_ellipse 分支）
    c = canvas()
    c = add(c, lambda d: _ellipse(d, 100, 100, 82, 52), (155, 89, 182, 255))
    imgs["ic_ellipse"] = c

    # 12) ic_capsule —— 胶囊/体育场形（2 半圆 + 2 直边，测弧-线-弧干净衔接）
    c = canvas()
    c = add(c, lambda d: _capsule(d, 100, 100, 150, 80), (52, 152, 219, 255))
    imgs["ic_capsule"] = c

    # 13) ic_rhex —— 圆角六边形（直边 + 圆角，测圆角多边形分支）
    c = canvas()
    c = add(c, lambda d: _roundpoly(d, 100, 100, 82, 6, 22), (241, 196, 15, 255))
    imgs["ic_rhex"] = c

    # 14) ic_pentagon —— 尖角正五边形（多边形，应保持锐利）
    c = canvas()
    c = add(c, lambda d: _pentagon(d, 100, 100, 84), (231, 76, 60, 255))
    imgs["ic_pentagon"] = c

    # 15) ic_drop —— 水滴（平滑曲线 + 1 尖点）
    c = canvas()
    c = add(c, lambda d: _drop(d, 100, 108, 60), (230, 126, 34, 255))
    imgs["ic_drop"] = c

    # 16) ic_chevron —— 双雪佛龙箭头（尖角多边形）
    c = canvas()
    c = add(c, lambda d: _chevron(d, 100, 100, 44), (46, 204, 113, 255))
    imgs["ic_chevron"] = c

    # 17) ic_rings3 —— 三同心环
    c = canvas()
    c = add(c, lambda d: _rings(d, 100, 100, [80, 55, 30]), (149, 165, 166, 255))
    imgs["ic_rings3"] = c

    for name, im in imgs.items():
        out = im.resize((N, N), Image.LANCZOS)
        path = os.path.join(OUT, f"{name}.png")
        out.save(path)
        print("wrote", path)


if __name__ == "__main__":
    build()
    print("done:", len(imgs), "icons in", OUT)
