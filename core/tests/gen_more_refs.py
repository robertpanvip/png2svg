#!/usr/bin/env python3
"""生成 png2svg 的扩展回归参考图，写入 tests/ref/。

覆盖：
  Bug1 斜线毛刺 —— 多角度斜矩形 / 平行四边形 / 正多边形（斜边）
  Bug2 倒角→圆 —— 多半径圆角矩形 / 胶囊 / 旋转圆角矩形 / 半圆 / 圆环
  圆椭圆精度 —— 小圆 / 旋转椭圆
  复合/渐变 —— 复合图标 / 45°线性渐变 / 偏心径向渐变
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
os.makedirs(REF, exist_ok=True)

W = H = 200


def write(name, body):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">\n{body}\n</svg>\n'
    )
    with open(os.path.join(REF, name), "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", name)


def poly(cx, cy, r, n, rot_deg=-90):
    pts = []
    for i in range(n):
        a = math.radians(rot_deg + 360.0 * i / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


# ---------------- Bug1：斜线毛刺（斜边应力） ----------------
# 多角度旋转矩形
for deg in (15, 45):
    write(f"t_diag{deg}.svg",
          f'<rect x="40" y="80" width="120" height="40" fill="#3366cc" '
          f'transform="rotate({deg} 100 100)"/>')
write("t_diagneg.svg",
      '<rect x="40" y="80" width="120" height="40" fill="#3366cc" '
      'transform="rotate(-30 100 100)"/>')

# 平行四边形（skewX，全部斜边）
write("t_parallelogram.svg",
      '<g transform="translate(100,100) skewX(30) translate(-100,-100)">'
      '<rect x="40" y="60" width="120" height="80" fill="#27ae60"/></g>')

# 正多边形（斜边）
write("t_triangle.svg", f'<polygon points="{poly(100, 100, 72, 3)}" fill="#e67e22"/>')
write("t_pentagon.svg", f'<polygon points="{poly(100, 100, 72, 5)}" fill="#8e44ad"/>')
write("t_octagon.svg", f'<polygon points="{poly(100, 100, 74, 8, 22.5)}" fill="#16a085"/>')

# ---------------- Bug2：倒角→圆（多半径/旋转） ----------------
write("t_chamfer_s.svg",
      '<rect x="50" y="40" width="100" height="120" rx="8" ry="8" fill="#e0333c"/>')
write("t_chamfer_l.svg",
      '<rect x="50" y="40" width="100" height="120" rx="50" ry="50" fill="#e0333c"/>')
write("t_pill.svg",
      '<rect x="30" y="70" width="140" height="60" rx="30" ry="30" fill="#e74c3c"/>')
write("t_chamfer_rot.svg",
      '<rect x="50" y="40" width="100" height="120" rx="24" ry="24" fill="#e0333c" '
      'transform="rotate(35 100 100)"/>')

# 半圆（弧 + 直径）
write("t_semicircle.svg",
      '<path d="M 40 100 A 60 60 0 0 1 160 100 Z" fill="#2980b9"/>')

# 圆环（evenodd + 双弧 + 孔洞）
write("t_ring.svg",
      '<path fill-rule="evenodd" '
      'd="M 100 40 A 60 60 0 1 0 100 160 A 60 60 0 1 0 100 40 Z '
      'M 100 70 A 30 30 0 1 1 100 130 A 30 30 0 1 1 100 70 Z" fill="#9b59b6"/>')

# ---------------- 圆/椭圆精度 ----------------
write("t_circle_s.svg",
      '<circle cx="100" cy="100" r="16" fill="#c0392b"/>')
write("t_ellipse_rot.svg",
      '<ellipse cx="100" cy="100" rx="70" ry="35" fill="#d35400" '
      'transform="rotate(30 100 100)"/>')

# ---------------- 复合图标 ----------------
icon = (
    '<circle cx="60" cy="55" r="35" fill="#e74c3c"/>'
    '<rect x="110" y="30" width="70" height="50" rx="12" ry="12" fill="#3498db"/>'
    '<polygon points="50,170 100,100 150,170" fill="#2ecc71"/>'
)
write("t_icon1.svg", icon)

# ---------------- 渐变变体 ----------------
write("t_linear2.svg",
      '<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">'
      '<stop offset="0" stop-color="#ff7e5f"/>'
      '<stop offset="1" stop-color="#feb47b"/></linearGradient></defs>'
      '<rect x="40" y="40" width="120" height="120" fill="url(#lg)"/>')

write("t_radial2.svg",
      '<defs><radialGradient id="rg" cx="0.35" cy="0.35" r="0.7">'
      '<stop offset="0" stop-color="#ffffff"/>'
      '<stop offset="1" stop-color="#2980b9"/></radialGradient></defs>'
      '<circle cx="100" cy="100" r="70" fill="url(#rg)"/>')

print("\n Done. New reference SVGs written to", REF)
