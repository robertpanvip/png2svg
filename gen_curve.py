"""生成旋转椭圆（平滑曲线、非轴对齐）：用于验证 Catmull-Rom 平滑回退路径。"""
from PIL import Image, ImageDraw
import math

W = 256
img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
cx, cy, rx, ry, th = 128, 128, 90, 45, math.radians(30)
pts = []
for i in range(160):
    t = 2 * math.pi * i / 160
    x = rx * math.cos(t)
    y = ry * math.sin(t)
    xr = x * math.cos(th) - y * math.sin(th)
    yr = x * math.sin(th) + y * math.cos(th)
    pts.append((cx + xr, cy + yr))
d.polygon(pts, fill=(120, 90, 200, 255))
img.save("curve.png")
print("saved curve.png")
