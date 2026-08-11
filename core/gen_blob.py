"""生成非椭圆的平滑曲线（三叶 blob），用于验证贝塞尔回退分支仍生效。
blob 不是圆/椭圆（半径随角度呈 3 次谐波变化），圆/椭圆拟合应失败 → 回退为贝塞尔曲线。
"""
from PIL import Image, ImageDraw
import math

S = 4  # 超采样倍数（抗锯齿）
W = H = 400
img = Image.new('RGBA', (W * S, H * S), (255, 255, 255, 0))
d = ImageDraw.Draw(img)
cx, cy = W * S / 2.0, H * S / 2.0
R = 140.0 * S
pts = []
for i in range(720):
    th = 2.0 * math.pi * i / 720.0
    r = R + 26.0 * S * math.sin(3.0 * th) + 14.0 * S * math.sin(2.0 * th)
    x = cx + r * math.cos(th)
    y = cy + r * math.sin(th)
    pts.append((x, y))
d.polygon(pts, fill=(0, 0, 0, 255))
img = img.resize((W, H), Image.LANCZOS)
img.save('blob.png')
print('wrote blob.png', img.size)
