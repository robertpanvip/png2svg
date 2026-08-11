"""生成一张混合形状测试图：星形(尖角) + 圆(平滑曲线) + 方块(直角)。"""
from PIL import Image, ImageDraw
import math

W = 256
img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 星形（5 角，明显尖角）
cx, cy = 80, 80
pts = []
for i in range(10):
    r = 60 if i % 2 == 0 else 25
    a = -math.pi / 2 + i * math.pi / 5
    pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
d.polygon(pts, fill=(220, 60, 60, 255))

# 圆（平滑曲线）
d.ellipse([190 - 50, 80 - 50, 190 + 50, 80 + 50], fill=(60, 120, 220, 255))

# 方块（直角）
d.rectangle([40, 170, 90, 220], fill=(60, 190, 120, 255))

img.save("star.png")
print("saved star.png")
