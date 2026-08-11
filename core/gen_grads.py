"""生成三类渐变测试图（均带透明背景，模拟真实 icon）：
    linear.png  : 圆盘内水平线性渐变（红 -> 蓝）
    radial.png  : 圆盘内径向渐变（白 -> 深蓝）
    mesh.png    : 圆盘内四角双线性（网格）渐变
透明背景保证工具有 alpha 通道、按形状整体取前景，梯度不被当成背景切掉。
"""
from PIL import Image

SIZE = 200
CX = CY = SIZE / 2.0
R = 88.0


def in_disk(x, y):
    return ((x - CX) ** 2 + (y - CY) ** 2) <= R * R


def make_linear():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    px = img.load()
    for y in range(SIZE):
        for x in range(SIZE):
            if not in_disk(x, y):
                continue
            # 水平线性渐变：左红(255,40,40) -> 右蓝(40,80,255)
            t = x / (SIZE - 1)
            r = int(255 * (1 - t) + 40 * t)
            g = int(40 * (1 - t) + 80 * t)
            b = int(40 * (1 - t) + 255 * t)
            px[x, y] = (r, g, b, 255)
    # 对照纯色小方块
    for y in range(150, 185):
        for x in range(20, 55):
            px[x, y] = (30, 200, 120, 255)
    img.save("linear.png")
    print("wrote linear.png")


def make_radial():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    px = img.load()
    for y in range(SIZE):
        for x in range(SIZE):
            if not in_disk(x, y):
                continue
            dx = x - CX
            dy = y - CY
            rr = (dx * dx + dy * dy) ** 0.5
            t = rr / R
            cr = int(255 * (1 - t) + 20 * t)
            cg = int(255 * (1 - t) + 40 * t)
            cb = int(255 * (1 - t) + 140 * t)
            px[x, y] = (cr, cg, cb, 255)
    img.save("radial.png")
    print("wrote radial.png")


def make_mesh():
    # 用方块（而非圆盘）承载四角网格渐变：圆盘会切掉四角，使双线性 xy 项贡献变小、
    # 被误判为线性；方块才能完整体现“四个角颜色各不同”的网格特征。
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    px = img.load()
    x0, x1, y0, y1 = 25, 175, 25, 175
    tl = (230, 40, 40)     # 左上 红
    tr = (240, 210, 40)    # 右上 黄
    bl = (40, 120, 230)    # 左下 蓝
    br = (40, 200, 120)    # 右下 绿
    for y in range(SIZE):
        for x in range(SIZE):
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            u = (x - x0) / (x1 - x0)
            v = (y - y0) / (y1 - y0)

            def lerp(a, b, t):
                return a * (1 - t) + b * t

            top = tuple(lerp(tl[c], tr[c], u) for c in range(3))
            bot = tuple(lerp(bl[c], br[c], u) for c in range(3))
            col = tuple(int(lerp(top[c], bot[c], v)) for c in range(3))
            px[x, y] = (col[0], col[1], col[2], 255)
    img.save("mesh.png")
    print("wrote mesh.png")


if __name__ == "__main__":
    make_linear()
    make_radial()
    make_mesh()
