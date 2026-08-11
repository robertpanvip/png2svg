#!/usr/bin/env python3
"""生成往返测试的参考 SVG fixtures（icon 风格、透明背景）。

每个用例覆盖一类特征：
  t_solid    纯色形状（圆/方/三角）
  t_linear   线性渐变（userSpaceOnUse 水平）
  t_radial   径向渐变（白→深蓝）
  t_mesh     网格（双线性）渐变（烘焙成 32x32 颜色网格）
  t_circle   实心圆（应被识别为真弧）
  t_ellipse  旋转 30° 的椭圆（应被识别为旋转椭圆弧）
  t_bezier   平滑闭合 blob（应被贝塞尔重建）
  t_star     星形多边形（尖角保留）
  t_shadow   半透明椭圆（应被识别为阴影）
  t_mixed    混合多区域（线性带 + 纯色圆 + 星形 + 径向圆）

运行: python tests/gen_refs.py  -> 写到 tests/ref/*.svg
"""
import math
import os
import xml.etree.ElementTree as ET

ET.register_namespace("", "http://www.w3.org/2000/svg")

W = H = 200
REF_DIR = os.path.join(os.path.dirname(__file__), "ref")
os.makedirs(REF_DIR, exist_ok=True)


def rgb(r, g, b):
    return f"rgb({r},{g},{b})"


def root(extra_attrs=None):
    attrs = {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(W),
        "height": str(H),
        "viewBox": f"0 0 {W} {H}",
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return ET.Element("svg", attrs)


def save(svg, name):
    path = os.path.join(REF_DIR, name)
    ET.ElementTree(svg).write(path, xml_declaration=True, encoding="utf-8")
    print("wrote", path)


def star_points(cx, cy, r_out, r_in, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / n
        pts.append(f"{cx + r * math.cos(a):.2f},{cy + r * math.sin(a):.2f}")
    return " ".join(pts)


# 1) 纯色
def t_solid():
    s = root()
    ET.SubElement(s, "circle", {"cx": "60", "cy": "65", "r": "38", "fill": rgb(226, 59, 59)})
    ET.SubElement(s, "rect", {"x": "108", "y": "30", "width": "72", "height": "72", "fill": rgb(47, 111, 226)})
    ET.SubElement(s, "polygon", {"points": "35,150 95,150 65,192", "fill": rgb(47, 175, 85)})
    save(s, "t_solid.svg")


# 2) 线性渐变
def t_linear():
    s = root()
    d = ET.SubElement(s, "defs")
    lg = ET.SubElement(d, "linearGradient", {
        "id": "lg", "gradientUnits": "userSpaceOnUse",
        "x1": "20", "y1": "100", "x2": "180", "y2": "100"})
    ET.SubElement(lg, "stop", {"offset": "0", "stop-color": rgb(255, 43, 43)})
    ET.SubElement(lg, "stop", {"offset": "1", "stop-color": rgb(43, 75, 255)})
    ET.SubElement(s, "rect", {"x": "20", "y": "40", "width": "160", "height": "120", "rx": "12", "fill": "url(#lg)"})
    save(s, "t_linear.svg")


# 3) 径向渐变
def t_radial():
    s = root()
    d = ET.SubElement(s, "defs")
    rg = ET.SubElement(d, "radialGradient", {
        "id": "rg", "gradientUnits": "userSpaceOnUse",
        "cx": "100", "cy": "100", "r": "90", "fx": "100", "fy": "100"})
    ET.SubElement(rg, "stop", {"offset": "0", "stop-color": rgb(255, 255, 255)})
    ET.SubElement(rg, "stop", {"offset": "1", "stop-color": rgb(20, 40, 140)})
    ET.SubElement(s, "circle", {"cx": "100", "cy": "100", "r": "90", "fill": "url(#rg)"})
    save(s, "t_radial.svg")


# 4) 网格（双线性）渐变 -> 烘焙成颜色网格
def t_mesh():
    s = root()
    N = 32
    cell = 160 // N
    x0, y0 = 20, 20
    TL = (255, 40, 40)
    TR = (255, 225, 40)
    BL = (40, 90, 255)
    BR = (40, 225, 120)

    def bil(u, v):
        out = []
        for k in range(3):
            c = (1 - u) * (1 - v) * TL[k] + u * (1 - v) * TR[k] + (1 - u) * v * BL[k] + u * v * BR[k]
            out.append(int(round(max(0, min(255, c)))))
        return out

    for i in range(N):
        for j in range(N):
            u = i / N
            v = j / N
            r, g, b = bil(u, v)
            ET.SubElement(s, "rect", {
                "x": str(x0 + i * cell), "y": str(y0 + j * cell),
                "width": str(cell), "height": str(cell),
                "fill": rgb(r, g, b)})
    save(s, "t_mesh.svg")


# 5) 实心圆
def t_circle():
    s = root()
    ET.SubElement(s, "circle", {"cx": "100", "cy": "100", "r": "72", "fill": rgb(43, 182, 160)})
    save(s, "t_circle.svg")


# 6) 旋转椭圆
def t_ellipse():
    s = root()
    ET.SubElement(s, "ellipse", {
        "cx": "100", "cy": "100", "rx": "86", "ry": "46",
        "transform": "rotate(30 100 100)", "fill": rgb(210, 105, 30)})
    save(s, "t_ellipse.svg")


# 7) 贝塞尔 blob
def t_bezier():
    s = root()
    path = ("M 100 35 "
            "C 140 35, 168 65, 165 105 "
            "C 162 150, 130 172, 95 170 "
            "C 55 168, 33 140, 38 100 "
            "C 43 60, 65 35, 100 35 Z")
    ET.SubElement(s, "path", {"d": path, "fill": rgb(142, 68, 173)})
    save(s, "t_bezier.svg")


# 8) 星形多边形
def t_star():
    s = root()
    ET.SubElement(s, "polygon", {"points": star_points(100, 100, 80, 32), "fill": rgb(241, 196, 15)})
    save(s, "t_star.svg")


# 9) 半透明阴影
def t_shadow():
    s = root()
    ET.SubElement(s, "ellipse", {
        "cx": "100", "cy": "100", "rx": "82", "ry": "52",
        "fill": rgb(0, 0, 0), "fill-opacity": "0.35"})
    save(s, "t_shadow.svg")


# 10) 混合多区域
def t_mixed():
    s = root()
    d = ET.SubElement(s, "defs")
    lgm = ET.SubElement(d, "linearGradient", {
        "id": "lgm", "gradientUnits": "userSpaceOnUse",
        "x1": "10", "y1": "45", "x2": "190", "y2": "45"})
    ET.SubElement(lgm, "stop", {"offset": "0", "stop-color": rgb(255, 91, 91)})
    ET.SubElement(lgm, "stop", {"offset": "1", "stop-color": rgb(91, 255, 143)})
    rgm = ET.SubElement(d, "radialGradient", {
        "id": "rgm", "gradientUnits": "userSpaceOnUse",
        "cx": "160", "cy": "150", "r": "30", "fx": "160", "fy": "150"})
    ET.SubElement(rgm, "stop", {"offset": "0", "stop-color": rgb(255, 255, 255)})
    ET.SubElement(rgm, "stop", {"offset": "1", "stop-color": rgb(122, 31, 156)})
    # 线性带
    ET.SubElement(s, "rect", {"x": "10", "y": "10", "width": "180", "height": "70", "rx": "8", "fill": "url(#lgm)"})
    # 纯色圆
    ET.SubElement(s, "circle", {"cx": "55", "cy": "140", "r": "32", "fill": rgb(47, 111, 226)})
    # 星形
    ET.SubElement(s, "polygon", {"points": star_points(112, 142, 30, 12), "fill": rgb(241, 196, 15)})
    # 径向圆
    ET.SubElement(s, "circle", {"cx": "160", "cy": "150", "r": "30", "fill": "url(#rgm)"})
    save(s, "t_mixed.svg")


if __name__ == "__main__":
    t_solid()
    t_linear()
    t_radial()
    t_mesh()
    t_circle()
    t_ellipse()
    t_bezier()
    t_star()
    t_shadow()
    t_mixed()
    print("done:", len(os.listdir(REF_DIR)), "files in", REF_DIR)
