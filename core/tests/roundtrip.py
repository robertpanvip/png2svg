#!/usr/bin/env python3
"""png2svg 往返测试框架。

流程（对每个参考 SVG）：
  参考 SVG --(resvg 渲染)--> 参考 PNG
  参考 PNG --(png2svg 还原)--> 输出 SVG
  输出 SVG --(resvg 渲染)--> 输出 PNG
  参考 PNG  vs  输出 PNG  ->  在白底上合成后算 MSE / 最大差 / alpha差 / SSIM，并生成 diff 图。

依赖：
  - Node + @resvg/resvg-js（tests/render_svg.js）做 SVG->PNG
  - png2svg 可执行文件（环境变量 PNG2SVG_BIN，默认 td20260808u/debug/png2svg.exe）
  - Python: Pillow + scikit-image

运行:
  PNG2SVG_BIN=td20260808u/debug/png2svg.exe python tests/roundtrip.py
结果:
  tests/out/report.html  +  各用例的 _ref.png / _out.svg / _out.png / _diff.png
"""
import glob
import html
import os
import subprocess
import sys

# 托管 Python 以隔离模式运行，忽略 PYTHONPATH；通过 PYLIBS 显式注入依赖目录。
_PYLIBS = os.environ.get("PYLIBS", "")
if _PYLIBS and os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)

import numpy as np
from PIL import Image


def _gaussian_1d(sigma=1.5, size=11):
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    k = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


_KG = _gaussian_1d()


def _sepconv(x, k):
    """可分离高斯模糊（反射边界），无需 scipy。"""
    p = len(k) // 2
    xp = np.pad(x, ((p, p), (0, 0)), mode="reflect")
    w = np.lib.stride_tricks.sliding_window_view(xp, len(k), axis=0)
    y0 = np.tensordot(w, k, axes=([2], [0]))
    xp2 = np.pad(y0, ((0, 0), (p, p)), mode="reflect")
    w2 = np.lib.stride_tricks.sliding_window_view(xp2, len(k), axis=1)
    return np.tensordot(w2, k, axes=([2], [0]))


def ssim_np(ref, out, L=255.0):
    """单尺度 SSIM（高斯窗，逐通道取均值）。ref/out: (H,W,3) float64。"""
    k1, k2 = 0.01, 0.03
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    tot = 0.0
    for ch in range(3):
        x = ref[:, :, ch]
        y = out[:, :, ch]
        mx, my = _sepconv(x, _KG), _sepconv(y, _KG)
        mx2, my2 = _sepconv(x * x, _KG), _sepconv(y * y, _KG)
        mxy = _sepconv(x * y, _KG)
        sx2 = mx2 - mx * mx
        sy2 = my2 - my * my
        sxy = mxy - mx * my
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx * mx + my * my + c1) * (sx2 + sy2 + c2)
        tot += float(np.mean(num / np.where(den == 0, 1e-12, den)))
    return tot / 3.0

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF_DIR = os.path.join(HERE, "ref")
OUT_DIR = os.path.join(HERE, "out")

NODE = "C:/Users/Administrator/.workbuddy/binaries/node/versions/22.22.2/node.exe"
NODE_PATH = "C:/Users/Administrator/.workbuddy/binaries/node/workspace/node_modules"
RENDER_JS = os.path.join(HERE, "render_svg.js")
PNG2SVG_BIN = os.environ.get("PNG2SVG_BIN", os.path.join(ROOT, "td20260808u", "debug", "png2svg.exe"))

# 软判阈值（以白底合成后的平均像素差为主；SSIM 受抗锯齿边缘影响大，仅作参考）
MEAN_OK = 8.0     # 平均差 < 8 -> 视为良好（形状/渐变基本无损还原）
MEAN_WARN = 20.0  # 8~20 -> 需关注；> 20 -> 失败


def run(cmd, **kw):
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def render_svg(in_svg, out_png):
    env = os.environ.copy()
    env["NODE_PATH"] = NODE_PATH
    env["PATH"] = NODE + ";" + env.get("PATH", "")
    run([NODE, RENDER_JS, in_svg, out_png], env=env)


def convert_png(png_in, svg_out):
    run([PNG2SVG_BIN, png_in, "-o", svg_out])


def detect_types(svg_text):
    kinds = []
    if "radialGradient" in svg_text:
        kinds.append("radial")
    if "linearGradient" in svg_text:
        kinds.append("linear")
    if "clip-path" in svg_text or "clipPath" in svg_text:
        kinds.append("mesh")
    n_path = svg_text.count("<path")
    return ",".join(kinds) if kinds else "solid/polygon", n_path


def composite_white(img):
    img = img.convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, img).convert("RGB")


def analyze(ref_png, out_png):
    ref = Image.open(ref_png)
    out = Image.open(out_png)
    if ref.size != out.size:
        out = out.resize(ref.size)
    ref_rgb = np.asarray(composite_white(ref), dtype=np.float64)
    out_rgb = np.asarray(composite_white(out), dtype=np.float64)
    diff = np.abs(ref_rgb - out_rgb)
    mean_diff = float(diff.mean())
    max_diff = int(diff.max())
    s = ssim_np(ref_rgb, out_rgb)

    # alpha 通道差异
    ref_a = np.asarray(ref.convert("RGBA").split()[3], dtype=np.float64)
    out_a = np.asarray(out.convert("RGBA").split()[3], dtype=np.float64)
    if out_a.shape != ref_a.shape:
        out_a = np.asarray(out.resize(ref.size).convert("RGBA").split()[3], dtype=np.float64)
    alpha_diff = float(np.abs(ref_a - out_a).mean())

    # 生成 diff 图（灰度放大，便于肉眼看）
    diff_gray = np.clip(diff.mean(axis=2) * 6.0, 0, 255).astype(np.uint8)
    return mean_diff, max_diff, alpha_diff, float(s), diff_gray


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = sorted(glob.glob(os.path.join(REF_DIR, "*.svg")))
    if not cases:
        print("no reference svgs found in", REF_DIR)
        sys.exit(1)
    print(f"png2svg bin: {PNG2SVG_BIN}")
    print(f"found {len(cases)} reference cases\n")

    rows = []
    for ref_svg in cases:
        name = os.path.splitext(os.path.basename(ref_svg))[0]
        ref_png = os.path.join(OUT_DIR, f"{name}_ref.png")
        out_svg = os.path.join(OUT_DIR, f"{name}_out.svg")
        out_png = os.path.join(OUT_DIR, f"{name}_out.png")
        diff_png = os.path.join(OUT_DIR, f"{name}_diff.png")
        print(f"=== {name} ===")
        try:
            render_svg(ref_svg, ref_png)
            convert_png(ref_png, out_svg)
            render_svg(out_svg, out_png)
            with open(out_svg, "r", encoding="utf-8") as f:
                svg_text = f.read()
            kinds, n_path = detect_types(svg_text)
            mean_diff, max_diff, alpha_diff, s, diff_gray = analyze(ref_png, out_png)
            Image.fromarray(diff_gray, "L").save(diff_png)
            if mean_diff < MEAN_OK:
                status = "OK"
            elif mean_diff < MEAN_WARN:
                status = "WARN"
            else:
                status = "FAIL"
            size = os.path.getsize(out_svg)
            print(f"  detected={kinds} paths={n_path} svg={size}B "
                  f"mean={mean_diff:.2f} max={max_diff} alpha={alpha_diff:.2f} ssim={s:.4f} -> {status}")
            rows.append(dict(name=name, kinds=kinds, n_path=n_path, size=size,
                             mean=mean_diff, max=max_diff, alpha=alpha_diff, ssim=s,
                             status=status))
        except Exception as e:
            print("  ERROR:", repr(e))
            rows.append(dict(name=name, kinds="ERROR", n_path=0, size=0,
                             mean=0, max=0, alpha=0, ssim=0, status="ERROR",
                             err=str(e)))

    write_report(rows)
    # 汇总
    print("\n--- summary ---")
    for r in rows:
        print(f"  {r['name']:<10} {r['status']:<5} detected={r['kinds']:<12} "
              f"mean={r['mean']:.2f} ssim={r['ssim']:.4f}")
    n_fail = sum(1 for r in rows if r["status"] in ("FAIL", "ERROR"))
    print(f"\n{len(rows)} cases, {n_fail} failing.")


def write_report(rows):
    def td(x):
        return f"<td>{html.escape(str(x))}</td>"
    body = ["<h1>png2svg 往返测试报告</h1>",
            f"<p>参考 SVG -> resvg 渲染 PNG -> png2svg 还原 SVG -> resvg 渲染 PNG -> 与参考对比（白底合成）。</p>",
            "<table border='1' cellpadding='4' cellspacing='0'>",
            "<tr><th>用例</th><th>判定类型</th><th>路径数</th><th>out.svg</th>"
            "<th>mean差</th><th>max差</th><th>alpha差</th><th>SSIM</th><th>状态</th>"
            "<th>参考</th><th>输出</th><th>diff</th></tr>"]
    for r in rows:
        n = r["name"]
        color = {"OK": "green", "WARN": "orange", "FAIL": "red", "ERROR": "red"}.get(r["status"], "black")
        body.append("<tr>")
        body.append(f"<td>{n}</td>")
        body.append(td(r["kinds"]))
        body.append(td(r["n_path"]))
        body.append(td(f"{r['size']}B"))
        body.append(td(f"{r['mean']:.2f}"))
        body.append(td(r["max"]))
        body.append(td(f"{r['alpha']:.2f}"))
        body.append(td(f"{r['ssim']:.4f}"))
        body.append(f"<td style='color:{color};font-weight:bold'>{r['status']}</td>")
        body.append(f"<td><img src='{n}_ref.png' width='120'/></td>")
        body.append(f"<td><img src='{n}_out.png' width='120'/></td>")
        body.append(f"<td><img src='{n}_diff.png' width='120'/></td>")
        body.append("</tr>")
    body.append("</table>")
    body.append("<p>判定：白底合成后 mean差&lt;%g 为 OK；&lt;%g 为 WARN；否则 FAIL。"
                "SSIM 受抗锯齿边缘影响偏大，仅作参考。diff 为白底合成差放大 6×。</p>"
                % (MEAN_OK, MEAN_WARN))
    out = os.path.join(OUT_DIR, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
                "<title>png2svg 往返测试</title></head><body>" +
                "\n".join(body) + "</body></html>")
    print("report ->", out)


if __name__ == "__main__":
    main()
