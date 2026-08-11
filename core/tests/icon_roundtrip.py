#!/usr/bin/env python3
"""icon 设计 PNG -> png2svg 还原 -> resvg 渲染回 PNG -> 与原图对比。

对每个 tests/gen_icons/in/*.png：
  in.png --(png2svg)--> out/<name>.svg
  out/<name>.svg --(resvg)--> out/<name>_rec.png
  in.png  vs  _rec.png  -> 白底合成比较 mean/max/alpha/ssim + diff 图

生成 tests/gen_icons/report.html（原图 | 还原渲染 | diff | svg 路径统计）。

依赖：Node @resvg/resvg-js；Pillow + numpy（PYLIBS）。
"""
import glob
import html
import os
import subprocess
import sys

_PYLIBS = os.environ.get("PYLIBS", "")
if _PYLIBS and os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen_icons")
IN_DIR = os.path.join(GEN, "in")
OUT_DIR = os.path.join(GEN, "out")
os.makedirs(OUT_DIR, exist_ok=True)

NODE = "C:/Users/Administrator/.workbuddy/binaries/node/versions/22.22.2/node.exe"
NODE_PATH = "C:/Users/Administrator/.workbuddy/binaries/node/workspace/node_modules"
RENDER_JS = os.path.join(HERE, "render_svg.js")
PNG2SVG_BIN = os.environ.get("PNG2SVG_BIN",
    "E:/AI-workspace/png2svg/png2svg/td20260810k/debug/png2svg.exe")

MEAN_OK = 8.0
MEAN_WARN = 20.0


def _gaussian_1d(sigma=1.5, size=11):
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    k = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


_KG = _gaussian_1d()


def _sepconv(x, k):
    p = len(k) // 2
    xp = np.pad(x, ((p, p), (0, 0)), mode="reflect")
    w = np.lib.stride_tricks.sliding_window_view(xp, len(k), axis=0)
    y0 = np.tensordot(w, k, axes=([2], [0]))
    xp2 = np.pad(y0, ((0, 0), (p, p)), mode="reflect")
    w2 = np.lib.stride_tricks.sliding_window_view(xp2, len(k), axis=1)
    return np.tensordot(w2, k, axes=([2], [0]))


def ssim_np(ref, out, L=255.0):
    k1, k2 = 0.01, 0.03
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    tot = 0.0
    for ch in range(3):
        x = ref[:, :, ch]
        y = out[:, :, ch]
        mx, my = _sepconv(x, _KG), _sepconv(y, _KG)
        sx2 = _sepconv(x * x, _KG) - mx * mx
        sy2 = _sepconv(y * y, _KG) - my * my
        sxy = _sepconv(x * y, _KG) - mx * my
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx * mx + my * my + c1) * (sx2 + sy2 + c2)
        tot += float(np.mean(num / np.where(den == 0, 1e-12, den)))
    return tot / 3.0


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
    n_path = svg_text.count("<path")
    n_arc = svg_text.count(" A ")
    return ",".join(kinds) if kinds else "solid/polygon", n_path, n_arc


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
    ref_a = np.asarray(ref.convert("RGBA").split()[3], dtype=np.float64)
    out_a = np.asarray(out.convert("RGBA").split()[3], dtype=np.float64)
    if out_a.shape != ref_a.shape:
        out_a = np.asarray(out.resize(ref.size).convert("RGBA").split()[3], dtype=np.float64)
    alpha_diff = float(np.abs(ref_a - out_a).mean())
    diff_gray = np.clip(diff.mean(axis=2) * 6.0, 0, 255).astype(np.uint8)
    return mean_diff, max_diff, alpha_diff, float(s), diff_gray


def main():
    cases = sorted(glob.glob(os.path.join(IN_DIR, "*.png")))
    if not cases:
        print("no icons found in", IN_DIR)
        sys.exit(1)
    print(f"png2svg bin: {PNG2SVG_BIN}")
    print(f"found {len(cases)} icons\n")
    rows = []
    for in_png in cases:
        name = os.path.splitext(os.path.basename(in_png))[0]
        out_svg = os.path.join(OUT_DIR, f"{name}.svg")
        rec_png = os.path.join(OUT_DIR, f"{name}_rec.png")
        diff_png = os.path.join(OUT_DIR, f"{name}_diff.png")
        print(f"=== {name} ===")
        try:
            convert_png(in_png, out_svg)
            render_svg(out_svg, rec_png)
            with open(out_svg, "r", encoding="utf-8") as f:
                svg_text = f.read()
            kinds, n_path, n_arc = detect_types(svg_text)
            mean_diff, max_diff, alpha_diff, s, diff_gray = analyze(in_png, rec_png)
            Image.fromarray(diff_gray, "L").save(diff_png)
            if mean_diff < MEAN_OK:
                status = "OK"
            elif mean_diff < MEAN_WARN:
                status = "WARN"
            else:
                status = "FAIL"
            size = os.path.getsize(out_svg)
            print(f"  detected={kinds} paths={n_path} arcs={n_arc} svg={size}B "
                  f"mean={mean_diff:.2f} max={max_diff} alpha={alpha_diff:.2f} ssim={s:.4f} -> {status}")
            rows.append(dict(name=name, kinds=kinds, n_path=n_path, n_arc=n_arc, size=size,
                             mean=mean_diff, max=max_diff, alpha=alpha_diff, ssim=s,
                             status=status))
        except Exception as e:
            print("  ERROR:", repr(e))
            rows.append(dict(name=name, kinds="ERROR", n_path=0, n_arc=0, size=0,
                             mean=0, max=0, alpha=0, ssim=0, status="ERROR", err=str(e)))

    write_report(rows)
    print("\n--- summary ---")
    for r in rows:
        print(f"  {r['name']:<10} {r['status']:<5} {r['kinds']:<12} "
              f"paths={r['n_path']} arcs={r['n_arc']} mean={r['mean']:.2f} ssim={r['ssim']:.4f}")
    n_fail = sum(1 for r in rows if r["status"] in ("FAIL", "ERROR"))
    print(f"\n{len(rows)} icons, {n_fail} failing.")


def write_report(rows):
    def td(x):
        return f"<td>{html.escape(str(x))}</td>"
    body = ["<h1>icon 设计 PNG -> png2svg 还原 报告</h1>",
            "<p>in.png（设计 PNG） --&gt; png2svg --&gt; out.svg --&gt; resvg 渲染 _rec.png --&gt; 与 in.png 白底对比。</p>",
            "<table border='1' cellpadding='4' cellspacing='0'>",
            "<tr><th>icon</th><th>判定</th><th>path</th><th>arc</th><th>svg</th>"
            "<th>mean</th><th>max</th><th>alpha</th><th>ssim</th><th>状态</th>"
            "<th>原图</th><th>还原渲染</th><th>diff</th><th>out.svg 路径</th></tr>"]
    for r in rows:
        n = r["name"]
        color = {"OK": "green", "WARN": "orange", "FAIL": "red", "ERROR": "red"}.get(r["status"], "black")
        body.append("<tr>")
        body.append(f"<td>{n}</td>")
        body.append(td(r["kinds"]))
        body.append(td(r["n_path"]))
        body.append(td(r["n_arc"]))
        body.append(td(f"{r['size']}B"))
        body.append(td(f"{r['mean']:.2f}"))
        body.append(td(r["max"]))
        body.append(td(f"{r['alpha']:.2f}"))
        body.append(td(f"{r['ssim']:.4f}"))
        body.append(f"<td style='color:{color};font-weight:bold'>{r['status']}</td>")
        body.append(f"<td><img src='../in/{n}.png' width='110'/></td>")
        body.append(f"<td><img src='{n}_rec.png' width='110'/></td>")
        body.append(f"<td><img src='{n}_diff.png' width='110'/></td>")
        # svg path text (collapse, show first 600 chars)
        svg_path = f"{n}.svg"
        body.append(f"<td><a href='{svg_path}'>svg</a></td>")
        body.append("</tr>")
    body.append("</table>")
    body.append(f"<p>mean&lt;{MEAN_OK} OK；&lt;{MEAN_WARN} WARN；否则 FAIL。diff 为白底差放大 6×。</p>")
    out = os.path.join(OUT_DIR, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
                "<title>icon 还原报告</title></head><body>" +
                "\n".join(body) + "</body></html>")
    print("report ->", out)


if __name__ == "__main__":
    main()
