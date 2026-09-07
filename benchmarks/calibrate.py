from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from svg.render import render_scene_resvg
from dataset.generator import GeneratorConfig, SceneGenerator
from dataset.renderer import SoftSVGRenderer
from benchmarks.metrics import compare_u8
from benchmarks.smoke_test import triptych


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--gammas", type=float, nargs="*", default=[0.005])
    ap.add_argument("--zetas", type=float, nargs="*", default=[0.1])
    ap.add_argument("--subpx", type=int, default=4)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--worst", type=int, default=4)
    ap.add_argument("--out", type=str, default="benchmarks/reports")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"generating {args.scenes} scenes and resvg refs ...")
    refs, scenes = [], []
    t0 = time.perf_counter()
    for i in range(args.scenes):
        gen = SceneGenerator(GeneratorConfig(), seed=args.seed0 + i)
        scene = gen.sample()
        scenes.append(scene)
        refs.append(render_scene_resvg(scene, args.size, args.size))
    print(f"  done in {time.perf_counter() - t0:.1f}s "
          f"(gen+resvg ≈ {args.scenes / max(time.perf_counter() - t0, 1e-9):.0f} scenes/s)")

    results = []
    for gamma in args.gammas:
        for zeta in args.zetas:
            rend = SoftSVGRenderer(out_size=args.size, gamma_px=gamma, zeta_px=zeta,
                                   sub_px=args.subpx, samples_per_curve=args.samples)
            mets, t_soft = [], []
            for scene, ref in zip(scenes, refs):
                t0 = time.perf_counter()
                soft = rend.to_uint8(rend.render_scene(scene))
                t_soft.append(time.perf_counter() - t0)
                mets.append(compare_u8(ref, soft))
            ssims = np.array([m["ssim"] for m in mets])
            psnrs = np.array([m["psnr"] for m in mets])
            maes = np.array([m["mae"] for m in mets])
            row = {
                "gamma_px": gamma, "zeta_px": zeta,
                "ssim_mean": float(ssims.mean()), "ssim_min": float(ssims.min()),
                "psnr_mean": float(psnrs.mean()), "psnr_min": float(psnrs.min()),
                "mae_mean": float(maes.mean()), "mae_max": float(maes.max()),
                "ms_per_scene": float(np.mean(t_soft) * 1000),
                "per_scene_ssim": [float(s) for s in ssims],
            }
            results.append(row)
            print(f"γ={gamma:.3f} ζ={zeta:.1f} | SSIM {row['ssim_mean']:.5f} (min {row['ssim_min']:.5f}) "
                  f"| PSNR {row['psnr_mean']:6.2f}dB | MAE {row['mae_mean']:.5f} | {row['ms_per_scene']:.1f}ms")

    best = max(results, key=lambda r: (r["ssim_mean"], r["ssim_min"]))
    print(f"\nbest: γ={best['gamma_px']:.3f} ζ={best['zeta_px']:.1f} "
          f"SSIM mean {best['ssim_mean']:.5f} min {best['ssim_min']:.5f}")

    target = 0.99
    if best["ssim_mean"] >= target and best["ssim_min"] >= target:
        verdict = f"PASS: SSIM >= {target} reached (mean and min)."
    elif best["ssim_mean"] >= target:
        verdict = (f"PARTIAL: mean SSIM >= {target} but worst scene {best['ssim_min']:.5f} below target; "
                   f"resvg AA is analytic area coverage while sigmoid AA is an approximation, "
                   f"remaining gap concentrated on shape boundaries.")
    else:
        verdict = (f"FAIL: mean SSIM {best['ssim_mean']:.5f} < {target}; "
                   f"gap dominated by boundary transition band and stroke soft-min width.")
    print(verdict)

    report = {
        "size": args.size, "scenes": args.scenes, "seed0": args.seed0,
        "target_ssim": target, "best": {k: v for k, v in best.items() if k != "per_scene_ssim"},
        "verdict": verdict, "grid": results,
    }
    report_path = os.path.join(args.out, "calibration.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report -> {report_path}")

    rend = SoftSVGRenderer(out_size=args.size, gamma_px=best["gamma_px"], zeta_px=best["zeta_px"],
                           sub_px=args.subpx, samples_per_curve=args.samples)
    ssims = []
    for i, (scene, ref) in enumerate(zip(scenes, refs)):
        soft = rend.to_uint8(rend.render_scene(scene))
        ssims.append((compare_u8(ref, soft)["ssim"], i, ref, soft))
    ssims.sort(key=lambda x: x[0])
    k = min(args.worst, len(ssims))
    tiles = [triptych(ref, soft) for _, _, ref, soft in ssims[:k]]
    while len(tiles) < 2:
        tiles.append(np.zeros_like(tiles[0]))
    cols = 2
    rows_n = (k + cols - 1) // cols
    while len(tiles) < rows_n * cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1)
                           for r in range(rows_n)], axis=0)
    worst_path = os.path.join(args.out, "worst_cases.png")
    Image.fromarray(grid).save(worst_path)
    print(f"worst-{k} triptychs -> {worst_path}")
    for s, i, _, _ in ssims[:k]:
        print(f"  scene {i}: SSIM={s:.5f}")


if __name__ == "__main__":
    main()
