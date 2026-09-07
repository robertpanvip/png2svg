from __future__ import annotations

import argparse
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


def triptych(ref_u8, soft_u8, scale: float = 6.0):
    diff = np.abs(soft_u8[:, :, :3].astype(np.int16) - ref_u8[:, :, :3].astype(np.int16))
    dn = (diff.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
    return np.concatenate([ref_u8[:, :, :3], soft_u8[:, :, :3], dn], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gamma", type=float, default=0.35)
    ap.add_argument("--zeta", type=float, default=0.8)
    ap.add_argument("--subpx", type=int, default=1)
    ap.add_argument("--out", type=str, default="benchmarks/reports/smoke.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    gen = SceneGenerator(GeneratorConfig(), seed=args.seed)
    rend = SoftSVGRenderer(out_size=args.size, gamma_px=args.gamma, zeta_px=args.zeta,
                           sub_px=args.subpx)

    tiles, ssims, times = [], [], []
    t_total0 = time.perf_counter()
    for i in range(args.scenes):
        scene = gen.sample()
        t0 = time.perf_counter()
        ref = render_scene_resvg(scene, args.size, args.size)
        t_resvg = time.perf_counter() - t0
        t0 = time.perf_counter()
        soft = rend.to_uint8(rend.render_scene(scene))
        t_soft = time.perf_counter() - t0
        m = compare_u8(ref, soft)
        ssims.append(m["ssim"])
        times.append(t_soft)
        print(f"scene {i}: SSIM={m['ssim']:.5f}  PSNR={m['psnr']:6.2f}dB  MAE={m['mae']:.5f}  "
              f"soft={t_soft * 1000:6.1f}ms  resvg={t_resvg * 1000:5.1f}ms")
        tiles.append(triptych(ref, soft))

    cols = 4
    rows_n = (args.scenes + cols - 1) // cols
    while len(tiles) < rows_n * cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1)
                           for r in range(rows_n)], axis=0)
    Image.fromarray(grid).save(args.out)

    print(f"\nmean SSIM = {np.mean(ssims):.5f}   min SSIM = {np.min(ssims):.5f}")
    print(f"mean soft render = {np.mean(times) * 1000:.1f} ms/scene  "
          f"({1.0 / np.mean(times):.1f} scenes/s single-thread CPU)")
    print(f"triptych grid -> {args.out}  (total {time.perf_counter() - t_total0:.1f}s)")


if __name__ == "__main__":
    main()
