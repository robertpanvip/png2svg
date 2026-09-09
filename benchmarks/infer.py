"""最小推理 demo（默认开启剪枝）— 对应 HANDOFF.md §9.3。

用法：
  # 把 png 转为 svg 字符串（打印到 stdout）
  python benchmarks/infer.py --ckpt runs/gpu/last.pt --png input.png

  # 用合成的 sample scene 演示
  python benchmarks/infer.py --ckpt runs/gpu/last.pt --sample

  # 写 SVG 到文件（不带 PNG 头包裹，由 svg/serializer 自带）
  python benchmarks/infer.py --ckpt runs/gpu/last.pt --sample --out pred.svg

  # 关闭剪枝，对比 full 模式
  python benchmarks/infer.py --ckpt runs/gpu/last.pt --sample --no-prune

这是纯 CPU 推理——验证项目硬指标 "CPU 推理 <1s"，可作为后续部署的最
小骨架。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from benchmarks.prune import prune_scene
from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.network import VectorNet
from model.spec import predictions_to_targets
from model.targets import decode_scene
from svg.render import render_scene_resvg
from svg.scene_graph import Scene
from svg.serializer import serialize


def to_tensor(arr_u8):
    return torch.from_numpy(arr_u8.astype(np.float32) / 255.0).permute(2, 0, 1)


def load_png(path, size):
    from PIL import Image
    img = Image.open(path).convert("RGBA").resize((size, size))
    return to_tensor(np.asarray(img, dtype=np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/gpu/last.pt")
    ap.add_argument("--png", type=str, default="", help="输入 PNG 路径；与 --sample 二选一")
    ap.add_argument("--sample", action="store_true", help="用 SceneGenerator 合成一张")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--prune-threshold", type=float, default=0.002)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--out", type=str, default="", help="可选：写 SVG 到此路径")
    args = ap.parse_args()

    net = VectorNet()
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(ck["net"])
    net.eval()

    soft = SoftSVGRenderer(args.size)
    if args.sample:
        scene = SceneGenerator(None, 0).sample()
        rgba = soft.render_scene(scene)
        print(f"[infer] generated sample: {len(scene.objects)} objects, "
              f"bg={scene.background}")
    elif args.png:
        if not os.path.exists(args.png):
            print(f"[error] --png not found: {args.png}", file=sys.stderr)
            sys.exit(1)
        rgba = load_png(args.png, args.size)
    else:
        ap.error("must provide --png or --sample")

    img_in = rgba  # [4,H,W] tensor
    t0 = time.perf_counter()
    with torch.no_grad():
        slots, aux, bg = net(img_in.unsqueeze(0))
        slots_np, bg_np = predictions_to_targets(slots[0], bg[0])
    scene_pred = decode_scene(slots_np, bg_np, canvas=args.size)
    t_net = time.perf_counter() - t0

    t0 = time.perf_counter()
    if args.no_prune or args.prune_threshold <= 0:
        pruned = scene_pred
        info = {"n_before": len(scene_pred.objects), "n_after": len(scene_pred.objects),
                "n_dropped": 0}
    else:
        pruned, info = prune_scene(scene_pred, img_in, args.size,
                                   threshold=args.prune_threshold)
    t_prune = time.perf_counter() - t0

    svg_str = serialize(pruned)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(svg_str)
        print(f"[saved] {args.out}  ({len(svg_str)} chars)")

    t_total = t_net + t_prune
    print(f"[infer] ckpt={os.path.basename(args.ckpt)} "
          f"size={args.size} pruned={not args.no_prune and args.prune_threshold > 0}")
    print(f"  net  : {t_net*1e3:6.1f} ms")
    print(f"  prune: {t_prune*1e3:6.1f} ms ({info['n_before']} -> {info['n_after']}, "
          f"-{info['n_dropped']})")
    print(f"  total: {t_total*1e3:6.1f} ms ({'CPU <1s OK' if t_total < 1 else 'OVER 1s'})")
    if not args.out:
        print("\n--- SVG ---\n")
        print(svg_str)


if __name__ == "__main__":
    main()