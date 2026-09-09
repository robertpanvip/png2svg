"""部署侧独立 CPU 推理：PNG/合成图 → SVG 字符串。

用法：
  python deploy/predict.py --weights deploy/weights/net_int8.pt --int8 --sample --out pred.svg
  python deploy/predict.py --weights deploy/weights/net_fp32.pt --png input.png
  python deploy/predict.py --weights deploy/weights/net_int8.pt --int8 --sample --bench

只依赖：torch(CPU) + numpy + PIL。管线：net → predictions_to_targets →
decode_scene → (可选 prune) → serialize。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.ao.quantization import quantize_dynamic

from benchmarks.prune import prune_scene, prune_scene_greedy
from model.network import VectorNet
from model.spec import predictions_to_targets
from model.targets import decode_scene
from svg.serializer import serialize


def load_model(weights: str, int8: bool, size: int) -> VectorNet:
    net = VectorNet()
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    if int8:
        # 先按相同规则量化出同构骨架，再 strict load 量化 state_dict
        net = quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
    net.load_state_dict(sd)
    net.eval()
    return net


def load_png(path: str, size: int) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("RGBA").resize((size, size))
    arr = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)


def predict(net, img_in: torch.Tensor, size: int,
            prune_threshold: float = 0.002, prune_mode: str = "greedy"):
    """返回 (svg_str, info)。纯 CPU。

    prune_mode: "greedy"（贪心迭代，抗 slot 重复绘制，O(N²) 渲染）
                | "independent"（独立消融，快但会被副本掩盖）
    """
    t0 = time.perf_counter()
    with torch.no_grad():
        slots, aux, bg = net(img_in.unsqueeze(0))
        slots_np, bg_np = predictions_to_targets(slots[0], bg[0])
    t_net = time.perf_counter() - t0

    scene_pred = decode_scene(slots_np, bg_np, canvas=size)

    t0 = time.perf_counter()
    if prune_threshold > 0:
        fn = prune_scene_greedy if prune_mode == "greedy" else prune_scene
        pruned, info = fn(scene_pred, img_in, size, threshold=prune_threshold)
    else:
        pruned, info = scene_pred, {"n_before": len(scene_pred.objects),
                                    "n_after": len(scene_pred.objects),
                                    "n_dropped": 0}
    t_prune = time.perf_counter() - t0

    svg_str = serialize(pruned)
    return svg_str, {"net_ms": t_net * 1e3, "prune_ms": t_prune * 1e3,
                     "total_ms": (t_net + t_prune) * 1e3, "prune": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join("deploy", "weights", "net_fp32.pt"))
    ap.add_argument("--int8", action="store_true", help="weights 为 INT8 量化版")
    ap.add_argument("--png", default="", help="输入 PNG；与 --sample/--bench 二选一")
    ap.add_argument("--sample", action="store_true", help="用 SceneGenerator 合成一张")
    ap.add_argument("--bench", action="store_true", help="跑 10 场景延迟基准")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--prune-threshold", type=float, default=0.002)
    ap.add_argument("--prune-mode", default="greedy", choices=["greedy", "independent"],
                    help="greedy=贪心迭代（默认，抗 slot 重复绘制）；independent=独立消融（快）")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--out", default="", help="SVG 输出路径（可选）")
    args = ap.parse_args()

    net = load_model(args.weights, args.int8, args.size)

    if args.bench:
        from dataset.generator import SceneGenerator
        from dataset.renderer import SoftSVGRenderer
        soft = SoftSVGRenderer(args.size)
        gen = SceneGenerator(None, 42)
        net_ms, prune_ms, tot_ms = [], [], []
        for _ in range(10):
            scene = gen.sample()
            rgba = soft.render_scene(scene)
            _, info = predict(net, rgba, args.size,
                              0.0 if args.no_prune else args.prune_threshold,
                              prune_mode=args.prune_mode)
            net_ms.append(info["net_ms"])
            prune_ms.append(info["prune_ms"])
            tot_ms.append(info["total_ms"])
        q = "int8" if args.int8 else "fp32"
        print(f"[bench] weights={os.path.basename(args.weights)}({q}) "
              f"prune={'off' if args.no_prune else args.prune_threshold} n=10")
        print(f"  net  median={statistics.median(net_ms):6.1f} ms  "
              f"max={max(net_ms):6.1f} ms")
        print(f"  prune median={statistics.median(prune_ms):6.1f} ms")
        print(f"  total median={statistics.median(tot_ms):6.1f} ms  "
              f"max={max(tot_ms):6.1f} ms  "
              f"({'CPU <1s OK' if max(tot_ms) < 1000 else 'OVER 1s'})")
        return

    if args.sample:
        from dataset.generator import SceneGenerator
        from dataset.renderer import SoftSVGRenderer
        scene = SceneGenerator(None, 0).sample()
        rgba = SoftSVGRenderer(args.size).render_scene(scene)
        print(f"[predict] sample: {len(scene.objects)} objects")
    elif args.png:
        if not os.path.exists(args.png):
            print(f"[error] --png not found: {args.png}", file=sys.stderr)
            sys.exit(1)
        rgba = load_png(args.png, args.size)
    else:
        ap.error("must provide --png or --sample or --bench")

    svg_str, info = predict(net, rgba, args.size,
                            0.0 if args.no_prune else args.prune_threshold,
                            prune_mode=args.prune_mode)
    print(f"[predict] net={info['net_ms']:.1f}ms prune={info['prune_ms']:.1f}ms "
          f"total={info['total_ms']:.1f}ms "
          f"objs={info['prune']['n_before']}->{info['prune']['n_after']}")
    if info["prune"]["n_after"] == 0 and info["prune"]["n_before"] > 0:
        print("[warn] 所有对象被剪掉——通常是权重未收敛（fill=none 不可见）"
              "或 --prune-threshold 过大，可试 --no-prune")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(svg_str)
        print(f"[saved] {args.out} ({len(svg_str)} chars)")
    else:
        print("\n--- SVG ---\n")
        print(svg_str)


if __name__ == "__main__":
    main()
