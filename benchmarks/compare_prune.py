"""推理侧冗余剪枝的 A/B 验证。

对 N 个场景（默认 mixed 分布），每个场景跑两次：
  baseline = 原 pred scene 的 mae / 对象数 / 召回
  pruned   = 用 prune.py 去掉无害冗余对象后的 mae / 对象数

输出 json + console 表：裁剪多少对象 / mae / ssim / 召回率变化。

用法：
  python benchmarks/compare_prune.py --ckpt runs/gpu/last.pt --num 50 --threshold 0.002
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from benchmarks.prune import prune_scene
from benchmarks.suite import (SUITES, _match, _bbox_of_obj, eval_scene,
                              objects_records, to_tensor_rgba)
from dataset.generator import GeneratorConfig, SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import _ssim_mean
from model.network import VectorNet
from svg.render import render_scene_resvg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/gpu/last.pt")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=202600)
    ap.add_argument("--threshold", type=float, default=0.002)
    ap.add_argument("--out", type=str, default="runs/prune_compare.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    net = VectorNet()
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(ck["net"]); net.eval()
    print(f"[prune-ab] ckpt={args.ckpt} step={ck.get('step', '?')} "
          f"num={args.num} threshold={args.threshold}")

    soft = SoftSVGRenderer(args.size)
    cfg = GeneratorConfig(**SUITES["mixed"])
    gen = SceneGenerator(cfg, args.seed)

    rows = []
    for i in range(args.num):
        scene = gen.sample()
        r = eval_scene(net, soft, scene, args.size)
        recs, n_m, n_g, n_p = objects_records(scene, r["scene_pred"])

        t0 = time.perf_counter()
        pruned, info = prune_scene(r["scene_pred"], r["img_in"], args.size,
                                   threshold=args.threshold)
        t_prune = time.perf_counter() - t0
        img_pruned = to_tensor_rgba(render_scene_resvg(pruned, args.size, args.size))
        ssim_full = float(1.0 - _ssim_mean(r["img_in"],
                                           to_tensor_rgba(r["img_pred"])))
        ssim_pruned = float(1.0 - _ssim_mean(r["img_in"], img_pruned))
        recs2, n_m2, _, n_p2 = objects_records(scene, pruned)

        rows.append({
            "seed": args.seed + i,
            "n_gt": n_g,
            "n_before": info["n_before"],
            "n_after": info["n_after"],
            "n_dropped": info["n_dropped"],
            "drop_ratio": info["n_dropped"] / max(info["n_before"], 1),
            "mae_full": info["mae_full"],
            "mae_pruned": info["mae_pruned"],
            "mae_delta": info["mae_pruned"] - info["mae_full"],
            "ssim_full": ssim_full,
            "ssim_pruned": ssim_pruned,
            "recall_before": n_m / max(n_g, 1),
            "recall_after": n_m2 / max(n_g, 1),
            "prune_ms": t_prune * 1e3,
            "dropped_idx": info["dropped_idx"],
            "delta_per_obj": info["delta_per_obj"],
        })

    df = lambda key: float(np.mean([r[key] for r in rows]))
    summary = {
        "num": args.num, "threshold": args.threshold,
        "avg_n_before": df("n_before"), "avg_n_after": df("n_after"),
        "avg_n_dropped": df("n_dropped"), "avg_drop_ratio": df("drop_ratio"),
        "avg_mae_full": df("mae_full"), "avg_mae_pruned": df("mae_pruned"),
        "avg_mae_delta": df("mae_delta"),
        "avg_ssim_full": df("ssim_full"), "avg_ssim_pruned": df("ssim_pruned"),
        "avg_recall_before": df("recall_before"),
        "avg_recall_after": df("recall_after"),
        "avg_prune_ms": df("prune_ms"),
    }
    out = {"summary": summary, "per_scene": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Summary over {args.num} scenes ===")
    print(f"对象数: {summary['avg_n_before']:.1f} -> {summary['avg_n_after']:.1f} "
          f"(-{summary['avg_n_dropped']:.1f}, drop {summary['avg_drop_ratio']*100:.1f}%)")
    print(f"mae: {summary['avg_mae_full']:.4f} -> {summary['avg_mae_pruned']:.4f} "
          f"(Δ {summary['avg_mae_delta']:+.4f})")
    print(f"ssim: {summary['avg_ssim_full']:.4f} -> {summary['avg_ssim_pruned']:.4f}")
    print(f"recall: {summary['avg_recall_before']:.3f} -> {summary['avg_recall_after']:.3f}")
    print(f"prune overhead: {summary['avg_prune_ms']:.0f}ms/scene")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()