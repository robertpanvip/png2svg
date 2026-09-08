"""Diagnose the valid-gating over-prediction problem.

For each eval scene, records per-slot valid probability and compares three
decoding variants:
  A) as-is            (threshold 0.5)
  B) top-k by prob    (k = number of GT objects)
  C) GT count only    -> tells us how much the extra slots actually cost in mae

Usage: python benchmarks/diag_gate.py --ckpt runs/gpu/last.pt --num 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.network import VectorNet
from model.spec import predictions_to_targets
from model.targets import decode_scene, encode_scene, I_VALID, NUM_SLOTS
from svg.render import render_scene_resvg


def to_tensor_rgba(arr_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr_u8.astype(np.float32) / 255.0).permute(2, 0, 1)


def slots_to_mae(slots_np: np.ndarray, bg_np, img_in: torch.Tensor, size: int) -> float:
    scene = decode_scene(slots_np, bg_np, canvas=size)
    rgba = render_scene_resvg(scene, size, size)
    return float((to_tensor_rgba(rgba) - img_in).abs().mean())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/gpu/last.pt")
    p.add_argument("--num", type=int, default=30)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--data-seed", type=int, default=4321)
    p.add_argument("--out", type=str, default="runs/diag_gate.json")
    args = p.parse_args()

    net = VectorNet()
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(ck["net"])
    net.eval()

    soft = SoftSVGRenderer(args.size, device="cpu", sub_px=1, grad_checkpoint=False)
    gen = SceneGenerator(None, args.data_seed)

    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    hist = {f"{a:.1f}-{b:.1f}": 0 for a, b in bins}
    rows = []
    mae_asis, mae_topk = [], []
    over, under, exact = 0, 0, 0

    for i in range(args.num):
        scene = gen.sample()
        img_in = soft.render_scene(scene)
        slots, aux, bg = net(img_in.unsqueeze(0))
        cls_ids = aux["cls"][0].argmax(-1).cpu().numpy()
        ftype_ids = aux["ftype"][0].argmax(-1).cpu().numpy()
        slots_np, bg_np = predictions_to_targets(slots[0], cls_ids, ftype_ids, bg[0])
        slots_gt, bg_gt = encode_scene(scene)

        prob = torch.sigmoid(slots[0][:, I_VALID]).numpy()
        n_pred = int((slots_np[:, 0] > 0.5).sum())
        n_gt = int((slots_gt[:, 0] > 0.5).sum())
        for v in prob:
            for a, b in bins:
                if a <= v < b:
                    hist[f"{a:.1f}-{b:.1f}"] += 1
                    break

        m_a = slots_to_mae(slots_np, bg_np, img_in, args.size)

        # variant B: keep only the top-n_gt slots by valid prob, zero the rest
        slots_topk = slots_np.copy()
        if n_pred > n_gt:
            order = np.argsort(-prob)
            drop = order[n_gt:]
            slots_topk[drop, 0] = 0.0
            m_b = slots_to_mae(slots_topk, bg_np, img_in, args.size)
        else:
            m_b = m_a

        mae_asis.append(m_a)
        mae_topk.append(m_b)
        if n_pred > n_gt:
            over += 1
        elif n_pred < n_gt:
            under += 1
        else:
            exact += 1
        rows.append({"i": i, "n_pred": n_pred, "n_gt": n_gt,
                     "mae_asis": round(m_a, 5), "mae_topk": round(m_b, 5),
                     "prob": [round(float(x), 3) for x in prob[:6]]})

    print(f"\n[gate diag] {args.num} scenes  ckpt={args.ckpt}")
    print(f"  门控精确匹配 {exact}/{args.num}   多预测 {over}   少预测 {under}")
    print(f"  平均对象数: pred {np.mean([r['n_pred'] for r in rows]):.2f} vs "
          f"gt {np.mean([r['n_gt'] for r in rows]):.2f}")
    print(f"  mae as-is      {np.mean(mae_asis):.5f}")
    print(f"  mae 裁到GT数量 {np.mean(mae_topk):.5f}   "
          f"(Δ {np.mean(mae_topk) - np.mean(mae_asis):+.5f})")
    print("  valid 概率分布:")
    for k, v in hist.items():
        print(f"    {k:<9} {v:>4}  {'#' * (v // 3)}")
    edge = hist["0.3-0.5"] + hist["0.5-0.7"]
    print(f"  边缘区间(0.3-0.7)占比: {100*edge/sum(hist.values()):.1f}%  <- 模型区分度")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"hist": hist, "mae_asis": float(np.mean(mae_asis)),
                       "mae_topk": float(np.mean(mae_topk)),
                       "over": over, "under": under, "exact": exact,
                       "per_scene": rows}, f, indent=2)
        print(f"  [saved] {args.out}")


if __name__ == "__main__":
    main()
