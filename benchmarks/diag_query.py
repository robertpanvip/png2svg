"""Slot / query 退化检查：判断 8 个 slot 是否收敛到同一向量（对称坍缩）。

判定指标：
  - queries 余弦相似度矩阵(8x8)：≈1.0 说明 query 退化
  - 单次前向中 8 个 slot_head 输出向量的两两余弦相似度：≈1.0 说明 slot 输出退化
  - 各 slot 预测 bbox 中心的方差：≈0 说明空间定位坍缩
  - slot_head.weight / bias、cls_head.bias 的范数/分布
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.generator import GeneratorConfig, SceneGenerator          # noqa: E402
from model.network import VectorNet                                     # noqa: E402
from model.targets import (I_BBOX, NUM_SLOTS)                          # noqa: E402
from svg.render import render_scene_resvg                               # noqa: E402
from benchmarks.suite import SUITES, to_tensor_rgba                     # noqa: E402


def _cos_mat(x: torch.Tensor) -> np.ndarray:
    """x: (N, D) -> (N, N) 余弦相似度。"""
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
    return (x @ x.t()).cpu().numpy()


def _load_net(ckpt: str) -> "VectorNet":
    """容错加载：跨架构 checkpoint（如含已删除的 spatial_anchor）也能比对。"""
    net = VectorNet()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["net"]
    own = net.state_dict()
    sd = {k: v for k, v in sd.items() if k in own}
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num", type=int, default=20)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--suite", default="mixed")
    p.add_argument("--out", default="runs/diag_query.json")
    args = p.parse_args()

    net = _load_net(args.ckpt)

    # --- 静态权重检查 ---
    with torch.no_grad():
        q = net.queries[0]                      # (NUM_SLOTS, d)
        qsim = _cos_mat(q)
        # 去掉对角线后的平均相似度
        off = qsim - np.eye(qsim.shape[0]) * 2.0
        q_sim_mean = float(off[off > -1].mean())

        sh_w = net.slot_head.weight
        sh_b = net.slot_head.bias
        cls_b = net.cls_head.bias
        stats = {
            "num_slots": NUM_SLOTS,
            "query_cos_sim_mean_offdiag": q_sim_mean,
            "query_cos_sim_max_offdiag": float(off[off > -1].max()),
            "slot_head_weight_std": float(sh_w.std()),
            "slot_head_weight_norm": float(sh_w.norm()),
            "slot_head_bias_norm": float(sh_b.norm()),
            "slot_head_bias_I_BBOX": [float(v) for v in sh_b[I_BBOX:I_BBOX + 4]],
            "cls_head_bias": [float(v) for v in cls_b],
            "cls_head_bias_std": float(cls_b.std()),
        }

    cfg = GeneratorConfig(**SUITES[args.suite])
    gen = SceneGenerator(cfg, 555000)

    slot_out_sims, center_vars, slot_active = [], [], []
    with torch.no_grad():
        for i in range(args.num):
            scene = gen.sample()
            img = to_tensor_rgba(render_scene_resvg(scene, args.size, args.size))
            slots, aux, bg = net(img.unsqueeze(0))
            slots = slots[0]                    # (NUM_SLOTS, SLOT_DIM)
            # slot 输出两两余弦相似度
            sim = _cos_mat(slots)
            off = sim - np.eye(sim.shape[0]) * 2.0
            slot_out_sims.append(float(off[off > -1].mean()))
            # bbox 中心方差（仅统计 valid）；cx/cy 在 slot 的 I_BBOX+0/+1 = 3/4
            cxr = slots[:, I_BBOX + 0]
            cyr = slots[:, I_BBOX + 1]
            cxv = torch.sigmoid(cxr).cpu().numpy()
            cyv = torch.sigmoid(cyr).cpu().numpy()
            valid = torch.sigmoid(slots[:, 0]).cpu().numpy()
            active = valid > 0.5
            slot_active.append(int(active.sum()))
            if active.sum() > 0:
                center_vars.append(float(np.var(cxv[active]) + np.var(cyv[active])))

    stats["forward_slot_out_cos_sim_mean"] = float(np.mean(slot_out_sims))
    stats["forward_slot_out_cos_sim_max"] = float(np.max(slot_out_sims))
    stats["per_scene_active_slots_mean"] = float(np.mean(slot_active))
    stats["bbox_center_variance_mean"] = float(np.mean(center_vars)) if center_vars else 0.0
    stats["num_scenes"] = args.num

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 62)
    print(f"NUM_SLOTS                 = {stats['num_slots']}")
    print(f"query 余弦相似度(非对角)  = {stats['query_cos_sim_mean_offdiag']:.4f} "
          f"(max {stats['query_cos_sim_max_offdiag']:.4f})")
    print(f"slot输出余弦相似度(前向)  = {stats['forward_slot_out_cos_sim_mean']:.4f} "
          f"(max {stats['forward_slot_out_cos_sim_max']:.4f})")
    print(f"每场景活跃 slot 均值      = {stats['per_scene_active_slots_mean']:.2f}")
    print(f"bbox 中心方差均值         = {stats['bbox_center_variance_mean']:.6f}")
    print(f"slot_head.weight std      = {stats['slot_head_weight_std']:.4f}")
    print(f"slot_head.bias[I_BBOX]    = {[round(v,3) for v in stats['slot_head_bias_I_BBOX']]}")
    print(f"cls_head.bias             = {[round(v,3) for v in stats['cls_head_bias']]}")
    print("=" * 62)
    print("判定：相似度≈1.0 / 中心方差≈0 → 对称坍缩 confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
