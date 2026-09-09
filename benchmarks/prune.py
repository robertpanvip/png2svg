"""推理侧冗余剪枝（对应 HANDOFF2 §4.3 与门控诊断结论）。

思路：HANDOFF.md/今日诊断显示 pred scene 倾向多预测 slot，但多余 slot
对渲染贡献几乎为 0（去掉后 mae 变化 ≈ 0）。因此对每个预测对象做一次
per-object ablation：

  Δ_i = mae(scene_pred - {obj_i}) - mae(scene_pred)

若 Δ_i < threshold（默认 0.002），说明该对象是无害冗余，可删除。

用法：
  from benchmarks.prune import prune_scene
  pruned, info = prune_scene(scene_pred, img_in, size, threshold=0.002)

代价：每个对象多渲染一次（~100ms），N=4-10 时总耗时 ~1s/场景。
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from svg.render import render_scene_resvg
from svg.scene_graph import Scene


def _to_tensor(arr_u8):
    return torch.from_numpy(arr_u8.astype(np.float32) / 255.0).permute(2, 0, 1)


def _mae(rgba, img_in):
    t = _to_tensor(rgba)
    return float((t - img_in).abs().mean())


def prune_scene(scene_pred: Scene, img_in: torch.Tensor, size: int,
                threshold: float = 0.002, keep_indices=None) -> tuple[Scene, dict]:
    """逐对象 ablation 剪枝。

    Args:
      scene_pred: 模型解码得到的 Scene。
      img_in: 输入 PNG 的 tensor（C,H,W），用于算误差基准。
      size: canvas 边长。
      threshold: Δmae 小于等于此值视为冗余（默认 0.002）。
      keep_indices: 强制保留的对象索引（剪枝不会动它们），如门控阈值下必留者。

    Returns:
      (pruned_scene, info_dict)
    """
    objs = list(scene_pred.objects)
    n = len(objs)
    if n == 0:
        return scene_pred, {"n_dropped": 0, "dropped_idx": [], "threshold": threshold,
                            "mae_full": 0.0, "mae_pruned": 0.0, "delta_per_obj": []}

    rgba_full = render_scene_resvg(scene_pred, size, size)
    mae_full = _mae(rgba_full, img_in)

    keep = set(keep_indices or [])
    deltas = []
    for i in range(n):
        if i in keep:
            deltas.append(float("inf"))
            continue
        sub = Scene(width=scene_pred.width, height=scene_pred.height,
                    background=scene_pred.background,
                    objects=objs[:i] + objs[i + 1:])
        rgba_no_i = render_scene_resvg(sub, size, size)
        mae_no_i = _mae(rgba_no_i, img_in)
        deltas.append(mae_no_i - mae_full)

    drop_idx = [i for i, d in enumerate(deltas) if d <= threshold]
    kept_idx = [i for i in range(n) if i not in drop_idx]
    pruned = Scene(width=scene_pred.width, height=scene_pred.height,
                   background=scene_pred.background,
                   objects=[objs[i] for i in kept_idx])
    mae_pruned = _mae(render_scene_resvg(pruned, size, size), img_in)
    info = {
        "n_before": n, "n_after": len(kept_idx),
        "n_dropped": len(drop_idx), "dropped_idx": drop_idx,
        "threshold": threshold,
        "mae_full": mae_full, "mae_pruned": mae_pruned,
        "mae_delta": mae_pruned - mae_full,
        "delta_per_obj": deltas,
    }
    return pruned, info

def prune_scene_greedy(scene_pred: Scene, img_in: torch.Tensor, size: int,
                       threshold: float = 0.002, keep_indices=None) -> tuple[Scene, dict]:
    """贪心迭代剪枝：每轮移除 Δ 最小的冗余对象后重算 Δ。

    背景（2026-09-10 40k 验收发现）：模型存在 slot 重复绘制（多副本覆盖同一
    区域），独立逐对象消融时每个对象的 Δ 都被"副本"掩盖（剪任何一个都有
    别的顶着 → Δ≈0 → 全部被剪）。贪心法每剪掉一个副本，剩余副本的真实
    贡献才会在下一轮显形，最终每个保留对象都是不可替代的。

    代价：O(N²) 次 resvg 渲染（N=8 时 ≤36 次 ×~20ms ≈ 0.7s CPU）；
    超过 1s 预算时可调大 threshold 或退回 prune_scene（独立消融）。
    """
    objs = list(scene_pred.objects)
    keep = set(keep_indices or [])
    dropped = []
    mae_full = _mae(render_scene_resvg(scene_pred, size, size), img_in)

    def mae_of(obj_list):
        if not obj_list:
            sc = Scene(width=scene_pred.width, height=scene_pred.height,
                       background=scene_pred.background, objects=[])
        else:
            sc = Scene(width=scene_pred.width, height=scene_pred.height,
                       background=scene_pred.background, objects=obj_list)
        return _mae(render_scene_resvg(sc, size, size), img_in)

    cur = objs
    while True:
        cur_mae = mae_of(cur)
        removable = [i for i in range(len(cur)) if i not in keep]
        if not removable:
            break
        best_i, best_d = None, threshold
        for i in removable:
            d = mae_of(cur[:i] + cur[i + 1:]) - cur_mae
            if d <= threshold and (best_i is None or d < best_d):
                best_i, best_d = i, d
        if best_i is None:
            break
        dropped.append(best_i)
        cur = cur[:best_i] + cur[best_i + 1:]

    kept_idx = [i for i in range(len(objs)) if i not in dropped]
    pruned = Scene(width=scene_pred.width, height=scene_pred.height,
                   background=scene_pred.background, objects=cur)
    mae_pruned = _mae(render_scene_resvg(pruned, size, size), img_in)
    info = {
        "n_before": len(objs), "n_after": len(cur),
        "n_dropped": len(dropped), "dropped_idx": dropped,
        "threshold": threshold, "mode": "greedy",
        "mae_full": mae_full, "mae_pruned": mae_pruned,
        "mae_delta": mae_pruned - mae_full,
        "delta_per_obj": [],
    }
    return pruned, info
