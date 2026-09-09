"""架构层诊断：量化各分支（cls / bbox / ftype）失效对重建质量的实际代价。

方法（oracle ablation）：在预测出的 slot 张量上，把某个分支换成 GT 真值，
再渲染对比 mae。若替换后 mae 大幅下降，说明该分支是瓶颈。

    A baseline       : 全部用预测
    B oracle cls     : 类别换成 GT
    C oracle bbox    : cx/cy/w/h 换成 GT
    D oracle cls+bbox: 两者都换 GT
    E oracle valid   : 只保留 GT 里真实存在的对象（其他 slot 置 invalid）

同时统计 cls logits 的分布/熵、bbox 的空间分布，判断是否存在模式坍缩。

用法：
  python -u benchmarks/diag_arch.py --ckpt runs/gpu/last.pt --num 30 \
      --out runs/diag_arch.json
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
from model.spec import predictions_to_targets                           # noqa: E402
from model.targets import (I_VALID, I_CLS, I_BBOX, decode_scene,        # noqa: E402
                           encode_scene, NUM_SLOTS)
from svg.render import render_scene_resvg                               # noqa: E402
from benchmarks.suite import SUITES, to_tensor_rgba                     # noqa: E402

CLS_NAMES = ["blob", "polygon", "ellipse", "rect", "stroke"]


def _mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean())


def _hungarian_match(slots_pred: np.ndarray, slots_gt: np.ndarray):
    """按 bbox IoU 做贪心匹配，返回 [(pred_idx, gt_idx)]。"""
    def bbox(v):
        return v[I_BBOX:I_BBOX + 4]

    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x0, y0 = max(ax - aw / 2, bx - bw / 2), max(ay - ah / 2, by - bh / 2)
        x1, y1 = min(ax + aw / 2, bx + bw / 2), min(ay + ah / 2, by + bh / 2)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        uni = aw * ah + bw * bh - inter
        return inter / uni if uni > 1e-9 else 0.0

    pairs = []
    for pi in range(NUM_SLOTS):
        if slots_pred[pi, I_VALID] < 0.5:
            continue
        best, bj = 0.0, -1
        for gi in range(NUM_SLOTS):
            if slots_gt[gi, I_VALID] < 0.5:
                continue
            s = iou(bbox(slots_pred[pi]), bbox(slots_gt[gi]))
            if s > best:
                best, bj = s, gi
        if bj >= 0 and best > 0.10:
            pairs.append((pi, bj, best))
    return pairs


def render_slots(slots_np: np.ndarray, bg_np: np.ndarray, size: int):
    scene = decode_scene(slots_np, bg_np, canvas=size)
    return to_tensor_rgba(render_scene_resvg(scene, size, size))


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
    p.add_argument("--num", type=int, default=30)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--suite", default="mixed")
    p.add_argument("--out", default="runs/diag_arch.json")
    args = p.parse_args()

    net = _load_net(args.ckpt)

    cfg = GeneratorConfig(**SUITES[args.suite])
    gen = SceneGenerator(cfg, 777000)

    maes = {k: [] for k in ("baseline", "oracle_cls", "oracle_bbox",
                            "oracle_cls_bbox", "oracle_valid")}
    cls_pred_hist = np.zeros(5)
    cls_gt_hist = np.zeros(5)
    cls_entropy = []
    bbox_pred, bbox_gt = [], []
    n_matched = 0
    n_gt_total = 0

    for i in range(args.num):
        scene = gen.sample()
        img_in = to_tensor_rgba(render_scene_resvg(scene, args.size, args.size))

        with torch.no_grad():
            img = img_in.unsqueeze(0)  # to_tensor_rgba 已是 (C,H,W)
            slots, aux, bg = net(img)
            cls_ids = aux["cls"][0].argmax(-1).cpu().numpy()
            ftype_ids = aux["ftype"][0].argmax(-1).cpu().numpy()
            slots_np, bg_np = predictions_to_targets(
                slots[0], cls_ids, ftype_ids, bg[0])

        slots_gt, bg_gt = encode_scene(scene)

        # --- 统计 ---
        probs = torch.softmax(aux["cls"][0], dim=-1).numpy()
        for k in range(NUM_SLOTS):
            if slots_np[k, I_VALID] > 0.5:
                cls_pred_hist[int(cls_ids[k])] += 1
                ent = float(-(probs[k] * np.log(probs[k] + 1e-9)).sum())
                cls_entropy.append(ent)
            if slots_gt[k, I_VALID] > 0.5:
                cls_gt_hist[int(round(float(slots_gt[k, I_CLS])))] += 1
                n_gt_total += 1
                bbox_gt.append(slots_gt[k, I_BBOX:I_BBOX + 4])
            if slots_np[k, I_VALID] > 0.5:
                bbox_pred.append(slots_np[k, I_BBOX:I_BBOX + 4])

        pairs = _hungarian_match(slots_np, slots_gt)
        n_matched += len(pairs)

        # --- A baseline ---
        maes["baseline"].append(_mae(render_slots(slots_np, bg_np, args.size),
                                     img_in))

        # --- B oracle cls ---
        b = slots_np.copy()
        for pi, gi, _ in pairs:
            b[pi, I_CLS] = slots_gt[gi, I_CLS]
        maes["oracle_cls"].append(_mae(render_slots(b, bg_np, args.size), img_in))

        # --- C oracle bbox ---
        c = slots_np.copy()
        for pi, gi, _ in pairs:
            c[pi, I_BBOX:I_BBOX + 4] = slots_gt[gi, I_BBOX:I_BBOX + 4]
        maes["oracle_bbox"].append(_mae(render_slots(c, bg_np, args.size), img_in))

        # --- D oracle cls + bbox ---
        d = slots_np.copy()
        for pi, gi, _ in pairs:
            d[pi, I_CLS] = slots_gt[gi, I_CLS]
            d[pi, I_BBOX:I_BBOX + 4] = slots_gt[gi, I_BBOX:I_BBOX + 4]
        maes["oracle_cls_bbox"].append(
            _mae(render_slots(d, bg_np, args.size), img_in))

        # --- E oracle valid（只保留 GT 真实对象对应的 slot）---
        e = slots_np.copy()
        keep = {pi for pi, _, _ in pairs}
        for k in range(NUM_SLOTS):
            if k not in keep:
                e[k, I_VALID] = 0.0
        maes["oracle_valid"].append(_mae(render_slots(e, bg_np, args.size), img_in))

        print(f"[{i + 1:>3}/{args.num}] base={maes['baseline'][-1]:.4f} "
              f"cls={maes['oracle_cls'][-1]:.4f} bbox={maes['oracle_bbox'][-1]:.4f} "
              f"both={maes['oracle_cls_bbox'][-1]:.4f} "
              f"valid={maes['oracle_valid'][-1]:.4f} match={len(pairs)}",
              flush=True)

    bp = np.asarray(bbox_pred) if bbox_pred else np.zeros((0, 4))
    bg_ = np.asarray(bbox_gt) if bbox_gt else np.zeros((0, 4))

    def bstats(a, tag):
        if len(a) == 0:
            return f"{tag}: n=0"
        return (f"{tag}: n={len(a)} cx={a[:, 0].mean():.3f}±{a[:, 0].std():.3f} "
                f"cy={a[:, 1].mean():.3f}±{a[:, 1].std():.3f} "
                f"w={a[:, 2].mean():.3f}±{a[:, 2].std():.3f} "
                f"h={a[:, 3].mean():.3f}±{a[:, 3].std():.3f}")

    report = {
        "ckpt": args.ckpt, "suite": args.suite, "num_scenes": args.num,
        "mae": {k: float(np.mean(v)) for k, v in maes.items()},
        "mae_gain_vs_baseline": {
            k: float(np.mean(maes["baseline"]) - np.mean(v))
            for k, v in maes.items() if k != "baseline"},
        "cls_pred_hist": {CLS_NAMES[i]: int(cls_pred_hist[i]) for i in range(5)},
        "cls_gt_hist": {CLS_NAMES[i]: int(cls_gt_hist[i]) for i in range(5)},
        "cls_entropy_mean": float(np.mean(cls_entropy)) if cls_entropy else None,
        "cls_max_entropy": float(np.log(5)),
        "bbox_pred": bstats(bp, "pred"),
        "bbox_gt": bstats(bg_, "gt"),
        "matched_pairs": n_matched, "gt_objects": n_gt_total,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print(f"{'配置':<18}{'mae':>9}{'相对 baseline':>16}")
    base = np.mean(maes["baseline"])
    for k in ("baseline", "oracle_cls", "oracle_bbox", "oracle_cls_bbox",
              "oracle_valid"):
        m = np.mean(maes[k])
        g = f"{(base - m) / base * 100:+.1f}%" if k != "baseline" else "-"
        print(f"{k:<18}{m:>9.4f}{g:>16}")
    print("-" * 62)
    print("cls 预测分布:", {CLS_NAMES[i]: int(cls_pred_hist[i]) for i in range(5)})
    print("cls 真值分布:", {CLS_NAMES[i]: int(cls_gt_hist[i]) for i in range(5)})
    print(f"cls logits 熵: {np.mean(cls_entropy):.3f} / 最大 {np.log(5):.3f}")
    print(bstats(bp, "bbox pred"))
    print(bstats(bg_, "bbox gt  "))
    print(f"匹配到 {n_matched}/{n_gt_total} 个 GT 对象")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
