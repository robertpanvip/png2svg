"""Synthetic Oracle Benchmark：固定 seed 的分组能力评测。

对应 HANDOFF2.md §9 / §10 / §21（P0）：
  1. 固定 benchmark seeds
  2. 生成 PNG + GT Scene + GT SVG
  3. 批量运行当前模型
  4. 对比 Scene prediction 与 GT Scene（对象级：召回/精确/类别/填充/几何/颜色）
  5. 渲染 predicted SVG 与 input PNG 比对（场景级：mae/ssim）
  7. 输出 feature-level metrics
  8. 保存失败样本和 seed
  9. 输出 benchmark_report.json

用法：
  python benchmarks/suite.py --ckpt runs/gpu/last.pt --num 20
  python benchmarks/report.py --report runs/benchmark_report.json
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

from dataset.generator import GeneratorConfig, SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import _ssim_mean
from model.network import VectorNet
from model.spec import predictions_to_targets
from model.targets import decode_scene, encode_scene
from svg.render import render_scene_resvg
from svg.scene_graph import CMD_M, FILL_NONE

IOU_MATCH = 0.25          # 对象匹配阈值
OCCLUDE_IOU = 0.15        # 判定"被遮挡"的重叠阈值
FAIL_MAE = 0.12           # 失败样本阈值（高于此值落盘）


# ---------------------------------------------------------------- 能力分组
# 说明：渐变 kind（linear/radial）由生成器内部随机决定，无法在 config 层强制，
# 因此 suite 只控制"是否出渐变"，linear/radial 的区分在指标层按 GT 实际属性细分。
SUITES: dict[str, dict] = {
    "basic": dict(min_objects=2, max_objects=3, overlap_prob=0.0, gradient_prob=0.0,
                  alpha_prob=0.0, stroke_prob=0.0, hole_prob=0.0,
                  transparent_bg_prob=0.0,
                  shape_weights={"ellipse": 0.5, "rect": 0.5}),
    "geometry": dict(min_objects=2, max_objects=4, overlap_prob=0.2, gradient_prob=0.0,
                     alpha_prob=0.0, stroke_prob=0.0, hole_prob=0.0,
                     transparent_bg_prob=0.0,
                     shape_weights={"blob": 0.4, "polygon": 0.4, "ellipse": 0.1, "rect": 0.1}),
    "bezier": dict(min_objects=2, max_objects=4, overlap_prob=0.2, gradient_prob=0.0,
                   alpha_prob=0.0, stroke_prob=0.5, hole_prob=0.0,
                   transparent_bg_prob=0.0,
                   shape_weights={"blob": 0.6, "stroke": 0.4}),
    "gradient": dict(min_objects=2, max_objects=4, overlap_prob=0.2, gradient_prob=1.0,
                     alpha_prob=0.0, stroke_prob=0.0, hole_prob=0.0,
                     transparent_bg_prob=0.0, max_stops=3,
                     shape_weights={"ellipse": 0.35, "rect": 0.35, "blob": 0.3}),
    "gradient_multistop": dict(min_objects=2, max_objects=4, overlap_prob=0.2,
                               gradient_prob=1.0, alpha_prob=0.0, stroke_prob=0.0,
                               hole_prob=0.0, transparent_bg_prob=0.0, max_stops=4,
                               shape_weights={"ellipse": 0.35, "rect": 0.35, "blob": 0.3}),
    "transparency": dict(min_objects=2, max_objects=4, overlap_prob=0.35, gradient_prob=0.3,
                         alpha_prob=1.0, stroke_prob=0.0, hole_prob=0.0,
                         transparent_bg_prob=0.5,
                         shape_weights={"ellipse": 0.3, "rect": 0.3, "blob": 0.2,
                                        "polygon": 0.2}),
    "occlusion": dict(min_objects=4, max_objects=6, overlap_prob=1.0, gradient_prob=0.3,
                      alpha_prob=0.2, stroke_prob=0.0, hole_prob=0.0,
                      transparent_bg_prob=0.0,
                      shape_weights={"ellipse": 0.3, "rect": 0.3, "blob": 0.2,
                                     "polygon": 0.2}),
    "holes": dict(min_objects=2, max_objects=4, overlap_prob=0.2, gradient_prob=0.4,
                  alpha_prob=0.0, stroke_prob=0.0, hole_prob=1.0,
                  transparent_bg_prob=0.0,
                  shape_weights={"blob": 1.0}),
    "layers": dict(min_objects=6, max_objects=8, overlap_prob=0.9, gradient_prob=0.4,
                   alpha_prob=0.3, stroke_prob=0.1, hole_prob=0.1,
                   transparent_bg_prob=0.0,
                   shape_weights={"ellipse": 0.3, "rect": 0.3, "blob": 0.2,
                                  "polygon": 0.2}),
    "dense": dict(min_objects=8, max_objects=10, overlap_prob=0.6, gradient_prob=0.4,
                  alpha_prob=0.2, stroke_prob=0.1, hole_prob=0.1,
                  transparent_bg_prob=0.0, min_bbox=0.08, max_bbox=0.35,
                  shape_weights={"ellipse": 0.25, "rect": 0.25, "blob": 0.25,
                                 "polygon": 0.25}),
    "stroke": dict(min_objects=2, max_objects=4, overlap_prob=0.2, gradient_prob=0.0,
                   alpha_prob=0.0, stroke_prob=1.0, hole_prob=0.0,
                   transparent_bg_prob=0.0,
                   shape_weights={"stroke": 0.5, "blob": 0.25, "polygon": 0.25}),
    "mixed": dict(),  # 默认 GeneratorConfig，等同训练分布
}

# 生成器当前不支持的能力（HANDOFF2 §15 Level 4），报告中标注为未覆盖
UNSUPPORTED = ["shadow", "blur", "glow"]


# ---------------------------------------------------------------- 工具函数
def _bbox_of_obj(obj):
    g = obj.geometry
    return getattr(g, "bbox", None)


def _area(b):
    return 0.0 if b is None else max(float(b.w), 1e-6) * max(float(b.h), 1e-6)


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    ax0, ay0 = a.cx - a.w / 2, a.cy - a.h / 2
    ax1, ay1 = a.cx + a.w / 2, a.cy + a.h / 2
    bx0, by0 = b.cx - b.w / 2, b.cy - b.h / 2
    bx1, by1 = b.cx + b.w / 2, b.cy + b.h / 2
    ix, iy = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    inter = max(ix, 0.0) * max(iy, 0.0)
    if inter <= 0:
        return 0.0
    uni = _area(a) + _area(b) - inter
    return inter / uni if uni > 0 else 0.0


def _has_hole(obj):
    segs = getattr(obj.geometry, "segments", None)
    if not segs:
        return False
    return sum(1 for s in segs if s.cmd == CMD_M) > 1


def _mean_color(obj):
    """把 fill 压成一个平均 RGB，用于颜色误差统计。"""
    f = obj.fill
    if f.type in ("none", FILL_NONE):
        return np.zeros(3, dtype=np.float64)
    if f.gradient is not None and f.gradient.stops:
        st = f.gradient.stops
        arr = np.array([list(s.rgb) for s in st], dtype=np.float64)
        return arr.mean(axis=0)
    return np.asarray(f.color, dtype=np.float64)[:3]


def _features(obj, idx, all_objs):
    b = _bbox_of_obj(obj)
    a = _area(b)
    later = all_objs[idx + 1:]
    occl = max((_iou(b, _bbox_of_obj(o)) for o in later), default=0.0)
    if a < 0.05:
        bucket = "small"
    elif a < 0.18:
        bucket = "mid"
    else:
        bucket = "large"
    return {
        "shape": obj.shape,
        "fill": obj.fill.type,
        "alpha": bool(obj.fill.alpha < 0.99 or obj.opacity < 0.99),
        "stroke": obj.stroke is not None,
        "hole": _has_hole(obj),
        "occluded": occl > OCCLUDE_IOU,
        "area": bucket,
    }


def _match(gts, preds):
    """贪心按 IoU 匹配，返回 [(gi, pi, iou), ...]。"""
    pairs = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            v = _iou(_bbox_of_obj(g), _bbox_of_obj(p))
            if v >= IOU_MATCH:
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, v))
    return out


def to_tensor_rgba(arr_u8):
    return torch.from_numpy(arr_u8.astype(np.float32) / 255.0).permute(2, 0, 1)


# ---------------------------------------------------------------- 主评测
@torch.no_grad()
def eval_scene(net, soft, scene, size):
    img_in = soft.render_scene(scene)

    t0 = time.perf_counter()
    slots, aux, bg = net(img_in.unsqueeze(0))
    cls_ids = aux["cls"][0].argmax(-1).cpu().numpy()
    ftype_ids = aux["ftype"][0].argmax(-1).cpu().numpy()
    slots_np, bg_np = predictions_to_targets(slots[0], cls_ids, ftype_ids, bg[0])
    t_net = time.perf_counter() - t0

    t0 = time.perf_counter()
    scene_pred = decode_scene(slots_np, bg_np, canvas=size)
    rgba = render_scene_resvg(scene_pred, size, size)
    t_vec = time.perf_counter() - t0

    img_pred = to_tensor_rgba(rgba)
    mae = float((img_pred - img_in).abs().mean())
    ssim_loss = float(1.0 - _ssim_mean(img_pred, img_in))

    slots_gt, bg_gt = encode_scene(scene)
    scene_gt = decode_scene(slots_gt, bg_gt, canvas=size)
    rgba_gt = render_scene_resvg(scene_gt, size, size)
    floor = float((to_tensor_rgba(rgba_gt) - img_in).abs().mean())

    return dict(mae=mae, ssim_loss=ssim_loss, floor=floor,
                net_ms=t_net * 1e3, vec_ms=t_vec * 1e3, total_ms=(t_net + t_vec) * 1e3,
                scene_pred=scene_pred, img_pred=rgba, img_in=img_in,
                n_pred=len(scene_pred.objects), n_gt=len(scene.objects))


def objects_records(scene_gt, scene_pred):
    gts, preds = scene_gt.objects, scene_pred.objects
    matches = _match(gts, preds)
    m_by_g = {gi: pi for gi, pi, _ in matches}
    recs = []
    for gi, g in enumerate(gts):
        feats = _features(g, gi, gts)
        rec = dict(suite_feature=feats, matched=gi in m_by_g)
        if gi in m_by_g:
            p = preds[m_by_g[gi]]
            iou = next(v for a, b, v in matches if a == gi)
            gb, pb = _bbox_of_obj(g), _bbox_of_obj(p)
            rec.update(
                iou=float(iou),
                center_dist=float(np.hypot(gb.cx - pb.cx, gb.cy - pb.cy)),
                shape_ok=bool(g.shape == p.shape),
                fill_ok=bool(g.fill.type == p.fill.type),
                stroke_ok=bool((g.stroke is not None) == (p.stroke is not None)),
                color_l1=float(np.abs(_mean_color(g) - _mean_color(p)).mean()),
            )
        else:
            rec.update(iou=0.0, center_dist=1.0, shape_ok=False, fill_ok=False,
                       stroke_ok=False, color_l1=1.0)
        recs.append(rec)
    return recs, len(matches), len(gts), len(preds)


def main():
    ap = argparse.ArgumentParser(description="Synthetic Oracle Benchmark")
    ap.add_argument("--ckpt", type=str, default="runs/gpu/last.pt")
    ap.add_argument("--num", type=int, default=20, help="每组场景数")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--only", type=str, default="", help="只跑指定组，逗号分隔")
    ap.add_argument("--cases-dir", type=str, default="benchmarks/cases")
    ap.add_argument("--out", type=str, default="runs/benchmark_report.json")
    ap.add_argument("--fail-mae", type=float, default=FAIL_MAE)
    args = ap.parse_args()

    torch.manual_seed(0)
    net = VectorNet()
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(ck["net"])
    net.eval()
    print(f"[bench] ckpt={args.ckpt} step={ck.get('step', '?')} "
          f"params={sum(p.numel() for p in net.parameters()):,} "
          f"suites={len(SUITES)} scenes/suite={args.num}")

    soft = SoftSVGRenderer(args.size)
    os.makedirs(args.cases_dir, exist_ok=True)
    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False

    names = [s for s in SUITES if (not args.only or s in args.only.split(","))]
    report = {"meta": {"ckpt": args.ckpt, "step": ck.get("step", None),
                       "size": args.size, "num_per_suite": args.num,
                       "base_seed": args.base_seed, "iou_match": IOU_MATCH,
                       "unsupported_features": UNSUPPORTED},
              "suites": {}, "objects": [], "failures": []}

    for si, name in enumerate(names):
        cfg = GeneratorConfig(**SUITES[name])
        rows, objs = [], []
        t0 = time.perf_counter()
        for i in range(args.num):
            seed = args.base_seed + si * 10000 + i
            scene = SceneGenerator(cfg, seed).sample()
            r = eval_scene(net, soft, scene, args.size)
            recs, n_m, n_g, n_p = objects_records(scene, r["scene_pred"])
            row = {k: r[k] for k in ("mae", "ssim_loss", "floor", "net_ms", "vec_ms",
                                     "total_ms", "n_pred", "n_gt")}
            row.update(seed=seed, n_matched=n_m, obj_exact=bool(n_g == n_p),
                       recall=n_m / max(n_g, 1), precision=n_m / max(n_p, 1))
            rows.append(row)
            for rec in recs:
                rec = dict(rec)
                rec["suite"] = name
                rec["seed"] = seed
                objs.append(rec)
            if r["mae"] > args.fail_mae and has_pil:
                tag = f"{name}_{seed}"
                Image.fromarray(r["img_in"].mul(255).byte().permute(1, 2, 0).numpy()
                                [..., :3]).save(f"{args.cases_dir}/{tag}_in.png")
                Image.fromarray(r["img_pred"][..., :3]).save(
                    f"{args.cases_dir}/{tag}_pred.png")
                report["failures"].append({"suite": name, "seed": seed,
                                           "mae": r["mae"], "tag": tag})
        el = time.perf_counter() - t0
        mae = float(np.mean([x["mae"] for x in rows]))
        report["suites"][name] = {
            "config": SUITES[name],
            "mae_mean": mae,
            "mae_p90": float(np.percentile([x["mae"] for x in rows], 90)),
            "ssim_mean": float(1.0 - np.mean([x["ssim_loss"] for x in rows])),
            "floor_mean": float(np.mean([x["floor"] for x in rows])),
            "recall": float(np.mean([x["recall"] for x in rows])),
            "precision": float(np.mean([x["precision"] for x in rows])),
            "obj_exact": float(np.mean([x["obj_exact"] for x in rows])),
            "n_objs_mean": float(np.mean([x["n_gt"] for x in rows])),
            "total_ms_median": float(np.median([x["total_ms"] for x in rows])),
            "total_ms_max": float(np.max([x["total_ms"] for x in rows])),
            "scenes": rows,
        }
        report["objects"].extend(objs)
        print(f"[{name:<18}] mae={mae:.4f} ssim={report['suites'][name]['ssim_mean']:.4f} "
              f"recall={report['suites'][name]['recall']:.2f} "
              f"prec={report['suites'][name]['precision']:.2f} "
              f"exact={report['suites'][name]['obj_exact']:.2f} ({el:.0f}s)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    n_obj = len(report["objects"])
    print(f"\n[saved] {args.out} | suites={len(names)} objects={n_obj} "
          f"failures={len(report['failures'])} -> {args.cases_dir}")


if __name__ == "__main__":
    main()
