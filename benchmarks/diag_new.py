"""新契约（§11.6, 280 维）验收诊断。

用法:
    python benchmarks/diag_new.py --ckpt runs/newrep/last.pt --device cuda --out runs/diag_new.json
    python benchmarks/diag_new.py --ckpt runs/newrep/last.pt --device cpu --num 30

指标（HANDOFF §11.7 验收判据）:
  - 对象匹配率: Hungarian(bbox L1 + 段类型 NLL) 匹配对数 / GT 对象总数
  - 段类型分布: 预测 vs GT（多样性，检查是否坍缩到单一类型）
  - 逐类型段准确率: 匹配对上, j < min(nseg) 段的类型 argmax 一致率
  - 诊断 mae: 可微渲染 |pred-gt| 均值（与旧表示 0.069 基线对比）
  - oracle_cls_seg: 把预测段类型换成 GT 类型后的 mae 变化（正值=类型分支差于 GT）
"""
import argparse
import collections
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import match_slots
from model.network import VectorNet
from model.spec import slots_to_objs, squash_slots
from model.targets import (I_BBOX, I_NSEG, I_SEG, I_VALID, N_SEG, SEG_DIM,
                           encode_scene)

SEG_NAMES = ["M", "L", "Q", "C", "A"]


def seg_cost_matrix(f, gt, device):
    """与 matched_auxiliary_losses 相同口径的匹配代价。"""
    pred_bbox = torch.stack([f["cx"], f["cy"], f["w"], f["h"]], dim=1).detach()
    gt_seg_all = gt[:, I_SEG:I_SEG + N_SEG * SEG_DIM].reshape(len(gt), N_SEG, SEG_DIM)
    gt_type_all = gt_seg_all[..., :5].argmax(-1)
    nseg_all = gt[:, I_NSEG]
    valid_gt = gt[:, I_VALID] > 0.5

    logp = torch.log_softmax(f["seg_type"].detach(), -1)
    jidx = torch.arange(N_SEG, device=device)
    bbox_cost = (pred_bbox.unsqueeze(1) - gt[None, :, I_BBOX:I_BBOX + 4]) \
        .abs().sum(-1).cpu().numpy()
    cost = np.full((pred_bbox.shape[0], len(gt)), 1e6, dtype=np.float64)
    for g in range(len(gt)):
        if not bool(valid_gt[g]):
            continue
        m = jidx < nseg_all[g]
        lp = logp[:, jidx, gt_type_all[g]]
        type_nll = (-lp[:, m].mean(-1)).cpu().numpy()
        cost[:, g] = bbox_cost[:, g] + type_nll
    return cost, gt_type_all, nseg_all, valid_gt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num", type=int, default=30)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=777000)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", default="runs/diag_new.json")
    args = p.parse_args()

    device = args.device
    # 与旧 diag 相同的容错加载（跨架构 ckpt）
    net = VectorNet()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["net"] if "net" in ck else ck
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[load] dropped {len(unexpected)} unexpected keys: {unexpected[:4]}")
    if missing:
        print(f"[load] missing {len(missing)} keys (random init)")
    net.to(device).eval()

    ren = SoftSVGRenderer(args.size, device=device, sub_px=1, grad_checkpoint=False)
    gen = SceneGenerator(None, args.seed)

    n_gt_total = 0
    n_matched = 0
    mae_sum = 0.0
    oracle_sum = 0.0
    n_scenes = 0
    pred_types = collections.Counter()
    gt_types = collections.Counter()
    seg_hit = collections.Counter()
    seg_tot = collections.Counter()
    nseg_gap = []

    with torch.no_grad():
        for si in range(args.num):
            scene = gen.sample()
            img_gt = ren.render_scene(scene).detach()
            slots_gt, bg_gt = encode_scene(scene)
            n_gt_total += int((np.asarray(slots_gt)[:, I_VALID] > 0.5).sum())
            slots, aux, bg = net(img_gt.unsqueeze(0).to(device))
            slots = slots[0]
            bg_raw = bg[0]

            objs, bg_s = slots_to_objs(slots, bg_raw, args.size, 5.0)
            img_pred = ren._render_objs(objs, bg_s)
            mae = (img_pred - img_gt.to(device)).abs().mean().item()
            mae_sum += mae
            n_scenes += 1

            f = squash_slots(slots)
            gt = torch.as_tensor(np.asarray(slots_gt), dtype=torch.float32,
                                 device=device)
            cost, gt_type_all, nseg_all, valid_gt = seg_cost_matrix(f, gt, device)
            assign = match_slots(cost)

            # oracle: 预测段类型换成 GT 类型（匹配对上）
            gt_seg_all = gt[:, I_SEG:I_SEG + N_SEG * SEG_DIM] \
                .reshape(len(gt), N_SEG, SEG_DIM)
            for g, k in enumerate(assign):
                if k < 0 or not bool(valid_gt[g]):
                    continue
                n_matched += 1
                nseg_g = int(nseg_all[g].item())
                kp = f["seg_type"][k].argmax(-1)          # [16]
                kp_oracle = gt_type_all[g].clone()
                # oracle 混合渲染：类型 one-hot 换 GT，坐标保留预测
                k_orig = kp.clone()
                f["seg_type"][k] = torch.eye(5, device=device)[kp_oracle]
                objs_o, _ = slots_to_objs(slots, bg_raw, args.size, 5.0)
                img_o = ren._render_objs(objs_o, bg_s)
                mae_o = (img_o - img_gt.to(device)).abs().mean().item()
                oracle_sum += mae_o - mae
                f["seg_type"][k] = torch.eye(5, device=device)[k_orig]

                for j in range(nseg_g):
                    tp = int(kp[j].item())
                    tg = int(gt_type_all[g, j].item())
                    pred_types[SEG_NAMES[tp]] += 1
                    gt_types[SEG_NAMES[tg]] += 1
                    seg_tot[SEG_NAMES[tg]] += 1
                    if tp == tg:
                        seg_hit[SEG_NAMES[tg]] += 1
                # nseg gap（预测对象有效段数 vs GT）
                nseg_p = int(round(float(f["nseg"][k].item())))
                nseg_gap.append(nseg_p - nseg_g)

    dist = {
        "mae_mean": mae_sum / max(n_scenes, 1),
        "match_rate": n_matched / max(n_gt_total, 1),
        "pred_seg_types": dict(pred_types),
        "gt_seg_types": dict(gt_types),
        "seg_type_acc": {k: round(seg_hit[k] / max(seg_tot[k], 1), 3)
                         for k in seg_tot},
        "nseg_gap_mean": float(np.mean(nseg_gap)) if nseg_gap else 0.0,
        "oracle_seg": round(oracle_sum / max(n_matched, 1), 5),
        "num_scenes": n_scenes,
        "ckpt": args.ckpt,
    }
    print(f"mae_mean      = {dist['mae_mean']:.4f}   (旧表示基线 0.069)")
    print(f"match_rate    = {dist['match_rate']:.3f}  ({n_matched}/{n_gt_total})")
    print(f"pred types    = {dict(pred_types)}")
    print(f"gt  types     = {dict(gt_types)}")
    print(f"seg_type_acc  = {dist['seg_type_acc']}")
    print(f"nseg_gap_mean = {dist['nseg_gap_mean']:+.2f}（预测段数 - GT 段数）")
    print(f"oracle_seg    = {dist['oracle_seg']:+.4f}（正=类型分支差于 GT）")

    import json
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dist, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
