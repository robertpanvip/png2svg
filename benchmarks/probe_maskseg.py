"""E 方案门控探针 v2（HANDOFF §11.16）：过拟合少量场景验证——
1) 掩码头 IoU 能否爬升（门控：overfit IoU >= 0.60）
2) 取色天花板分层：token 粒度 vs 像素粒度（掩码上采样 256 逐像素池化，
   ± erode 腐蚀切描边），对比灰均值基线 0.26

v1 教训（2026-09-11）：32×32 token 粒度下 GT 掩码 oracle L1 也有 0.257
（≈灰基线）——粒度不足时赢了掩码也赢不了取色；且训练从 150 步起静默
冻结（gn NaN 跳步未打印）、eval 贪心匹配有槽位复用 bug。

用法：python benchmarks/probe_maskseg.py [--ckpt runs/genA40k/last.pt]
    [--scenes 4] [--steps 400] [--lr 1e-3] [--mask-only] [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import sys

# 不依赖调用方 cwd：以脚本位置定位项目根（benchmarks/ 的上一级）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import compute_losses
from model.network import VectorNet
from model.spec import slots_to_objs
from model.targets import I_FTYPE, I_FRGB, NUM_SLOTS, encode_scene

ERODE_K = 7   # 像素粒度腐蚀核（切掉 ~3px 描边过渡带）


def _pixel_pool(img: torch.Tensor, m32: torch.Tensor, erode: bool = False):
    """m32 [32,32] → 上采样 256 后逐像素加权池化 img[:3]，返回 [3] 颜色。"""
    m = F.interpolate(m32.reshape(1, 1, *m32.shape[-2:]),
                      size=img.shape[-2:], mode="bilinear",
                      align_corners=False)                  # [1,1,256,256]
    if erode:
        m = -F.max_pool2d(-m, ERODE_K, stride=1, padding=ERODE_K // 2)
    m = m[0, 0].clamp(0.0, 1.0)
    s = m.sum().clamp_min(1e-6)
    return torch.stack([(img[c] * m).sum() / s for c in range(3)])


def eval_scene(net, img_gt, masks_gt, slots_gt, device):
    """返回 dict：ious / L1_{pm_tok, pm_pix, gm_pix, gm_pix_er, gray} / fill L1。"""
    with torch.no_grad():
        slots, aux, bg = net(img_gt.unsqueeze(0).to(device), anchor_scale=1.0)
        slots = slots[0]
        pm = torch.sigmoid(aux["mask"][0])                    # [K,32,32]
        n_obj = int((np.asarray(slots_gt)[:, 0] > 0.5).sum())
        gt_m = torch.zeros(NUM_SLOTS, 32, 32, device=device)
        gt_m[:n_obj] = masks_gt[:n_obj].to(device)

        # IoU 矩阵 + 不复用贪心匹配（按 IoU 全局降序取对）
        pf = pm.reshape(NUM_SLOTS, -1)
        gf = gt_m.reshape(NUM_SLOTS, -1)
        inter = pf @ gf.T
        union = pf.sum(1, keepdim=True) + gf.sum(1).unsqueeze(0) - inter
        iou = (inter / union.clamp_min(1e-6)).cpu().numpy()
        pairs = sorted(((float(iou[k, g]), k, g)
                        for k in range(NUM_SLOTS) for g in range(n_obj)),
                       reverse=True)
        assign = {}
        used_k = set()
        for v, k, g in pairs:
            if g in assign or k in used_k or v <= 0.05:
                continue
            assign[g] = k
            used_k.add(k)
        ious = [v for v, k, g in pairs if g in assign and assign[g] == k]

        img = img_gt.to(device)
        gray = torch.full((3,), 0.37, device=device)          # 数据集 fill 均值灰（§11.11 基线）
        gt_np = np.asarray(slots_gt)
        ft = gt_np[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
        solid = [g for g in range(n_obj) if ft[g] == 1]
        out = {"iou": ious, "pm_tok": [], "pm_pix": [], "gm_pix": [],
               "gm_pix_er": [], "gray": [], "fill": []}
        for g in solid:
            gt_rgb = torch.tensor(gt_np[g, I_FRGB:I_FRGB + 3], device=device)
            # 与匹配解耦的参照系（全量 solid 对象，避免样本集漂移）
            c_gm = _pixel_pool(img, gt_m[g])
            c_gm_er = _pixel_pool(img, gt_m[g], erode=True)
            out["gm_pix"].append(float((c_gm - gt_rgb).abs().mean()))
            out["gm_pix_er"].append(float((c_gm_er - gt_rgb).abs().mean()))
            out["gray"].append(float((gray - gt_rgb).abs().mean()))
            k = assign.get(g)
            if k is None:
                continue
            # token 粒度（模型 cf_mask 现状）
            raw32 = F.avg_pool2d(img, 8)
            w = pm[k].unsqueeze(0)
            ws = w.sum().clamp_min(1e-6)
            c_tok = torch.stack([(raw32[c] * w).sum() / ws for c in range(3)])
            out["pm_tok"].append(float((c_tok - gt_rgb).abs().mean()))
            # 像素粒度（预测掩码）
            c_pix = _pixel_pool(img, pm[k])
            out["pm_pix"].append(float((c_pix - gt_rgb).abs().mean()))
            out["fill"].append(float(
                (slots[k, I_FRGB:I_FRGB + 3] - gt_rgb).abs().mean()))
        return out


def merge(dsts):
    out = {}
    for d in dsts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def report(tag, m):
    def f(key):
        return float(np.mean(m[key])) if m.get(key) else -1.0
    print(f"{tag} IoU={f('iou'):.3f} tok={f('pm_tok'):.3f} pix={f('pm_pix'):.3f} "
          f"| oracle pix={f('gm_pix'):.3f} pix+er={f('gm_pix_er'):.3f} "
          f"| gray={f('gray'):.3f} fillhead={f('fill'):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/genA40k/last.pt")
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-only", action="store_true",
                    help="冻结掩码头以外全部参数（隔离头容量问题）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    device = args.device
    net = VectorNet().to(device)
    if args.ckpt and os.path.isfile(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        state = ck["net"]
        ms = net.state_dict()
        skip = [k for k, v in state.items() if k in ms and ms[k].shape != v.shape]
        new = [k for k in ms if k not in state]
        state = {k: v for k, v in state.items() if k not in skip}
        net.load_state_dict(state, strict=False)
        print(f"[probe] resume {args.ckpt} (skip {len(skip)}, fresh {len(new)})")
    net.train()
    if args.mask_only:
        for n_, p in net.named_parameters():
            p.requires_grad_(n_.startswith("mask_"))
        print("[probe] mask-only: 冻结掩码头以外全部参数")

    ren = SoftSVGRenderer(256, device=device, sub_px=1, grad_checkpoint=False)
    gen = SceneGenerator(None, args.seed)
    data = []
    for _ in range(args.scenes):
        scene = gen.sample()
        img_gt = ren.render_scene(scene).detach()
        masks_gt = ren.render_fill_masks(scene, 32).cpu()
        slots_gt, bg_gt = encode_scene(scene)
        data.append((scene, img_gt, masks_gt, slots_gt, bg_gt))

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    keys = ["mae", "render", "geom", "appearance", "mask", "bbox", "total"]
    avg = {k: 0.0 for k in keys}
    n_log = 0
    n_skip_gn = 0

    def evaluate(tag):
        m = merge([eval_scene(net, img_gt, mk, sg, device)
                   for _, img_gt, mk, sg, _ in data])
        report(tag, m)

    evaluate("step    0")
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        skip_step = False
        for scene, img_gt, masks_gt, slots_gt, bg_gt in data:
            slots, aux, bg = net(img_gt.unsqueeze(0).to(device), anchor_scale=1.0)
            slots = slots[0]
            if not (torch.isfinite(slots).all() and torch.isfinite(bg[0]).all()):
                print(f"step {step}: non-finite slots, skip scene", flush=True)
                skip_step = True
                break
            objs, bg_s = slots_to_objs(slots, bg[0], 256, 5.0)
            img_pred = ren._render_objs(objs, bg_s)
            total, parts = compute_losses(img_pred, img_gt.to(device), slots, bg[0],
                                          slots_gt, bg_gt, w_fill=2.0, w_mask=1.5,
                                          palette_logits=aux.get("palette"),
                                          mask_logits=aux.get("mask"),
                                          masks_gt=masks_gt.numpy())
            if not torch.isfinite(total):
                print(f"step {step}: non-finite loss, skip scene", flush=True)
                skip_step = True
                break
            (total / len(data)).backward()
            for k in keys:
                avg[k] += float(parts[k])
            n_log += 1
        if skip_step:
            opt.zero_grad(set_to_none=True)
            continue
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            n_skip_gn += 1
            opt.zero_grad(set_to_none=True)
            if n_skip_gn % 10 == 1:
                print(f"step {step}: non-finite grad norm (skip #{n_skip_gn})",
                      flush=True)
            continue
        opt.step()
        if step % 50 == 0:
            m = {k: avg[k] / max(1, n_log) for k in keys}
            print(f"step {step:4d} mask={m['mask']:.4f} mae={m['mae']:.4f} "
                  f"gn={float(gn):.2f} skips={n_skip_gn}", flush=True)
            evaluate(f"         ")
            avg = {k: 0.0 for k in keys}
            n_log = 0

    print("\n==== GATE (E4 v2) ====")
    m = merge([eval_scene(net, img_gt, mk, sg, device)
               for _, img_gt, mk, sg, _ in data])
    f = lambda key: float(np.mean(m[key])) if m.get(key) else -1.0
    iou, pix = f("iou"), f("pm_pix")
    print(f"overfit mask IoU        = {iou:.3f}  (pass >= 0.60)")
    print(f"pred-mask pixel-pool L1 = {pix:.3f}  (gray {f('gray'):.3f}, "
          f"oracle pix {f('gm_pix'):.3f}, oracle erode {f('gm_pix_er'):.3f})")
    ok = iou >= 0.60 and pix < max(0.20, 0.8 * f("gray"))
    print(f"GO = {ok}")


if __name__ == "__main__":
    main()
