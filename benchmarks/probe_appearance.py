"""外观瓶颈梯度探针：实测 fill/bg 各监督通路的梯度强度与当前预测状态。

用法：python benchmarks/probe_appearance.py --ckpt runs/newrep/last.pt --n 8 [--device cuda]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import matched_auxiliary_losses, render_losses, spatial_diversity
from model.network import VectorNet
from model.spec import slots_to_objs, squash_bg, squash_slots
from model.targets import I_FALPHA, I_FRGB, I_FTYPE, I_OPACITY, I_SRGB, encode_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/newrep/last.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    net = VectorNet().to(device)
    ck = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ck["net"])
    net.train()  # 需要梯度

    ren = SoftSVGRenderer(args.size, device=args.device, sub_px=1)
    gen = SceneGenerator(None, 777000)

    acc = {k: [] for k in [
        "bg_rgb_pred", "bg_alpha_pred", "bg_rgb_gt",
        "fill_rgb_pred", "fill_rgb_gt", "fill_n_solid",
        "grad_bg_rgb", "grad_fill_rgb", "grad_fill_alpha",
        "grad_opacity", "grad_stroke_rgb", "grad_geom_coord",
        "obj_coverage", "mae", "mae_obj_region",
    ]}

    for i in range(args.n):
        scene = gen.sample()
        img_gt = ren.render_scene(scene).detach()
        slots_gt, bg_gt = encode_scene(scene)

        slots, aux, bg = net(img_gt.unsqueeze(0).to(device))
        slots = slots[0]
        bg_raw = bg[0]
        slots.retain_grad()
        bg_raw.retain_grad()

        objs, bg_s = slots_to_objs(slots, bg_raw, args.size, 5.0)
        img_pred = ren._render_objs(objs, bg_s)

        total, parts = None, None
        r = render_losses(img_pred, img_gt.to(device), 0.3)
        a = matched_auxiliary_losses(slots, bg_raw, slots_gt, bg_gt)
        aux_total = (0.3 * a["cls"] + 0.3 * a["ftype"] + 0.1 * a["valid"]
                     + 0.1 * a["svalid"] + 0.7 * a["geom"] + 0.2 * a["bg"]
                     + 0.05 * spatial_diversity(slots) + 0.5 * a["bbox"])
        total = r["total"] + aux_total

        # 清掉中间图后对 raw 参数求梯度
        net.zero_grad()
        if bg_raw.grad is not None:
            bg_raw.grad = None
        total.backward()

        f = squash_slots(slots)
        bg_sq = squash_bg(bg_raw).detach().cpu().numpy()

        # 匹配对上的 fill 实测（用 GT 索引近似：前 n_obj 个 slot 即 GT 顺序）
        n_obj = int(sum(1 for o in scene.objects))
        ft_gt = np.asarray(slots_gt)[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
        solid_idx = [k for k in range(min(n_obj, len(ft_gt))) if ft_gt[k] == 1]

        acc["bg_rgb_pred"].append(bg_sq[1:4].tolist())
        acc["bg_alpha_pred"].append(float(bg_sq[4]) if bg_sq.shape[0] >= 5 else float("nan"))
        acc["bg_rgb_gt"].append(np.asarray(bg_gt)[1:4].tolist())

        if solid_idx:
            fr = f["rgb"][torch.tensor(solid_idx, device=device)].detach().cpu().numpy()
            fg = np.asarray(slots_gt)[solid_idx][:, I_FRGB:I_FRGB + 3]
            acc["fill_rgb_pred"].append(fr.mean(0).tolist())
            acc["fill_rgb_gt"].append(fg.mean(0).tolist())
            acc["fill_n_solid"].append(len(solid_idx))
            g = slots.grad[torch.tensor(solid_idx, device=device)][:, I_FRGB:I_FRGB + 3]
            acc["grad_fill_rgb"].append(float(g.norm()))
            g = slots.grad[torch.tensor(solid_idx, device=device)][:, I_FALPHA]
            acc["grad_fill_alpha"].append(float(g.norm()))
        g = slots.grad[:, I_OPACITY]
        acc["grad_opacity"].append(float(g.norm()))
        # stroke rgb（有效 stroke 的槽）
        sv = np.asarray(slots_gt)[:, 233] > 0.5
        if sv.any():
            idx = torch.tensor(np.where(sv)[0], device=device)
            g = slots.grad[idx][:, I_SRGB:I_SRGB + 3]
            acc["grad_stroke_rgb"].append(float(g.norm()))
        # 段坐标梯度（对比基准）
        g = slots.grad[:, 7:7 + 16 * 12]
        acc["grad_geom_coord"].append(float(g.norm()))

        # bg 梯度
        gbg = squash_bg(bg_raw)
        if bg_raw.grad is not None:
            # rgb 三通道的梯度范数（raw 空间）
            acc["grad_bg_rgb"].append(float(bg_raw.grad[1:4].norm()))
        else:
            acc["grad_bg_rgb"].append(0.0)

        # 对象覆盖：对象层 alpha 通道非零像素占比（相对 GT 对象区域）
        with torch.no_grad():
            layers, straight = ren.render_object_layers(scene)
            obj_a = sum(l[3] for l in layers).clamp(0, 1)
            acc["obj_coverage"].append(float(obj_a.mean()))
            mae_full = float((img_pred - img_gt).abs().mean())
            # 对象区域 mae（用 GT 对象层 alpha>0.5 的区域）
            mask = obj_a > 0.5
            if mask.any():
                diff = (img_pred[:3] - img_gt[:3].to(device)).abs().mean(0)
                acc["mae_obj_region"].append(float(diff[mask].mean()))
            acc["mae"].append(mae_full)

    print("=" * 72)
    print(f"ckpt={args.ckpt} n={args.n} device={args.device}")
    print("-" * 72)

    def m(k):
        v = acc[k]
        return float(np.mean(v)) if v else float("nan")

    print(f"bg  GT rgb      : {np.mean(acc['bg_rgb_gt'], 0).round(3)}")
    print(f"bg  pred rgb    : {np.mean(acc['bg_rgb_pred'], 0).round(3)}  alpha={m('bg_alpha_pred'):.3f}")
    print(f"bg  grad(rgb raw): {m('grad_bg_rgb'):.3e}")
    print(f"fill(GT solid n = {m('fill_n_solid'):.1f})")
    print(f"    GT   rgb    : {np.mean(acc['fill_rgb_gt'], 0).round(3)}")
    print(f"    pred rgb    : {np.mean(acc['fill_rgb_pred'], 0).round(3)}")
    print(f"    grad raw-norm: fill_rgb={m('grad_fill_rgb'):.3e} fill_alpha={m('grad_fill_alpha'):.3e}")
    print(f"    grad 对比    : opacity={m('grad_opacity'):.3e} stroke_rgb={m('grad_stroke_rgb'):.3e} geom_coords={m('grad_geom_coord'):.3e}")
    print(f"scene mae={m('mae'):.4f}  obj-region mae={m('mae_obj_region'):.4f}  GT obj coverage={m('obj_coverage'):.3f}")


if __name__ == "__main__":
    main()
