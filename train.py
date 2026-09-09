"""Phase 2 training: online SceneGenerator -> VectorNet -> differentiable render loss."""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import compute_losses
from model.network import VectorNet
from model.spec import slots_to_objs
from model.targets import encode_scene


def parse_args():
    p = argparse.ArgumentParser(description="PNG->SVG vectorizer training (Phase 2)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--size", type=int, default=256, help="train render/input resolution")
    p.add_argument("--sub-px", type=int, default=2, help="renderer sub-pixel samples (grad path)")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="disable per-object activation checkpointing in renderer")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--w-cls", type=float, default=0.3)
    p.add_argument("--w-ftype", type=float, default=0.3)
    p.add_argument("--w-valid", type=float, default=0.1)
    p.add_argument("--w-svalid", type=float, default=0.1)
    p.add_argument("--w-geom", type=float, default=0.7)
    p.add_argument("--w-bg", type=float, default=0.2)
    p.add_argument("--w-div", type=float, default=0.05)
    p.add_argument("--w-bbox", type=float, default=0.5,
                   help="匹配对象 cx/cy 专项 L1 监督权重，提升 bbox 中心精度")
    p.add_argument("--centroid-scale", type=float, default=1.0,
                   help="anchor-free：cross-attn 注意力质心作为空间先验的强度（数据驱动，无网格偏置）")
    p.add_argument("--w-cent", type=float, default=0.05,
                   help="注意力熵正则权重，鼓励每个 slot 聚焦到紧凑区域（防质心扩散回中心）")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--out", type=str, default="runs/phase2")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--pad-px", type=float, default=5.0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def train_step(net, ren, gen, args, device, centroid_scale: float = 1.0):
    scene = gen.sample()
    img_gt = ren.render_scene(scene).detach()
    slots_gt, bg_gt = encode_scene(scene)
    slots, aux, bg = net(img_gt.unsqueeze(0).to(device), centroid_scale=centroid_scale)
    slots = slots[0]
    bg_raw = bg[0]
    cls_ids = aux["cls"][0].detach().argmax(-1).cpu().numpy()
    ftype_ids = aux["ftype"][0].detach().argmax(-1).cpu().numpy()
    objs, bg_s = slots_to_objs(slots, cls_ids, ftype_ids, bg_raw,
                               args.size, args.pad_px)
    img_pred = ren._render_objs(objs, bg_s)
    total, parts = compute_losses(img_pred, img_gt.to(device), slots, bg_raw,
                                  aux, slots_gt, bg_gt,
                                  w_cls=args.w_cls, w_ftype=args.w_ftype,
                                  w_valid=args.w_valid, w_svalid=args.w_svalid,
                                  w_geom=args.w_geom, w_bg=args.w_bg,
                                  w_div=args.w_div, w_bbox=args.w_bbox,
                                  w_cent=args.w_cent)
    total.backward()
    return total.detach(), parts


def save_ckpt(path, net, opt, step, args, gen):
    tmp = path + ".tmp"
    torch.save({
        "step": step,
        "net": net.state_dict(),
        "opt": opt.state_dict(),
        "gen_rng": gen._rng.bit_generator.state,
        "config": {"size": args.size, "sub_px": args.sub_px, "seed": args.seed,
                   "data_seed": args.data_seed, "lr": args.lr},
    }, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"[error] --device {args.device} but CUDA is not available")

    net = VectorNet().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    ren = SoftSVGRenderer(args.size, device=args.device, sub_px=args.sub_px,
                          grad_checkpoint=not args.no_grad_checkpoint)
    gen = SceneGenerator(None, args.data_seed)
    print(f"[train] device={device} params={n_params:,} size={args.size} "
          f"sub_px={args.sub_px} grad_checkpoint={not args.no_grad_checkpoint}")

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=args.device)
        # anchor-free 重构后 ckpt 可能含旧的 spatial_anchor；strict=False 容忍缺失/多余键
        missing, unexpected = net.load_state_dict(ck["net"], strict=False)
        if missing or unexpected:
            print(f"[resume] partial load: missing={missing} unexpected={unexpected}")
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as e:
            print(f"[train] optimizer state skipped (arch mismatch): {e} — fresh optimizer")
        start_step = int(ck["step"])
        gen._rng.bit_generator.state = ck["gen_rng"]
        print(f"[resume] {args.resume} @ step {start_step}")

    log_path = os.path.join(args.out, "log.jsonl")
    keys = ["mae", "ssim", "render", "cls", "ftype", "valid", "svalid",
            "geom", "bbox", "bg", "div", "cent", "aux", "total"]
    avg = {k: 0.0 for k in keys}
    t0 = time.time()
    n_log = 0

    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)

        opt.zero_grad(set_to_none=True)
        total, parts = train_step(net, ren, gen, args, args.device, args.centroid_scale)
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()

        for k in keys:
            avg[k] += parts[k]
        n_log += 1

        if (step + 1) % args.log_every == 0 or step == start_step:
            m = {k: avg[k] / max(1, n_log) for k in keys}
            sps = n_log / max(1e-6, time.time() - t0)
            rec = {"step": step + 1, "lr": opt.param_groups[0]["lr"],
                   "grad_norm": float(gn), "steps_per_s": round(sps, 3),
                   **{k: round(v, 5) for k, v in m.items()}}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{step + 1}/{args.steps}] total={m['total']:.4f} "
                  f"render={m['render']:.4f} mae={m['mae']:.4f} "
                  f"geom={m['geom']:.4f} bbox={m['bbox']:.4f} "
                  f"cent={m['cent']:.3f} gn={float(gn):.2f} "
                  f"({sps:.2f} it/s)", flush=True)
            avg = {k: 0.0 for k in keys}
            n_log = 0
            t0 = time.time()

        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(os.path.join(args.out, "last.pt"), net, opt, step + 1, args, gen)

    save_ckpt(os.path.join(args.out, "last.pt"), net, opt, args.steps, args, gen)
    print(f"[done] checkpoint -> {os.path.join(args.out, 'last.pt')}")


if __name__ == "__main__":
    main()
