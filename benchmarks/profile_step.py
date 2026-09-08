"""Profile one training step segment-by-segment on GPU.

Run while main training is idle (or accept contention): prints per-segment
wall time (CUDA-synced) averaged over N steps, plus VRAM peak.
Usage: python benchmarks/profile_step.py [--steps 20] [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import compute_losses
from model.network import VectorNet
from model.spec import slots_to_objs
from model.targets import encode_scene


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--sub-px", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234)
    return p.parse_args()


class Timer:
    def __init__(self):
        self.totals = {}

    def span(self, name, fn):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.totals.setdefault(name, []).append(time.perf_counter() - t0)
        return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    net = VectorNet().to(device).eval()
    ren = SoftSVGRenderer(args.size, device=args.device, sub_px=args.sub_px,
                          grad_checkpoint=not args.no_grad_checkpoint)
    gen = SceneGenerator(None, args.data_seed)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

    T = Timer()
    warm = 3
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)

        scene = T.span("1 gen.sample (cpu)", lambda: gen.sample())
        img_gt = T.span("2 render_scene GT", lambda: ren.render_scene(scene).detach())
        slots_gt, bg_gt = T.span("3 encode_scene (cpu)", lambda: encode_scene(scene))

        def _net():
            s, aux, bg = net(img_gt.unsqueeze(0).to(device))
            return s[0], aux, bg[0]
        slots, aux, bg_raw = T.span("4 net forward", _net)

        def _objs():
            cls_ids = aux["cls"][0].detach().argmax(-1).cpu().numpy()
            ftype_ids = aux["ftype"][0].detach().argmax(-1).cpu().numpy()
            return slots_to_objs(slots, cls_ids, ftype_ids, bg_raw,
                                 args.size, 5.0), cls_ids, ftype_ids
        (objs, bg_s), cls_ids, ftype_ids = T.span("5 slots_to_objs (cpu)", _objs)

        img_pred = T.span("6 _render_objs diff", lambda: ren._render_objs(objs, bg_s))
        total, parts = T.span(
            "7 compute_losses", lambda: compute_losses(
                img_pred, img_gt.to(device), slots, bg_raw, aux, slots_gt, bg_gt))

        def _bwd():
            total.backward()
        T.span("8 backward", _bwd)
        T.span("9 opt.step", lambda: opt.step())

        if i == warm - 1:  # reset stats after warmup
            T.totals = {k: v[warm:] for k, v in T.totals.items()}

    rows = {k: np.mean(v) * 1000 for k, v in T.totals.items() if len(v) > 0}
    tot = sum(rows.values())
    print(f"\n[profile] {args.steps} steps (warmup {warm}) size={args.size} "
          f"sub_px={args.sub_px} grad_ckpt={not args.no_grad_checkpoint}")
    for k, v in rows.items():
        print(f"  {k:<28} {v:8.1f} ms  {100*v/tot:5.1f}%")
    print(f"  {'TOTAL (synced)':<28} {tot:8.1f} ms  -> {1000/tot:.2f} it/s equivalent")
    if torch.cuda.is_available():
        print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB / "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
