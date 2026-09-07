"""Phase 2 evaluation: prediction -> Scene -> SVG -> resvg, quality + CPU latency."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import _ssim_mean
from model.network import VectorNet
from model.spec import predictions_to_targets
from model.targets import decode_scene, encode_scene
from svg.render import render_scene_resvg


def parse_args():
    p = argparse.ArgumentParser(description="PNG->SVG eval: quality vs GT floor + CPU latency")
    p.add_argument("--ckpt", type=str, default="runs/phase2/last.pt")
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=4321)
    p.add_argument("--out", type=str, default="", help="optional JSON report path")
    return p.parse_args()


def to_tensor_rgba(arr_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr_u8.astype(np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def evaluate_one(net, soft, scene, size):
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
    ssim = float(1.0 - _ssim_mean(img_pred, img_in))

    slots_gt, bg_gt = encode_scene(scene)
    scene_gt = decode_scene(slots_gt, bg_gt, canvas=size)
    rgba_gt = render_scene_resvg(scene_gt, size, size)
    floor = float((to_tensor_rgba(rgba_gt) - img_in).abs().mean())

    return {"mae": mae, "ssim_loss": ssim, "floor": floor,
            "net_ms": t_net * 1e3, "vec_ms": t_vec * 1e3,
            "total_ms": (t_net + t_vec) * 1e3,
            "n_objs": int((slots_np[:, 0] > 0.5).sum()),
            "n_gt": int((slots_gt[:, 0] > 0.5).sum())}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    net = VectorNet()
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(ck["net"])
    net.eval()
    print(f"[eval] ckpt={args.ckpt} step={ck.get('step', '?')} "
          f"params={sum(p.numel() for p in net.parameters()):,}")

    soft = SoftSVGRenderer(args.size)
    gen = SceneGenerator(None, args.data_seed)

    warm = evaluate_one(net, soft, gen.sample(), args.size)
    print(f"[warmup] total_ms={warm['total_ms']:.0f}")

    rows = []
    for i in range(args.num):
        r = evaluate_one(net, soft, gen.sample(), args.size)
        rows.append(r)
        print(f"scene {i + 1:02d}: mae={r['mae']:.4f} (floor {r['floor']:.4f}) "
              f"ssim={1 - r['ssim_loss']:.4f} objs={r['n_objs']}/{r['n_gt']} "
              f"net={r['net_ms']:6.1f}ms vec={r['vec_ms']:6.1f}ms "
              f"total={r['total_ms']:7.1f}ms")

    tot = np.array([r["total_ms"] for r in rows])
    agg = {
        "mae_mean": float(np.mean([r["mae"] for r in rows])),
        "floor_mean": float(np.mean([r["floor"] for r in rows])),
        "ssim_mean": float(np.mean([1 - r["ssim_loss"] for r in rows])),
        "total_ms_mean": float(tot.mean()),
        "total_ms_median": float(np.median(tot)),
        "total_ms_p95": float(np.percentile(tot, 95)),
        "total_ms_max": float(tot.max()),
        "net_ms_mean": float(np.mean([r["net_ms"] for r in rows])),
        "vec_ms_mean": float(np.mean([r["vec_ms"] for r in rows])),
        "under_1s_ratio": float((tot < 1000).mean()),
        "num_scenes": args.num,
        "size": args.size,
    }
    print("\n[summary]")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"aggregate": agg, "per_scene": rows}, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
