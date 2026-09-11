"""E 方案冒烟测试：掩码头形状 / 参数预算 / 单步训练（CPU）。"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import compute_losses
from model.network import VectorNet, count_params
from model.spec import slots_to_objs
from model.targets import encode_scene


def main():
    device = "cpu"
    net = VectorNet().to(device)
    n = count_params(net)
    print(f"params={n:,}  (budget 3M..8M: {3_000_000 <= n <= 8_000_000})")

    ren = SoftSVGRenderer(256, device=device, sub_px=1)
    gen = SceneGenerator(None, 42)
    scene = gen.sample()
    img_gt = ren.render_scene(scene).detach()
    masks_gt = ren.render_fill_masks(scene, out_size=32)
    slots_gt, bg_gt = encode_scene(scene)
    n_obj = len(scene.objects)
    print(f"scene objects={n_obj}, masks shape={tuple(masks_gt.shape)}, "
          f"mask cover mean={masks_gt[:n_obj].mean():.4f}")

    slots, aux, bg = net(img_gt.unsqueeze(0))
    print(f"slots={tuple(slots.shape)} mask_logits={tuple(aux['mask'].shape)} "
          f"palette={tuple(aux['palette'].shape)} bg={tuple(bg.shape)}")
    assert aux["mask"].shape == (1, net.num_slots, 32, 32)

    # 掩码 GT pad 检查（encode_scene pad 到 NUM_SLOTS）
    K = net.num_slots
    gt_m = np.zeros((K, 32, 32), dtype=np.float32)
    gt_m[:n_obj] = masks_gt.numpy()
    print(f"gt pad ok: {gt_m.shape}")

    slots_raw = slots[0]
    bg_raw = bg[0]
    objs, bg_s = slots_to_objs(slots_raw, bg_raw, 256, 5.0)
    img_pred = ren._render_objs(objs, bg_s)
    total, parts = compute_losses(img_pred, img_gt, slots_raw, bg_raw,
                                  slots_gt, bg_gt, w_mask=1.0,
                                  palette_logits=aux.get("palette"),
                                  mask_logits=aux.get("mask"),
                                  masks_gt=masks_gt.numpy())
    print("loss parts:", {k: round(v, 5) for k, v in parts.items()})
    assert np.isfinite(parts["mask"]), "mask loss not finite"

    total.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9)
    mk = [k for k, p in net.named_parameters() if "mask" in k]
    print(f"grad_norm={float(gnorm):.3f} mask params: {mk}")
    for k in mk:
        p = dict(net.named_parameters())[k]
        print(f"  {k}: grad={'None' if p.grad is None else f'{p.grad.norm():.3e}'}")

    # 计时：掩码 GT 生成开销
    t0 = time.time()
    for _ in range(20):
        s = gen.sample()
        ren.render_fill_masks(s, out_size=32)
    dt = (time.time() - t0) / 20
    print(f"render_fill_masks avg {dt*1000:.1f} ms/scene")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
