"""隔离实验：固定 10 场景，只训 fill/stroke 头 + HR 分支（其余冻结）。

若这样都学不会 → 训练机械链路有 bug；若能学会 → 在线分布下的问题
（如 HR 分支在在线训练中学不到定位）。
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import matched_auxiliary_losses
from model.network import VectorNet
from model.targets import I_FTYPE, I_FRGB, encode_scene


def main():
    ONLINE = len(sys.argv) > 1 and sys.argv[1] == "online"
    print(f"mode: {'online' if ONLINE else 'fixed-10-scenes'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VectorNet().to(device)
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "runs/newrep/last.pt"
    ck = torch.load(ckpt, map_location="cpu")
    model_state = net.state_dict()
    skip = [k for k, v in ck["net"].items()
            if k in model_state and model_state[k].shape != v.shape]
    net.load_state_dict({k: v for k, v in ck["net"].items() if k not in skip},
                        strict=False)

    ren = SoftSVGRenderer(256, device=str(device), sub_px=1)
    gen = SceneGenerator(None, 777000)
    scenes = []
    for _ in range(10):
        scene = gen.sample()
        img = ren.render_scene(scene).detach()
        slots_gt, bg_gt = encode_scene(scene)
        scenes.append((scene, img, slots_gt, bg_gt))

    # 冻结主干，只训 fill/stroke 头 + HR 分支
    for name, p in net.named_parameters():
        p.requires_grad_(name.startswith(("fill_head", "stroke_head", "hr_")))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=5e-4, weight_decay=0.0)

    ACCUM = 8
    opt.zero_grad(set_to_none=True)
    for step in range(2001):
        scene, img, slots_gt, bg_gt = scenes[step % 10]
        if ONLINE:
            scene = gen.sample()
            img = ren.render_scene(scene).detach()
            slots_gt, bg_gt = encode_scene(scene)
        slots, aux, bg = net(img.unsqueeze(0).to(device), anchor_scale=1.0)
        slots = slots[0]
        bg_raw = bg[0]
        a = matched_auxiliary_losses(slots, bg_raw, slots_gt, bg_gt)
        loss = 2.0 * a["appearance"] / ACCUM
        loss.backward()
        if (step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad], 5.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % 150 == 0:
            gt = np.asarray(slots_gt)
            ft = gt[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
            from model.spec import squash_slots
            with torch.no_grad():
                f = squash_slots(slots)
                solid = [g for g in range(8) if ft[g] == 1]
                msg = f"step {step:5d} app={float(a['appearance']):.4f}"
                if solid:
                    pr = np.array([f["rgb"][g].cpu().numpy() for g in solid])
                    gr = gt[solid][:, I_FRGB:I_FRGB + 3]
                    msg += f" pred0={pr[0].round(2).tolist()} gt0={gr[0].round(2).tolist()} meanL1={np.abs(pr-gr).mean():.3f}"
                print(msg, flush=True)


if __name__ == "__main__":
    main()
