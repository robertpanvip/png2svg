"""单场景过拟合测试：分辨颜色通路是"优化断了"还是"h 缺颜色信息"。

固定一个场景，用 appearance loss + render loss 训练整网 600 步。
- 若 fill 颜色收敛到 GT → 通路健康，问题是跨场景条件化（h 无颜色信息）
- 若仍卡灰 → 优化/梯度链路断裂
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import render_losses, spatial_diversity
from model.network import VectorNet
from model.spec import slots_to_objs, squash_slots
from model.targets import I_FTYPE, encode_scene
from model.losses import matched_auxiliary_losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VectorNet().to(device)
    ck = torch.load("runs/newrep/last.pt", map_location=device)
    net.load_state_dict(ck["net"])
    net.train()

    ren = SoftSVGRenderer(256, device=str(device), sub_px=1)
    gen = SceneGenerator(None, 777000)
    scene = gen.sample()
    img_gt = ren.render_scene(scene).detach()
    slots_gt, bg_gt = encode_scene(scene)
    gt_np = np.asarray(slots_gt)
    ft_gt = gt_np[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
    n_obj = len(scene.objects)
    print(f"scene: {n_obj} objects, ftypes={ft_gt[:n_obj].tolist()}")

    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.0)

    for step in range(601):
        opt.zero_grad(set_to_none=True)
        slots, aux, bg = net(img_gt.unsqueeze(0).to(device))
        slots = slots[0]
        bg_raw = bg[0]
        slots.retain_grad()
        objs, bg_s = slots_to_objs(slots, bg_raw, 256, 5.0)
        img_pred = ren._render_objs(objs, bg_s)
        r = render_losses(img_pred, img_gt.to(device), 0.3)
        a = matched_auxiliary_losses(slots, bg_raw, slots_gt, bg_gt)
        loss = r["total"] + 2.0 * a["appearance"] + a["geom"] * 0.7 \
            + 0.3 * a["cls"] + 0.3 * a["ftype"] + 0.5 * a["bbox"] + 0.1 * a["valid"]
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skip")
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):
            print(f"step {step}: non-finite grad norm, skip")
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        if step % 100 == 0:
            f = squash_slots(slots)
            solid = [k for k in range(min(n_obj, 8)) if ft_gt[k] == 1]
            msg = f"step {step:4d} app={float(a['appearance']):.4f} mae={float(r['mae']):.4f}"
            if solid:
                idx = torch.tensor(solid, device=device)
                pr = f["rgb"][idx].detach().cpu().numpy().mean(0).round(3)
                gr = gt_np[solid][:, 203:206].mean(0).round(3)
                g = slots.grad[idx][:, 203:206].norm().item()
                msg += f" fill_pred={pr} gt={gr} grad={g:.2e}"
            print(msg, flush=True)


if __name__ == "__main__":
    main()
