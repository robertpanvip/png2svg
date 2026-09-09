"""任务 #25 验证：GT→raw→可微渲染 管线一致性与梯度流。"""
import numpy as np
import torch

from dataset.generator import SceneGenerator, GeneratorConfig
from dataset.renderer import SoftSVGRenderer
from model.targets import encode_scene, NUM_SLOTS
from model.spec import slots_to_objs, targets_to_raw, bg_to_raw

ren = SoftSVGRenderer(256, device="cuda", sub_px=1, grad_checkpoint=False)

maes = []
for seed in range(20):
    sc = SceneGenerator(GeneratorConfig(), seed=seed).sample()
    img_gt = ren.render_scene(sc).detach()
    slots, bg = encode_scene(sc)
    raw = targets_to_raw(slots).cuda().requires_grad_(True)
    objs, bg_s = slots_to_objs(raw, bg_to_raw(bg).cuda(), 256, 5.0)
    img = ren._render_objs(objs, bg_s)
    mae = (img - img_gt.cuda()).abs().mean()
    maes.append(float(mae))
    if seed < 5:
        mae.backward(retain_graph=False)
        g = raw.grad
        assert g is not None and torch.isfinite(g).all(), f"grad nan seed={seed}"
        print(f"seed={seed} mae={mae:.5f} grad_norm={float(g.norm()):.3f}")
    else:
        print(f"seed={seed} mae={mae:.5f}")

print("mean mae:", float(np.mean(maes)), "max:", float(np.max(maes)))
print("OK_25" if np.mean(maes) < 0.02 else "MAE_TOO_HIGH")
