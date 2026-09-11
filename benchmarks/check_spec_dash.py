"""P3b dash 验证（HANDOFF §11.17）：dash 接入 soft 渲染路径的一致性/梯度/resvg 对齐。"""
import numpy as np
import torch

from dataset.generator import SceneGenerator, GeneratorConfig
from dataset.renderer import SoftSVGRenderer
from model.targets import encode_scene, decode_scene, NUM_SLOTS
from model.spec import slots_to_objs, targets_to_raw, bg_to_raw
from svg.render import render_scene_resvg

ren = SoftSVGRenderer(256, device="cuda", sub_px=1, grad_checkpoint=False)

maes, dash_scenes = [], 0
resvg_maes = []
for seed in range(30):
    cfg = GeneratorConfig(stroke_attr_prob=1.0)
    sc = SceneGenerator(cfg, seed=seed).sample()
    has_dash = any(o.stroke is not None and len(o.stroke.dash) > 0 for o in sc.objects)
    dash_scenes += int(has_dash)

    img_gt = ren.render_scene(sc).detach()          # GT 直渲（含 dash 调制）
    slots, bg = encode_scene(sc)
    raw = targets_to_raw(slots).cuda().requires_grad_(True)
    objs, bg_s = slots_to_objs(raw, bg_to_raw(bg).cuda(), 256, 5.0)
    img = ren._render_objs(objs, bg_s)              # GT→raw→解码→soft（同一路径）
    mae = (img - img_gt.cuda()).abs().mean()
    maes.append(float(mae))

    if has_dash:
        # resvg 对齐：decode_scene → serialize → resvg vs soft GT
        scene_d = decode_scene(slots, bg, canvas=256)
        rgba_u8 = render_scene_resvg(scene_d, 256, 256)          # HWC uint8
        rgba = torch.tensor(rgba_u8, dtype=torch.float32, device="cuda")
        rgba = rgba.permute(2, 0, 1) / 255.0                     # → CHW [0,1]
        resvg_maes.append(float((rgba - img_gt.cuda()).abs().mean()))

    if seed < 8:
        mae.backward(retain_graph=False)
        g = raw.grad
        assert g is not None and torch.isfinite(g).all(), f"grad nan seed={seed}"
        # dash 槽梯度应非零（当场景含 dash 时）
        dash_grad = g[:, 239:244].abs().sum()
        print(f"seed={seed} dash={has_dash} mae={mae:.5f} "
              f"grad_norm={float(g.norm()):.3f} dash_grad={float(dash_grad):.4f}")
    else:
        mae.backward(retain_graph=False)

print(f"dash scenes: {dash_scenes}/30")
print(f"soft-pipeline mae: mean={np.mean(maes):.5f} max={np.max(maes):.5f}")
if resvg_maes:
    print(f"soft-vs-resvg mae (dash scenes, n={len(resvg_maes)}): "
          f"mean={np.mean(resvg_maes):.5f} max={np.max(resvg_maes):.5f}")

ok = np.mean(maes) < 0.02 and all(np.isfinite(maes))
print("OK_P3B" if ok else "FAIL_P3B")
