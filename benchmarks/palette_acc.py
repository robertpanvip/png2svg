"""色板分类精度快速判定（genA_pal ckpt）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.network import VectorNet
from model.targets import I_FTYPE, I_FRGB, encode_scene
from model.palette import N_PALETTE, rgb_to_id


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/genA_pal/last.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = VectorNet().to(device)
    ck = torch.load(ckpt, map_location="cpu")
    ms = net.state_dict()
    skip = [k for k, v in ck["net"].items() if k in ms and ms[k].shape != v.shape]
    net.load_state_dict({k: v for k, v in ck["net"].items() if k not in skip}, strict=False)
    net.eval()
    ren = SoftSVGRenderer(256, device=device, sub_px=1)
    gen = SceneGenerator(None, 777000)
    correct = total = hue_ok = 0
    with torch.no_grad():
        for _ in range(60):
            scene = gen.sample()
            img = ren.render_scene(scene)
            slots_gt, _ = encode_scene(scene)
            gt = np.asarray(slots_gt)
            ft = gt[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
            slots, aux, bg = net(img.unsqueeze(0).to(device))
            preds = aux["palette"][0].argmax(-1).cpu().numpy()
            pred_set = set(int(p) for p in preds)
            for k in range(min(len(scene.objects), 8)):
                if ft[k] != 1:
                    continue
                gid = int(rgb_to_id(gt[k, I_FRGB:I_FRGB + 3][None])[0])
                total += 1
                if gid in pred_set:
                    correct += 1
                # hue 12 档判定（色相错但明度/饱和度对也给部分分）
                if (gid // 4) in set(int(p) // 4 for p in preds):
                    hue_ok += 1
    print(f"ckpt={ckpt}")
    print(f"n solid = {total}")
    print(f"top-1 命中率（GT id ∈ 预测集合）: {correct / max(total, 1):.3f}  随机={min(8 / N_PALETTE, 1.0):.3f}")
    print(f"hue 档命中率: {hue_ok / max(total, 1):.3f}  随机≈{8 / 12:.3f}")


if __name__ == "__main__":
    main()
