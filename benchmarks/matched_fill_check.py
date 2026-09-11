"""匹配对 fill 颜色检查：判断颜色未学会是"慢"还是"匹配抖动/冲突"。

复现 losses.py 的 Hungarian 匹配，逐对打印 pred vs GT fill rgb。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.matching import match_slots
from model.network import VectorNet
from model.spec import squash_slots
from model.targets import I_BBOX, I_FALPHA, I_FRGB, I_FTYPE, I_NSEG, I_SEG, I_VALID, N_SEG, SEG_DIM, encode_scene


def matched_pairs(slots_raw, gt, device):
    f = squash_slots(slots_raw)
    gv = torch.from_numpy(np.asarray(gt)[:, I_VALID]).to(device)
    valid_gt = gv > 0.5
    pred_bbox = torch.stack([f["cx"], f["cy"], f["w"], f["h"]], dim=1).detach()
    gt_seg_all = torch.from_numpy(np.asarray(gt)[:, I_SEG:I_SEG + N_SEG * SEG_DIM]) \
        .reshape(-1, N_SEG, SEG_DIM).to(device)
    gt_type_all = gt_seg_all[..., :5].argmax(-1)
    nseg_all = torch.from_numpy(np.asarray(gt)[:, I_NSEG]).to(device)
    n = slots_raw.shape[0]
    with torch.no_grad():
        logp = torch.log_softmax(f["seg_type"].detach(), -1)
        bbox_cost = (pred_bbox.unsqueeze(1)
                     - torch.from_numpy(np.asarray(gt)[:, I_BBOX:I_BBOX + 4]).to(device)[None]).abs().sum(-1).cpu().numpy()
        jidx = torch.arange(N_SEG, device=device)
        cost = np.full((n, n), 1e6)
        for g in range(n):
            if not bool(valid_gt[g]):
                continue
            m = jidx < nseg_all[g]
            lp = logp[:, jidx, gt_type_all[g]]
            cost[:, g] = bbox_cost[:, g] + (-lp[:, m].mean(-1)).cpu().numpy()
    return match_slots(cost)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VectorNet().to(device)
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "runs/newrep/last.pt"
    ck = torch.load(ckpt, map_location=device)
    model_state = net.state_dict()
    skip = [k for k, v in ck["net"].items()
            if k in model_state and model_state[k].shape != v.shape]
    if skip:
        print(f"[load] shape-mismatch re-init: {skip}")
    net.load_state_dict({k: v for k, v in ck["net"].items() if k not in skip},
                        strict=False)
    net.eval()

    ren = SoftSVGRenderer(256, device=str(device), sub_px=1)
    gen = SceneGenerator(None, 777000)

    n_scene = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    solid_l1 = []
    gray_l1 = []
    with torch.no_grad():
        for si in range(n_scene):
            scene = gen.sample()
            img = ren.render_scene(scene)
            slots_gt, _ = encode_scene(scene)
            slots, _, bg = net(img.unsqueeze(0).to(device))
            slots = slots[0]
            f = squash_slots(slots)
            assign = matched_pairs(slots, slots_gt, device)
            gt = np.asarray(slots_gt)
            print(f"--- scene {si} ({sum(1 for o in scene.objects)} obj) ---")
            for g in range(len(assign)):
                k = assign[g]
                if k < 0:
                    continue
                ft = gt[g, I_FTYPE:I_FTYPE + 4].argmax()
                name = ["none", "solid", "linear", "radial"][ft]
                if ft == 1:
                    pr = f["rgb"][k].cpu().numpy()
                    gr = gt[g, I_FRGB:I_FRGB + 3]
                    d = float(np.abs(pr - gr).mean())
                    solid_l1.append(d)
                    gray_l1.append(float(np.abs(0.37 - gr).mean()))
                    print(f"  slot{k}<->gt{g} solid  pred={pr.round(2)} gt={gr.round(2)} L1={d:.3f}")
                elif ft == 0:
                    print(f"  slot{k}<->gt{g} none   pred_ftype={f['ftype'][k].argmax().item()} "
                          f"pred_alpha={float(f['fill_alpha'][k]):.2f}")
                else:
                    n_s = int(gt[g, 232])
                    ps = f["stops"][k][:n_s].cpu().numpy()
                    gs = gt[g, 212:212 + 4 * 5].reshape(4, 5)[:n_s]
                    d = np.abs(ps[:, 1:4] - gs[:, 1:4]).mean()
                    print(f"  slot{k}<->gt{g} {name:6s} pred_stops={ps[:, 1:4].round(2).tolist()} "
                          f"gt={gs[:, 1:4].round(2).tolist()} L1={d:.3f}")

    if solid_l1:
        print(f"\n==== SUMMARY ({len(solid_l1)} solid matched pairs, {n_scene} scenes) ====")
        print(f"pred fill rgb L1 = {np.mean(solid_l1):.3f}   (灰均值 0.37 基线 L1 = {np.mean(gray_l1):.3f})")
        print(f"胜过灰基线的对数 = {sum(1 for a, b in zip(solid_l1, gray_l1) if a < b)}/{len(solid_l1)}")


if __name__ == "__main__":
    main()
