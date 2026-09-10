"""hrf 线性探针：测高分辨率特征是否携带对象颜色信息。

收集 40 场景的 (hrf[matched_slot], GT fill rgb)，岭回归 R²。
R² 高 → HR 分支已有颜色，fill_head 是瓶颈；
R² 低 → HR 分支没学会"看颜色"，需密集颜色探针监督。
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.network import VectorNet
from model.targets import I_FTYPE, I_FRGB, encode_scene
from benchmarks.matched_fill_check import matched_pairs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VectorNet().to(device)
    ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "runs/newrep_hr/last.pt",
                    map_location=device)
    net.load_state_dict(ck["net"])
    net.eval()

    ren = SoftSVGRenderer(256, device=str(device), sub_px=1)
    gen = SceneGenerator(None, 777000)

    feats, colors = [], []
    with torch.no_grad():
        for i in range(40):
            scene = gen.sample()
            img = ren.render_scene(scene)
            slots_gt, _ = encode_scene(scene)
            slots, aux, bg = net(img.unsqueeze(0).to(device))
            slots = slots[0]
            gt = np.asarray(slots_gt)
            ft = gt[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
            assign = matched_pairs(slots, slots_gt, device)
            hrf = net._hr_feat(img.unsqueeze(0).to(device), None) if False else None
            # 直接重取 hrf：复用 forward 内部——用 hook
            feat = {}
            hook = net._hr_feat
            # 简化：重跑一次内部
            import torch.nn.functional as F
            enc = net.encoder(img.unsqueeze(0).to(device))
            pos = net.pos_emb
            if pos.shape[-2:] != enc.shape[-2:]:
                pos = F.interpolate(pos, size=enc.shape[-2:], mode="bilinear",
                                    align_corners=False)
            tokens = (enc + pos).flatten(2).transpose(1, 2)
            q = net.queries.expand(1, -1, -1)
            for layer in net.layers:
                q = layer(q, tokens)
            h = net.final_norm(q)
            from model.spec import C_MIN, C_MAX
            geom = net.geom_head(h)
            cx = (C_MIN + (C_MAX - C_MIN) * torch.sigmoid(
                geom[..., 1] + net.spatial_anchor[..., 0])).detach()
            cy = (C_MIN + (C_MAX - C_MIN) * torch.sigmoid(
                geom[..., 2] + net.spatial_anchor[..., 1])).detach()
            hrf = net._hr_feat(img.unsqueeze(0).to(device), h, cx, cy)[0]   # [K,64]
            for g in range(len(assign)):
                k = assign[g]
                if k < 0 or ft[g] != 1:
                    continue
                feats.append(hrf[k].cpu().numpy())
                colors.append(gt[g, I_FRGB:I_FRGB + 3])
    X = np.array(feats)
    Y = np.array(colors)
    print(f"n={len(X)}  hrf_dim={X.shape[1]}")
    # 岭回归（闭式）——带 train/test 切分（无切分的 R² 会被 65 维记忆污染）
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    n_tr = int(len(X) * 0.7)
    tr, te = idx[:n_tr], idx[n_tr:]
    X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    lam = 1e-2
    A = X1[tr].T @ X1[tr] + lam * np.eye(X1.shape[1])
    W = np.linalg.solve(A, X1[tr].T @ Y[tr])
    pred = X1 @ W
    for split, ii in [("train", tr), ("TEST ", te)]:
        p, y = pred[ii], Y[ii]
        for c, name in enumerate("RGB"):
            ss_res = ((p[:, c] - y[:, c]) ** 2).sum()
            ss_tot = ((y[:, c] - y[:, c].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            r = np.corrcoef(p[:, c], y[:, c])[0, 1]
            print(f"{split} {name}: R2={r2:.3f}  corr={r:.3f}  L1={np.abs(p[:, c]-y[:, c]).mean():.3f}")


if __name__ == "__main__":
    main()
