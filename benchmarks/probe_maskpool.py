"""零训练验证：用模型预测几何的覆盖掩码池化输入图颜色，与匹配 GT fill 相关。

corr 高（≈oracle 0.65）→ "predict-then-read 颜色精修" 设计成立。
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.matching import match_slots
from model.network import VectorNet
from model.spec import predictions_to_targets, slots_to_objs, squash_slots
from model.targets import I_BBOX, I_FTYPE, I_FRGB, I_NSEG, I_SEG, I_VALID, N_SEG, SEG_DIM, encode_scene


def matched_pairs(slots_raw, gt, device):
    f = squash_slots(slots_raw)
    gv = torch.from_numpy(np.asarray(gt)[:, I_VALID]).to(device)
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
            if not bool(gv[g]):
                continue
            m = jidx < nseg_all[g]
            lp = logp[:, jidx, gt_type_all[g]]
            cost[:, g] = bbox_cost[:, g] + (-lp[:, m].mean(-1)).cpu().numpy()
    return match_slots(cost)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VectorNet().to(device)
    ck = torch.load("runs/newrep/last.pt", map_location=device)
    model_state = net.state_dict()
    skip = [k for k, v in ck["net"].items()
            if k in model_state and model_state[k].shape != v.shape]
    net.load_state_dict({k: v for k, v in ck["net"].items() if k not in skip},
                        strict=False)
    net.eval()

    ren = SoftSVGRenderer(256, device=str(device), sub_px=1)
    gen = SceneGenerator(None, 777000)

    P, G = [], []
    with torch.no_grad():
        for i in range(30):
            scene = gen.sample()
            img = ren.render_scene(scene)
            slots_gt, _ = encode_scene(scene)
            slots, aux, bg = net(img.unsqueeze(0).to(device))
            slots = slots[0]
            # 预测对象列表（预测几何）。注意 objs 跳过了 invalid/无几何的 slot，
            # 需要重建 obj 位置 → slot 的映射。
            f_all = squash_slots(slots)
            obj_slots = [k for k in range(8)
                         if float(f_all["valid"][k].detach()) >= 0.05]
            objs, bg_s = slots_to_objs(slots, bg[0], 256, 5.0)
            if len(obj_slots) != len(objs):      # geo 为 None 的 slot 被跳过，兜底
                obj_slots = obj_slots[-len(objs):]
            # 逐对象覆盖掩码池化输入图
            gt = np.asarray(slots_gt)
            ft = gt[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
            assign = matched_pairs(slots, slots_gt, device)
            # 渲染每对象层拿覆盖
            for oi, item in enumerate(objs):
                k_slot = obj_slots[oi]
                x0, y0, x1, y1 = item["crop"]
                if x1 <= x0 or y1 <= y0:
                    continue
                layer = ren._object_layer(item)        # [4,h,w] coverage*alpha...
                cov = layer[3]
                m = cov > 0.5
                if int(m.sum()) < 4:
                    continue
                img_crop = img[:3, y0:y1, x0:x1]
                pooled = img_crop[:, m].mean(dim=1).cpu().numpy()
                # 该对象对应哪个 GT？（按 slot 索引匹配）
                g = None
                for gg, kk in enumerate(assign):
                    if kk == k_slot:
                        g = gg
                        break
                if g is None or ft[g] != 1:
                    continue
                P.append(pooled)
                G.append(gt[g, I_FRGB:I_FRGB + 3])
    P = np.array(P)
    G = np.array(G)
    print(f"n={len(P)}")
    for c, name in enumerate("RGB"):
        r = np.corrcoef(P[:, c], G[:, c])[0, 1]
        l1 = np.abs(P[:, c] - G[:, c]).mean()
        print(f"{name}: corr(pred-mask pooled color, GT fill)={r:.3f}  L1={l1:.3f}")


if __name__ == "__main__":
    main()
