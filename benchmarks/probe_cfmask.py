"""E-v3 诊断：模型内部 cf_mask 在【在线场景】上到底带没带颜色信息。

对 N 个新鲜场景，Hungarian 匹配后分 solid 对象对比：
- L1(cf_mask rgb, gt rgb)   —— 掩码内像素池化颜色（v3 新特征）
- L1(cf_raw rgb, gt rgb)    —— bbox 高斯池化颜色（旧特征）
- L1(pred fill rgb, gt rgb) —— 颜色头最终输出
- L1(灰 0.37, gt rgb)       —— 基线
若 cf_mask 好、pred 差 → 颜色头/优化问题；cf_mask 差 → 在线掩码不够准。
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
from model.network import VectorNet
from model.targets import I_FTYPE, I_FRGB, encode_scene

import importlib.util
spec = importlib.util.spec_from_file_location(
    "mfc", os.path.join(_ROOT, "benchmarks", "matched_fill_check.py"))
mfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfc)


def main():
    n_scene = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "runs/genE/last.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = VectorNet().to(device)
    ck = torch.load(ckpt, map_location=device)
    ms = net.state_dict()
    skip = [k for k, v in ck["net"].items()
            if k in ms and ms[k].shape != v.shape]
    net.load_state_dict({k: v for k, v in ck["net"].items() if k not in skip},
                        strict=False)
    net.eval()

    # 钩子抓内部 cf_mask / cf_raw（forward 里不返回，直接挂 forward hook 代价大；
    # 简化：重新实现一次 _hr_feat 之后的部分不值得——改用 monkeypatch 记录。
    store = {}
    orig = VectorNet.forward

    def patched(self, img, anchor_scale=1.0):
        out = orig(self, img, anchor_scale)
        return out

    # 直接从 forward 源码级复制太脆；改为在 eval 时重新跑一遍 _hr_feat + mask 逻辑
    ren = SoftSVGRenderer(256, device=device, sub_px=1)
    gen = SceneGenerator(None, 777000)

    l1s = {"cf_mask": [], "cf_raw": [], "pred": [], "gray": []}
    with torch.no_grad():
        for si in range(n_scene):
            scene = gen.sample()
            img = ren.render_scene(scene).to(device)
            slots_gt, _ = encode_scene(scene)
            slots, aux, bg = net(img.unsqueeze(0), anchor_scale=1.0)
            slots = slots[0]
            assign = mfc.matched_pairs(slots, slots_gt, device)
            gt = np.asarray(slots_gt)

            # 重算 cf_mask/cf_raw：复用 net 内部模块（与 forward 同逻辑）
            from model.network import _lin, C_MIN, C_MAX, W_MIN, W_MAX
            from model.targets import I_BBOX
            feats = net.encoder(img.unsqueeze(0))
            pos = net.pos_emb
            if pos.shape[-2:] != feats.shape[-2:]:
                import torch.nn.functional as F
                pos = F.interpolate(pos, size=feats.shape[-2:], mode="bilinear",
                                    align_corners=False)
            tokens = (feats + pos).flatten(2).transpose(1, 2)
            q = net.queries.expand(1, -1, -1)
            for layer in net.layers:
                q = layer(q, tokens)
            h = net.final_norm(q)[0]
            geom = net.geom_head(h)
            cx = _lin(geom[:, I_BBOX + 0] + net.spatial_anchor[:, 0],
                      C_MIN, C_MAX)
            cy = _lin(geom[:, I_BBOX + 1] + net.spatial_anchor[:, 1],
                      C_MIN, C_MAX)
            bw = _lin(geom[:, I_BBOX + 2], W_MIN, W_MAX)
            bh = _lin(geom[:, I_BBOX + 3], W_MIN, W_MAX)
            hrf, cf_raw, hr_ctx, rawt = net._hr_feat(
                img.unsqueeze(0), h.unsqueeze(0), cx, cy, bw, bh)
            # 掩码 + 像素池化（同 forward v3 逻辑）
            import torch.nn.functional as F
            gh, gw = hr_ctx.shape[-2:]
            mfeat = net.mask_tok(hr_ctx)
            mft = mfeat.flatten(2).transpose(1, 2)
            from model.network import _sine_pos_2d
            mft = mft + _sine_pos_2d(net.HR_DIM, gh, gw, mft.device, mft.dtype)
            pix = net.mask_pix(mfeat).flatten(2).transpose(1, 2)
            mq = net.mask_qproj(h.unsqueeze(0))
            x = net.mask_n1(mq)
            mq = mq + net.mask_attn(x, mft, mft, need_weights=False)[0]
            mq = mq + net.mask_ffn(net.mask_n2(mq))
            ml = (torch.einsum("bkc,btc->bkt", mq, pix)
                  * (net.HR_DIM ** -0.5) + net.mask_bias.view(1, -1, 1))
            ml = ml.reshape(1, net.num_slots, gh, gw)
            pm = torch.sigmoid(ml)
            blk = img.shape[-1] // gh
            pm_up = F.interpolate(pm, size=img.shape[-2:], mode="bilinear",
                                  align_corners=False)
            prod = F.avg_pool2d(
                (pm_up.unsqueeze(2) * img.unsqueeze(0).unsqueeze(1)).flatten(0, 1),
                blk).reshape(1, net.num_slots, 4, gh, gw)
            mass = F.avg_pool2d(pm_up, blk)
            cf_mask = (torch.einsum("bkchw,bkhw->bkc", prod, mass)
                       / mass.sum((2, 3)).clamp_min(1e-6).unsqueeze(-1))[0]
            cfr = cf_raw[0]

            for g in range(len(assign)):
                k = assign[g]
                if k < 0:
                    continue
                if gt[g, I_FTYPE:I_FTYPE + 4].argmax() != 1:
                    continue
                gt_rgb = torch.tensor(gt[g, I_FRGB:I_FRGB + 3], device=device)
                l1s["cf_mask"].append(float((cf_mask[k, :3] - gt_rgb).abs().mean()))
                l1s["cf_raw"].append(float((cfr[k, :3] - gt_rgb).abs().mean()))
                l1s["pred"].append(float(
                    (slots[k, 203:206] - gt_rgb).abs().mean()))
                l1s["gray"].append(float((torch.full((3,), 0.37, device=device)
                                          - gt_rgb).abs().mean()))
    n = len(l1s["pred"])
    print(f"==== E-v3 诊断（{n} solid 对 / {n_scene} 场景, {ckpt}）====")
    for k in ["cf_mask", "cf_raw", "pred", "gray"]:
        v = l1s[k]
        print(f"{k:9s} L1 = {np.mean(v):.3f}" if v else f"{k:9s} 无样本")
    if l1s["cf_mask"]:
        win = sum(1 for a in l1s["cf_mask"] if a < np.mean(l1s["gray"]))
        print(f"cf_mask 胜灰基线比例 = {win}/{len(l1s['cf_mask'])}")


if __name__ == "__main__":
    main()
