"""A 方案（generator 减压）40k 一键验收。

用法: python benchmarks/eval_genA.py
产出: runs/genA40k/ACCEPTANCE.md + vis 对比图
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dataset.generator import SceneGenerator
from dataset.renderer import SoftSVGRenderer
from model.losses import render_losses
from model.network import VectorNet
from model.spec import slots_to_objs
from model.targets import encode_scene, I_FTYPE, I_FRGB

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/genA40k/last.pt"
OUT_DIR = os.path.dirname(CKPT)
SIZE = 256


def load_net(ckpt: str) -> VectorNet:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = VectorNet().to(device)
    ck = torch.load(ckpt, map_location=device)
    ms = net.state_dict()
    skip = [k for k, v in ck["net"].items() if k in ms and ms[k].shape != v.shape]
    if skip:
        print(f"[load] skip shape-mismatch: {skip}")
    net.load_state_dict({k: v for k, v in ck["net"].items() if k not in skip}, strict=False)
    net.eval()
    return net


def eval_render(net, ren, gen, n: int = 50) -> dict:
    """渲染 mae / ssim（同旧 40k 验收口径）。"""
    maes, ssims = [], []
    with torch.no_grad():
        for _ in range(n):
            scene = gen.sample()
            img = ren.render_scene(scene)
            slots, aux, bg = net(img.unsqueeze(0).to(next(net.parameters()).device))
            objs, bg_s = slots_to_objs(slots[0], bg[0], SIZE, 5.0)
            pred = ren._render_objs(objs, bg_s)
            r = render_losses(pred, img.to(pred.device), 0.3)
            maes.append(float(r["mae"]))
            ssims.append(float(r["ssim"]))
    return {"mae_mean": float(np.mean(maes)), "ssim_mean": float(np.mean(ssims)), "n": n}


def eval_fill_colors(net, ren, gen, n: int = 40) -> dict:
    """匹配对颜色：pred 是否脱离灰均值。返回 |pred_rgb - 灰| 与 |gt - 灰| 的均值差。"""
    GRAY = np.array([0.37, 0.37, 0.37])
    devs, gt_devs = [], []
    device = next(net.parameters()).device
    with torch.no_grad():
        for _ in range(n):
            scene = gen.sample()
            img = ren.render_scene(scene)
            slots_gt, _ = encode_scene(scene)
            gt = np.asarray(slots_gt)
            ft = gt[:, I_FTYPE:I_FTYPE + 4].argmax(-1)
            slots, aux, bg = net(img.unsqueeze(0).to(device))
            objs, _ = slots_to_objs(slots[0], bg[0], SIZE, 5.0)
            for item in objs:
                fill = item.get("fill")
                if fill is None or fill[0] != "solid":
                    continue
                rgb = fill[1].detach().cpu().numpy()
                gate = float(fill[2])
                if gate < 0.3:   # 模型实际没画的槽不计
                    continue
                devs.append(np.abs(np.asarray(rgb[:3]) - GRAY).mean())
            for k in range(min(len(scene.objects), 8)):
                if ft[k] == 1:
                    gt_devs.append(np.abs(gt[k, I_FRGB:I_FRGB + 3] - GRAY).mean())
    return {
        "pred_color_dev_from_gray": float(np.mean(devs)) if devs else -1.0,
        "gt_color_dev_from_gray": float(np.mean(gt_devs)) if gt_devs else -1.0,
        "n_solid_objs": len(devs),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_net(CKPT)
    ren = SoftSVGRenderer(SIZE, device=device, sub_px=1)
    gen = SceneGenerator(None, 777000)   # 评测种子固定（新分布）

    rep = eval_render(net, ren, gen, 50)
    print("[render]", rep)
    rep2 = eval_fill_colors(net, ren, gen, 40)
    print("[fill]", rep2)

    # 训练日志尾部
    log_path = os.path.join(OUT_DIR, "log.jsonl")
    tail = []
    if os.path.isfile(log_path):
        rows = [json.loads(l) for l in open(log_path)]
        for r in rows[:: max(1, len(rows) // 10)] + rows[-1:]:
            tail.append((r["step"], r["appearance"], r["mae"]))

    lines = ["# A 方案 40k 验收（generator 减压分布）", "",
             f"ckpt: `{CKPT}`", "",
             f"## 渲染（50 场景 @256px）", "",
             f"- mae_mean: **{rep['mae_mean']:.4f}**  (旧分布 40k: 0.0531)",
             f"- ssim_mean: **{rep['ssim_mean']:.4f}**  (旧: 0.895)", "",
             "## 颜色是否脱离灰均值", "",
             f"- pred solid 色 |RGB-0.37|: **{rep2['pred_color_dev_from_gray']:.4f}**",
             f"- GT   solid 色 |RGB-0.37|: {rep2['gt_color_dev_from_gray']:.4f}",
             f"- 比值 pred/GT = {rep2['pred_color_dev_from_gray'] / max(rep2['gt_color_dev_from_gray'], 1e-6):.2f}"
             "（1.0=完美用色，~0=仍全灰）", "",
             "## 训练日志趋势", ""]
    for step, app, mae in tail:
        lines.append(f"- step {step}: app={app:.4f} mae={mae:.4f}")

    out_md = os.path.join(OUT_DIR, "ACCEPTANCE.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] -> {out_md}")


if __name__ == "__main__":
    main()
