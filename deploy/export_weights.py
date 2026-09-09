"""权重导出：训练 ckpt → 部署权重（fp32 / INT8 动态量化）。

用法：
  python deploy/export_weights.py --ckpt runs/newrep/last.pt
  # → deploy/weights/net_fp32.pt  deploy/weights/net_int8.pt  deploy/weights/manifest.json

对应 HANDOFF §10（完整部署包：抽权重 <20MB、INT8 量化）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.ao.quantization import quantize_dynamic

from model.network import VectorNet


def state_dict_from_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["net"] if isinstance(ck, dict) and "net" in ck else ck


def size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练 ckpt（含 net 键或裸 state_dict）")
    ap.add_argument("--out-dir", default=os.path.join("deploy", "weights"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sd = state_dict_from_ckpt(args.ckpt)

    net = VectorNet()
    net.load_state_dict(sd)  # 严格加载，验证 ckpt 与当前架构一致
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())

    # fp32
    fp32_path = os.path.join(args.out_dir, "net_fp32.pt")
    torch.save(net.state_dict(), fp32_path)

    # INT8 动态量化（Linear 层；conv 保持 fp32）
    qnet = quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
    qnet.eval()
    int8_path = os.path.join(args.out_dir, "net_int8.pt")
    torch.save(qnet.state_dict(), int8_path)

    manifest = {
        "source_ckpt": os.path.abspath(args.ckpt),
        "params": n_params,
        "fp32_mb": round(size_mb(fp32_path), 2),
        "int8_mb": round(size_mb(int8_path), 2),
        "quantization": "dynamic qint8 on torch.nn.Linear (conv fp32)",
        "arch": "VectorNet 280-dim path-segment contract (HANDOFF §11.6)",
        "note": "int8 权重需用 quantize_dynamic 重建模型后 strict load（见 deploy/predict.py）",
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[export] params={n_params:,}")
    print(f"  fp32: {fp32_path}  ({manifest['fp32_mb']} MB)")
    print(f"  int8: {int8_path}  ({manifest['int8_mb']} MB)")


if __name__ == "__main__":
    main()
