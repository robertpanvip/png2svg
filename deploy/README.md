# png2svg 部署包（CPU 推理）

把训练好的 VectorNet（280 维 path-segment 契约，HANDOFF §11.6）转成
纯 CPU 可跑的最小推理骨架：**PNG/合成图 → SVG 字符串**。

依赖：`torch`（CPU 版即可）+ `numpy` + `Pillow`。无需 CUDA、无需 resvg
（prune 内部用 resvg-py，若不可用可 `--no-prune` 跳过）。

## 1. 导出权重

```bash
python deploy/export_weights.py --ckpt runs/newrep/last.pt
```

产出 `deploy/weights/`：

| 文件 | 说明 | 典型大小（5.03M 参数） |
|---|---|---|
| `net_fp32.pt` | 纯 state_dict | ~20 MB |
| `net_int8.pt` | Linear 层动态量化 qint8（conv 保持 fp32） | 更小 |
| `manifest.json` | 来源 ckpt、参数量、大小、量化方式 | - |

## 2. 推理

```bash
# 合成样例 → SVG
python deploy/predict.py --weights deploy/weights/net_int8.pt --int8 --sample --out pred.svg

# 真实 PNG → SVG
python deploy/predict.py --weights deploy/weights/net_fp32.pt --png input.png --out pred.svg

# 10 场景延迟基准（median/max）
python deploy/predict.py --weights deploy/weights/net_int8.pt --int8 --bench
```

管线：`net → predictions_to_targets → decode_scene → prune（可选）→ serialize`。

- `--prune-threshold 0.002`（默认）：per-object ablation 剪掉零贡献对象，
  代价约 +100ms/对象（需 resvg-py）；赶时间用 `--no-prune`。
- 项目硬指标：CPU 推理 < 1s/张。

## 3. Python API

```python
from deploy.predict import load_model, predict, load_png
net = load_model("deploy/weights/net_int8.pt", int8=True, size=256)
img = load_png("input.png", 256)
svg_str, info = predict(net, img, 256)   # info: net_ms / prune_ms / total_ms / prune
```

## 待办（对应 HANDOFF 剩余工作）

- [ ] Rust/WASM 移植（推理侧；训练用 PyTorch 保留）
- [ ] 全 INT8（含 conv 的 QAT）进一步压权重
- [ ] 最终权重（40k 训练完成）出包后复测延迟与剪枝收益
