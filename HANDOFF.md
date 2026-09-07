# PNG→SVG 矢量化模型 — 交接文档（GPU 训练切换指南）

> 用途：在 GPU 环境开新会话/换机器训练时，把本文件作为上下文输入。
> 状态：Phase 1 / Phase 2 全部完成并验证；**剩余工作只有训练时长**（需 GPU）。

---

## 1. 项目目标（规格要点）

- **任务**：把 PNG 图标/插图转成 SVG 矢量文件的小型神经网络。
- **路线（已冻结）**：结构化 Scene Graph 预测——网络输出槽位张量，解码成 Scene→SVG。
  **不是** SVG 字符串生成，**不是**传统描摹（potrace/像素聚类）。
- **主损失**：可微渲染损失 `render(预测SVG) ≈ 输入PNG`（MAE + 0.3·(1−SSIM)）+ 辅助结构损失（分类/几何/门控）。
- **硬指标**：
  - 参数量 3–8M → 当前 **4,699,425** ✅
  - CPU 推理 < 1s → 实测 **median 20.6ms / max 58.7ms** ✅
  - 支持：渐变填充/透明度/描边/遮挡/孔洞/多层/Bézier 曲线 ✅（编码、解码、损失、渲染四层全支持）
- **部署目标**：Rust/WASM（推理侧，未开始，属后续阶段）。
- **数据**：`dataset/generator.py` 在线生成（blob/polygon/ellipse/rect/stroke 五类对象 + 纯色/线性/径向渐变填充 + 背景色），无需外部数据集。

## 2. Phase 1 完成内容（此前已验收）

- **可微光栅化器** `dataset/renderer.py::SoftSVGRenderer`：SVG path → 亚像素采样网格 → 可微 coverage（填充 even-odd 类并集、描边 sigmoid band）→ 逐对象 alpha 合成 → RGBA 张量；支持 `keep_layers` 分层输出。
- **对齐验证**：与 resvg 栅格化结果 SSIM **0.99717**（PASS）——可微渲染器与真实渲染器数值一致，这是"渲染损失可信赖"的基础。
- **工具链**：`svg/serializer.py::serialize(scene)->str`、`svg/render.py::render_scene_resvg(scene,w,h)`（评估用真实渲染）、`benchmarks/metrics.py`（MAE/SSIM/PSNR）。

## 3. Phase 2 完成内容（本次，全部验证通过）

### 组件
| 文件 | 内容 | 验证结果 |
|---|---|---|
| `model/targets.py` | slot 常量与布局（K=10×67 维、I_* 索引、CLS_*、FILL_*_I）、`encode_scene`/`decode_scene` 双向编码、blob 控制点反演、孔洞 scale | 往返 per-object MAE max 0.0046（=量化地板） |
| `model/network.py` | VectorNet：CNN 编码器（stride2×4+ResBlock→256d）+ 2 层 slot transformer 解码器 + slot(67)/cls(5)/ftype(4)/bg(5) 四头；pos_emb 自适应插值→任意输入分辨率 | 4,699,425 参数（断言 3–8M） |
| `model/losses.py` | `render_losses`(MAE+0.3·(1−SSIM)) + `auxiliary_losses`(cls/ftype CE、valid/svalid BCE、几何 masked-L1、bg) → `compute_losses` | 冒烟 3 case：近GT 0.146 / 随机 4.54（31×）/ 空场景守卫 OK，梯度全程有限 |
| `model/spec.py` | `slots_to_objs`（预测张量→renderer objs，**返回 (objs, bg) 元组**）、`predictions_to_targets`、`squash_slots`/`squash_bg`；stop 截断+排序 | 冒烟 A/B/C/D 全绿 |
| `dataset/renderer.py`（改造） | tensor 透传、`_render_objs`/`_object_layer` 拆分、**逐对象梯度检查点**（`grad_checkpoint` 开关，use_reentrant=False） | 4GB OOM 修复；开关 parity **0.0**（逐位一致） |
| `train.py` | 在线生成→可微渲染→网络→解码→可微合成→损失→AdamW+warmup+cosine+clip；JSONL 日志；原子 checkpoint；`--resume` 续训；`--device` 已接入网络与渲染器 | 30 步冒烟 + 35→185 续训 + CPU 回归（step-1 total=4.7505 改动前后一致） |
| `evaluate.py` | 预测→Scene→SVG→**resvg 真实渲染**→MAE/SSIM vs 输入 + GT floor + CPU 耗时（net/vec/total、under_1s_ratio） | 全链路跑通 |

### 训练曲线证据（185 步，128px，CPU ~0.25 it/s，分箱平均）
| steps | total | render | mae | geom | grad_norm |
|---|---|---|---|---|---|
| 1–40 | 3.575 | 0.456 | 0.317 | 5.499 | 18.3 |
| 41–80 | 3.169 | 0.342 | 0.226 | 5.024 | 19.3 |
| 81–120 | 3.091 | 0.417 | 0.276 | 4.606 | 7.7 |
| 121–160 | 2.714 | 0.273 | 0.181 | 4.317 | 7.8 |
| 161–185 | **2.643** | 0.271 | 0.184 | 4.169 | 11.0 |

### 当前基线（185 步模型 @256px，evaluate.py，5 场景）
- mae_mean **0.209**、ssim_mean 0.719、GT floor_mean 0.0036
- **valid 门控全部激活**（35 步时 5 场景中 4 个预测 0 对象 → 门控已开始正确训练；场景 2 预测 3/4 对象）
- **CPU 推理 median 20.6ms / max 58.7ms，under_1s_ratio=1.0**
- 质量未收敛——纯算力问题：沙箱 CPU 0.03–0.09 it/s @256px，完整 20k 步不可行；**管线已验证可无缝续训**

### 产物路径
- `runs/smoke/last.pt` — 185 步 checkpoint（含 net/opt/step/生成器 RNG 状态/config，可跨设备 resume）
- `runs/smoke/log.jsonl` — 1–185 完整训练曲线
- `runs/eval_report_185.json` — 185 步评估报告；`runs/eval_report.json` — 35 步对照

## 4. 切换 GPU 训练 — 操作步骤

### 4.1 环境准备
```bash
# 1) 复制整个 /workspace（无外部数据依赖；runs/smoke/last.pt 可选，用于续训）
# 2) 装依赖（腾讯镜像；Linux 上官方 torch 轮子自带 CUDA）
pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple
# 3) 验证 CUDA
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.2 启动训练
```bash
# 全新训练（推荐；先用 200 步试跑看 it/s，再放全量）
python3 -u train.py --steps 20000 --size 256 --sub-px 2 --device cuda \
  --warmup 500 --log-every 50 --ckpt-every 1000 --out runs/gpu

# 或从 CPU 185 步权重续训（map_location 已处理跨设备；--steps 必须 > resume 步数，
# cosine 尾段按新 --steps 重算）
python3 -u train.py --steps 20000 --size 256 --sub-px 2 --device cuda \
  --resume runs/smoke/last.pt --out runs/gpu

# 提速选项：显存 ≥8GB 可加 --no-grad-checkpoint（数值 parity=0.0 已验证，关掉更快；
# 显存紧张则保持默认开启的检查点模式）
```

### 4.3 评估（保持 CPU 是有意的——部署目标就是 CPU <1s）
```bash
python3 evaluate.py --ckpt runs/gpu/last.pt --num 20 --size 256 --out runs/eval_gpu.json
```

### 4.4 收敛判据与预期
- GPU 预计数十 it/s 量级（CPU 0.25 it/s 是唯一瓶颈），20k 步数小时内完成
- 判据：mae(render) 持续下行至 <0.05；`n_objs ≈ n_gt`（门控准确）；eval mae_mean 显著低于 0.209 基线、趋近 floor_mean 0.0036
- 建议每 1–2k 步跑一次 evaluate.py

### 4.5 已知注意点（如实）
- **CUDA 路径未经真实 GPU 实测**（本沙箱无 GPU）：`--device` 接线为代码层验证 + CPU 回归通过；首步若报设备不匹配错误，贴报错即可修
- 同 seed 下 CUDA 与 CPU 数值有微小差异，训练曲线不会逐位一致（正常）
- 梯度检查点用 `use_reentrant=False`，要求 torch ≥2.0（requirements 已是 ≥2.2）
- `slots_to_objs` 返回**元组** `(objs, bg)`，不要整个传给 `_render_objs`

## 5. 剩余差距（后续阶段）
1. 质量收敛：20k 步 @256px GPU 训练（本文件 §4 即可启动）
2. 训练/生成分布 = SceneGenerator 分布；真实图片泛化需扩充数据管线
3. AMP 混合精度、多卡（可选提速，未实现）
4. Rust/WASM 部署侧（导出/推理移植，未开始）

## 6. 关键文件索引
```
model/targets.py    常量 + encode_scene/decode_scene（Scene↔张量）
model/spec.py       slots_to_objs / predictions_to_targets / squash_*
model/network.py    VectorNet（4.70M）
model/losses.py     compute_losses / render_losses / auxiliary_losses
dataset/renderer.py SoftSVGRenderer（可微；grad_checkpoint 开关；device 参数）
dataset/generator.py SceneGenerator（在线数据）
svg/serializer.py   serialize(scene)->str
svg/render.py       render_scene_resvg（评估用真实渲染）
train.py            训练入口（--device cuda 即 GPU 训练）
evaluate.py         评估入口（CPU 全链路 + 耗时）
HANDOFF.md          本文件
```
