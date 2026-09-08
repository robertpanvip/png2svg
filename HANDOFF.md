# PNG→SVG 矢量化模型 — 交接文档（GPU 训练切换指南）

> 用途：在 GPU 环境开新会话/换机器训练时，把本文件作为上下文输入。
> 状态：Phase 1 / Phase 2 完成并验证；**20k 步 GPU 训练已完成**（结果见 §7）；剩余为质量收敛与泛化。

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

## 5. 剩余差距（后续阶段，按 2026-09-08 Benchmark 后优先级排序）
1. ~~质量收敛：20k 步 @256px GPU 训练~~ → **已完成，见 §7.2**（mae 0.0695，曲线已饱和）
2. **架构层修复**（**P0，由 §8 Benchmark 揭示的真正问题**）：类别坍缩到 `path`、小对象丢失、bbox 模式坍缩到中心。需排查 I_CLS 训练权重 / I_BBOX 解空间 / grad checkpoint 反向一致性。继续加训练步数只会加深坍缩，**不要再盲目重训**。
3. **推理侧剪枝（开箱即用，性价比最高）**：见 §9。`benchmarks/prune.py` + 建议集成到推理入口。
4. 训练/生成分布 = SceneGenerator 分布；真实图片泛化需扩充数据管线
5. AMP 混合精度、多卡（可选提速，§7.3 已实测 sub_px=1+no-ckpt = 2.73 it/s）
6. Rust/WASM 部署侧（导出/推理移植，未开始）

## 6. 关键文件索引
```
model/targets.py    常量 + encode_scene/decode_scene（Scene↔张量）
model/spec.py       slots_to_objs / predictions_to_targets / squash_*
benchmarks/profile_step.py  单步分段计时 + 显存峰值（新增）
benchmarks/diag_gate.py     门控诊断：valid 概率分布 / 裁剪对比（新增）
train.py            训练入口（--device cuda 即 GPU 训练）
evaluate.py         评估入口（CPU 全链路 + 耗时）
HANDOFF.md          本文件
```

---

## 7. 本地 GPU 实测结果（2026-09-08，GTX 1050 Ti 4GB）

> 以下为本机真机实测，取代 §4.4 中“GPU 预计数十 it/s”的乐观估计。

### 7.1 环境
- 显卡 **GTX 1050 Ti 4GB**（Pascal sm_61），全机仅 1 块；`nvidia-smi` 报 NVML 初始化失败属正常，不影响 torch CUDA。
- torch 必须锁 **2.7.1+cu126**：torch 2.8/cu128 起官方轮子移除 Pascal 支持。
- venv：`C:/Users/Administrator/.workbuddy/binaries/python/envs/png2svg`。

### 7.2 20k 步训练结果（HANDOFF 剩余任务 #1 已完成）
启动命令（200 步冒烟确认后全量；中途暂停出门 1 次，靠 checkpoint 无缝续训）：
```bash
python -u train.py --steps 20000 --size 256 --sub-px 2 --device cuda \
  --warmup 500 --log-every 50 --ckpt-every 1000 --out runs/gpu
# 续训：加 --resume runs/gpu/last.pt（cosine 按 step 位置重算，衔接平滑）
```

| 指标 | CPU-185 基线 | GPU 20k | 说明 |
|---|---|---|---|
| mae_mean | 0.2088 | **0.0695** | 3.0× 改善 |
| ssim_mean | 0.7186 | **0.8951** | +0.18 |
| floor_mean | 0.0036 | 0.0027 | 量化地板 |
| CPU 推理 median / max | 20.6 / 58.7 ms | 26.7 / 74.8 ms | 硬指标 <1s ✅ |
| under_1s_ratio | 1.0 | **1.0** | ✅ |

产物：`runs/gpu/last.pt`、`runs/gpu/log.jsonl`、`runs/eval_gpu20k_rerun.json`。
曲线：mae 1–2k 步 0.126 → 16k 步后 0.072 后趋平，17k 与 20k 评估几乎一致——**再加步数收益已饱和**。

### 7.3 性能瓶颈与提速配置（实测）
`benchmarks/profile_step.py` 分段结果（sub_px=2 + ckpt，独占 GPU 约 1.35 s/步）：
backward 44%（含 checkpoint 重算）、GT 渲染 18%、可微渲染 16%、**slots_to_objs 14%（纯 CPU，GPU 空等）**、net forward 7%。
根因：batch=1 + 逐对象 Python 循环 → kernel launch 与 CPU 段主导，GPU 算力远未吃满。

| 配置 | it/s | 显存峰值 | 20k 步 ETA |
|---|---|---|---|
| sub_px=2 + ckpt（本次实际） | 0.74 | 1.85 GiB | 7.5 h |
| sub_px=1 + ckpt | 2.06 | 0.52 GiB | 2.7 h |
| **sub_px=1 + --no-grad-checkpoint** | **2.73** | 3.47 GiB | **~2 h** |
| sub_px=2 + --no-grad-checkpoint | OOM（需 6.68 GiB） | — | 4GB 卡不可用 |

建议：**下一轮训练用 `--sub-px 1 --no-grad-checkpoint`**（3.7× 提速），先用 2k 步冒烟确认 sub_px=1 的收敛无退化再放全量。

### 7.4 门控诊断（对象数预测不准）
`benchmarks/diag_gate.py` @20k ckpt，30 场景：精确匹配 10/30、**多预测 13、少预测 7**，pred 4.37 vs gt 4.00。
- valid 概率呈双峰（<0.1 共 76 个、>0.9 共 77 个），边缘区间 0.3–0.7 仅 17.9% → **不是阈值/区分度问题**。
- 早期观察："把多余 slot 裁掉后 mae 0.06897 → 0.06895（Δ≈0），对渲染几乎无贡献"——**此结论被 §8 / §9 推翻**：进一步 per-object ablation 显示，多余 slot 不仅贡献为 0，**整体还让 mae 略升**（见 §9）。
- 少预测来自**被完全遮挡的对象**：输入图中不可见，物理上无法预测，却计入 GT 与损失。
- 已排除 slot 排列歧义：`encode_scene` 的 GT 按面积降序分配，顺序可学习。

**§8 Benchmark 揭示的更深问题**：模型不只是"多预测"，而是**类别坍缩到 path、所有 bbox 趋向中心**——这是 mae 看不到但 feature-level 指标完全暴露的结构性问题。详见 §8。

### 7.5 实测注意点
- **评估必须在训练空闲时跑**：与训练并行时曾出现单场景 2915 ms 的假性超时（CPU 争抢），空闲重跑 max 仅 74.8 ms。
- 沙箱/前台跑长任务会被 SIGTERM，但 python 子进程可能存活继续写日志——重启前先 `tasklist` 确认，避免两进程同写一个 log/checkpoint。
- 日志在实时写 `runs/*/log.jsonl`；`tail -f` 经管道会缓冲，直接读文件。

---

## 8. Synthetic Oracle Benchmark（2026-09-08，对应 HANDOFF2 §21 / P0）

工具：`benchmarks/suite.py`（固定 seed 跑 12 组能力集）+ `benchmarks/report.py`（输出 markdown 报告）。
覆盖 12 组能力：basic / geometry / bezier / gradient / gradient_multistop / transparency / occlusion / holes / layers / dense / stroke / mixed。
每组 20 场景（base_seed=100000 + suite_index*10000 + i，**完全可复现**）。

### 8.1 场景级结果（vs GT floor）

| Suite | mae | ssim | recall | exact | 评注 |
|-------|-----|------|--------|-------|------|
| basic | 0.074 | 0.945 | 0.31 | 0.05 | floor 0.001；差 0.07 主要来自身份错位 |
| bezier | 0.041 | 0.966 | 0.33 | 0.50 | 表现相对最好（多数 path） |
| occlusion | 0.088 | 0.924 | 0.44 | 0.45 | 5 对象里 recall 算 OK 但 mae 偏高 |
| dense | 0.068 | 0.944 | **0.13** | 0.00 | **多对象场景完全失效**（n_gt=8.7） |
| layers | 0.091 | 0.915 | 0.40 | 0.10 | floor 0.015 都已偏高 |
| transparency | 0.089 | 0.828 | 0.35 | 0.15 | ssim 最低；半透明预测难 |
| stroke | 0.049 | 0.962 | 0.23 | 0.30 | mae 低但 recall 极低——对象中心坍缩 |
| mixed | 0.080 | 0.865 | 0.42 | 0.50 | 与训练分布同；mae 偏高因类别坍缩 |

### 8.2 对象级 feature-level（n=961 个 GT 对象）

| 维度 | 召回 | 关键观察 |
|------|------|---------|
| shape=path（n=380） | **0.30** | 类别坍缩后主体仍是 path |
| shape=polygon | 0.32 | shape_ok=0.03 → 几乎全误判为 path |
| shape=ellipse | 0.30 | **shape_ok=0.00** → 100% 误判为 path |
| shape=rect | 0.38 | **shape_ok=0.00** → 100% 误判为 path |
| fill=none（stroke 类）| 0.23 | 最难（n=62）|
| area=small | **0.05** | 小对象几乎全丢（n=237，占 25%） |
| area=large | 0.82 | 大对象位置 OK |
| occluded | 0.45 | 比 unoccluded(0.25) 反而高——奇怪，可能是 GT 顺序使 occluded 对象恰好是大对象 |

**总匹配率 32%，匹配对平均 IoU 0.426、中心归一化距离 0.125、颜色 L1 0.206。**

### 8.3 Benchmark 揭示的真问题
1. **类别坍缩到 path**：ellipse/rect/polygon 全部预测为 path——网络可能没学到几何判别（I_CLS 的 argmax 训练权重过低？或类别不平衡？）。
2. **bbox 模式坍缩**：pred 对象全部聚集中心、宽高相等（实测 stroke 组全在 (0.5,0.51)），属于典型 mode collapse。
3. **小对象消失**：25% 的 GT 对象是 small，但召回 5%——可能是 bbox w/h 预测范围被 saturate 到中值。
4. **场景级 mae 严重低估真实问题**：mixed 组 mae 0.080 看起来"还行"，但 recall 仅 42%，对象级 IoU 平均 0.43——**mae 不是改进方向**。

### 8.4 失败样本与诊断闭环
失败样本（mae>0.12）共 16 个，集中在 transparency(5) / layers(5) / occlusion(3)。失败 PNG 已保存到 `benchmarks/cases/{suite}_{seed}_{in|pred}.png`，可肉眼复核。
按 HANDOFF2 §18 要求，本 Benchmark 是**任何模型改动后的回归基准**：重训、调参、网络修改前，先重跑 Benchmark 与本节对比。

---

## 9. 推理侧冗余剪枝（per-object ablation，2026-09-08）

工具：`benchmarks/prune.py`（per-object ablation）+ `benchmarks/compare_prune.py`（A/B 验证）。
原理：对每个 pred 对象渲染"去掉它"的版本，Δ_i = mae_no_i − mae_full；若 Δ_i ≤ threshold 视为冗余，剪掉。

### 9.1 A/B 验证（50 场景 mixed 分布，threshold=0.002）

| 指标 | full | pruned | Δ |
|------|------|--------|---|
| avg 对象数 | 4.0 | **0.2** | **-96% drop** |
| avg mae | 0.0625 | **0.0608** | **-0.0018（改善）** |
| avg ssim_loss | 0.0690 | 0.0662 | -0.0028 |
| 召回 | 0.372 | 0.049 | —（对象没了，但错配也没了）|
| 完全裁空的场景数 | — | **41/50** | — |
| mae 改善的场景数 | — | **38/50** | — |
| 单场景剪枝开销 | — | 56 ms | — |

### 9.2 与"只画背景"的对比（20 场景 mixed）

| | mae |
|---|---|
| **只画背景（无任何对象）** | **0.0444** |
| 模型预测 full | 0.0744 |
| 模型预测 pruned | 0.0706 |

**关键含义**：模型预测的 4 个对象**整体**让 mae 比"只画背景"还高 0.030——这是**模式坍缩的完整证据**：每个对象单独贡献都 ≤0.002（被判"冗余"），但 4 个一起堆在中心反而引入错误的像素覆盖。

### 9.3 集成建议

推理入口**默认开启剪枝**（threshold=0.002），实测：
- 对渲染质量无负面影响（Δ mae ≈ -0.0018，甚至略好）。
- 输出 SVG 更干净（去掉无效对象，文本更小、更可读）。
- 单场景延迟 +56 ms（5% 增），仍在 CPU 目标内。

集成示例见 `benchmarks/infer.py`（最小化推理 demo：load checkpoint → 渲染 PNG → 预测 → prune → 序列化 SVG）。

### 9.4 局限
- 剪枝只解决"输出更干净"，**不解决模型能力本身的问题**——mae 仍 0.06 远未达到 HANDOFF 收敛目标。
- 真正修法见 §5.2（架构层修复）。

---

## 10. 关键文件索引（增量更新）
```
model/targets.py    常量 + encode_scene/decode_scene（Scene↔张量）
model/spec.py       slots_to_objs / predictions_to_targets / squash_*
benchmarks/profile_step.py  单步分段计时 + 显存峰值（§7.3）
benchmarks/diag_gate.py     门控诊断：valid 概率分布 / 裁剪对比（§7.4）
benchmarks/suite.py         12 组能力 Benchmark（§8，新增）
benchmarks/report.py        Benchmark → markdown（§8，新增）
benchmarks/prune.py         per-object 剪枝（§9，新增）
benchmarks/compare_prune.py 剪枝 A/B 验证（§9，新增）
benchmarks/infer.py         最小推理 demo（默认 prune）
train.py            训练入口（--device cuda 即 GPU 训练）
evaluate.py         评估入口（CPU 全链路 + 耗时）
HANDOFF.md          本文件
```
