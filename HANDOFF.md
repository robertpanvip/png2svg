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
2. **架构层修复**（**P0，由 §8 Benchmark 揭示**）：根因已数据锁定（§5.2）= 无 Hungarian 匹配 + 空间定位归纳偏置缺失 + cls 梯度被 detach 切断 → 表现为 bbox 空间坍缩 + 类别 mode collapse。**修复已实施并验证**：spatial_anchor + Hungarian 匹配 + 调高 cls/geom 权重 + 空间多样性正则；20k 训练（§5.2.3）确认**空间定位坍缩已彻底打破**（中心方差 0.000026→0.048，匹配 GT 数 61→90/119），但暴露**新问题：2×4 硬网格锚点过刚性（网格偏置，oracle_bbox 4k +0.3%→20k +8.3%）**。类别 mode collapse 仍在但已非质量主因（oracle_cls +2.3%）。**下一步：软化空间锚点（缩小尺度 / warmup 后退火 / 改软正则）以消除网格偏置，目标 mae<0.0733**（§5.2.4）。
3. **推理侧剪枝（开箱即用，性价比最高）**：见 §9。`benchmarks/prune.py` + 建议集成到推理入口。
4. 训练/生成分布 = SceneGenerator 分布；真实图片泛化需扩充数据管线
5. AMP 混合精度、多卡（可选提速，§7.3 已实测 sub_px=1+no-ckpt = 2.73 it/s）
6. Rust/WASM 部署侧（导出/推理移植，未开始）

## 5.2 架构层根因与修复（P0，2026-09-08 架构诊断）

> 本节对应 §5 第 2 项。基于 `benchmarks/diag_arch.py`（30 场景 oracle ablation）与
> `benchmarks/diag_query.py`（slot 退化检查）的**数据结论**，非猜测。

### 5.2.1 已确认根因（数据驱动）

| 现象 | 实测数据 | 结论 |
|------|----------|------|
| bbox 中心坍缩 | `bbox 中心方差均值 = 0.000026`（≈0）；pred 中心全部堆在 (0.496, 0.505)±0.008 | **空间定位彻底失效**：所有 slot 能预测不同颜色/形状，但 cx/cy 全坍缩到画布中心 |
| 类别头未训练 | cls 预测分布 `blob:126 / 其他全 0`；cls logits 熵 1.46 ≈ 随机基线 ln5=1.61 | **mode collapse 到 blob**；类别头几乎随机 |
| query 未退化 | `query 余弦相似度(非对角) = -0.029`（近乎正交） | **不是 query 对称坍缩**——可学习 query 是分散的，问题在解码/监督 |
| slot 输出部分差异 | `slot 输出余弦相似度 = 0.336`（max 0.727） | slot 之间并非完全相同，但空间属性高度同质 |

**三个结构性缺陷：**

1. **无 slot↔object 匹配（Hungarian matching 缺失）**：可微渲染损失在图像层面是**置换不变**的（composite 后比像素），而辅助损失 `auxiliary_losses` 按**固定 slot 下标**对齐 GT（encode_scene 按面积降序分配）。两者目标冲突——渲染损失把每个 slot 拉向"质心/平均解"，辅助损失又要求 slot i = 第 i 大对象。模型落入对称局部极小。

2. **空间定位归纳偏置缺失**：解码器仅 2 层 cross-attn + bg 全局池化，没有把 query 绑定到具体空间位置的机制。渲染损失的置换不变性不提供任何定位梯度，辅助几何 L1 又在与渲染损失的拉扯中落败 → cx/cy 全部坍缩到中心。

3. **cls/ftype 头梯度被切断**：`train.py` 中 `cls_ids = aux["cls"][0].detach().argmax(-1)`，渲染路径的类别分支选择被 detach，类别头**唯一**的梯度来源只剩辅助 cls CE（原权重仅 w_cls=0.05）。中心 blob 最像 "blob"，自我强化成 mode collapse。

### 5.2.2 已实施的修复

| 文件 | 改动 | 作用 |
|------|------|------|
| `model/matching.py`（新增） | O(n³) Hungarian 解算器 `match_slots` | 每步把 8 个预测 slot 与 8 个 GT 对象最优一一匹配 |
| `model/network.py` | `spatial_anchor` 参数（NUM_SLOTS×2），按 2×4 网格初始化，加到 bbox 的 cx/cy 原始 logit | 强制每个 slot 偏向画布不同区域，打破定位坍缩 |
| `model/losses.py` | `matched_auxiliary_losses`（匹配后再算 cls/ftype/geom/valid）+ `spatial_diversity` 正则；调高 w_cls 0.05→0.3、w_geom 0.5→0.7、w_ftype→0.3、新增 w_div=0.05 | 消除固定顺序冲突；惩罚活跃 slot 中心过度集中 |
| `train.py` | 暴露 `--w-cls/--w-ftype/--w-valid/--w-svalid/--w-geom/--w-bg/--w-div` | 便于调参 |

### 5.2.3 验证结果（4k + 20k 步 @256px，sub_px=1+no-ckpt）

> 训练命令：
> - 4k：`python -u train.py --steps 4000 --size 256 --sub-px 1 --no-grad-checkpoint --device cuda --out runs/fix4k`
> - 20k：`python -u train.py --resume runs/fix4k/last.pt --steps 20000 --size 256 --sub-px 1 --no-grad-checkpoint --device cuda --out runs/fix20k`（已完成，ckpt `runs/fix20k/last.pt`）
> 对比基线：`runs/gpu/last.pt`（20k，未修复）；诊断脚本 `diag_query.py` / `diag_arch.py`；产物 `runs/diag_*_fix20k.json`。

| 指标 | 原 20k(未修复) | 修复 4k | 修复 20k | 解读 |
|------|---------------|---------|----------|------|
| bbox 中心方差均值 | 0.000026（≈0 全坍缩） | 0.0535 | **0.0478** | ✅ 空间坍缩已结构性打破（≈2000×） |
| query 余弦相似度 | −0.029 | −0.087 | −0.102 | ✅ 仍分散，未退化 |
| cls logits 熵 | 1.46 | 1.03 | **0.995** | ⚠️ 下降但仍 100% blob |
| cls 预测分布 | blob:126 | blob:240 | blob:240 | ❌ mode collapse 未解（非质量主因） |
| 匹配 GT 对象数 | — | 61/119 | **90/119** | ✅ 捕捉到的 GT 对象显著增加 |
| mae（diag baseline） | **0.0733** | 0.0754 | 0.0769 | ➖ 略高于原基线 |
| oracle_bbox 增益 | +11.3% | +0.3% | **+8.3%** | ⚠️ 20k 回升 → 网格偏置 |
| oracle_cls+bbox 增益 | +18.6% | +2.1% | +16.2% | ➖ 上限略低于原模型 |

**结论（数据驱动）：**
- **空间定位坍缩：已彻底修复**（中心方差 0.000026 → 0.048，稳定于 4k 与 20k；匹配 GT 数 61→90/119）。这是 P0 最关键、最难修的一块，已结构性解决。原 20k 模型"mae 0.0733 略优"是**假象**——它是把所有对象堆在画布中心 (0.5,0.5)±0.008 叠加出来的，输出 SVG 不可用；修复模型的对象是真实分散的。
- **新暴露的问题：2×4 硬网格锚点过刚性（grid-bias）**。证据：oracle_bbox 增益在 4k 仅 +0.3%（模型位置已近似正确），但 20k 回升到 +8.3%——说明随着训练推进，锚点把对象"吸"向网格点 (cx∈{0.125,0.375,0.625,0.875})，与 matched 几何监督拉扯，反而损害了连续位置学习。pred 中心均值 cx=0.23（偏左上）即网格偏置的实证。
- **类别 mode collapse 仍在**（100% blob），但 oracle_cls 仅 +2.3%——对像素 MAE 影响极小，已降级为"输出美观度"后续项，非质量 blocker。
- **净评估**：修复把"不可用（全堆叠中心）"变成"可用（真实分散）"，但硬锚点的网格偏置使 mae 比原模型高 ~0.004。下一步应**软化锚点**以释放位置精度。

### 5.2.4 软化空间锚点（已选定并运行中）：warmup 后冻结+退火释放

> 目标：保留"打破坍缩"能力，但允许对象学到任意连续位置，把 oracle_bbox 从 +8.3% 压回 ~0、mae 推到原基线 0.0733 以下。
> **已决策（用户授权"你选吧"）：采用 方案②+③ 组合** —— 冻结 `spatial_anchor` 使其不被训练放回到网格，并按 step 把 `anchor_scale` 从 1.0 退火到 0.0，释放后由 `w_div` 正则 + matched 几何损失维持对象分散、学到真实连续位置。

**代码改动（已提交）：**
- `model/network.py`：`forward(img, anchor_scale=1.0)`，`slots[...,I_BBOX:I_BBOX+2] += spatial_anchor * anchor_scale`（缩放而非硬加）。
- `train.py`：新增 `--anchor-scale`（floor）、`--anchor-anneal-start/end`、`--freeze-anchor`；`anchor_scale_at(step)` 线性调度；日志加 `asc=` 监控。

**运行（2026-09-09，已完成 task hqBwtH）：**
- 命令：`python -u train.py --resume runs/fix20k/last.pt --steps 32000 --size 256 --sub-px 1 --no-grad-checkpoint --device cuda --anchor-scale 0.0 --anchor-anneal-start 20000 --anchor-anneal-end 28000 --freeze-anchor --out runs/fix_anchor`
- 收尾：step 32000，train mae=0.0721，geom=5.18，asc=0.00，无报错。

**验证结果（同 30 场景 seed=777000，与 fix20k 直接可比）：**
| 指标 | fix20k（锚点固定） | fix_anchor（释放到 0） | 判定 |
|------|--------------------|------------------------|------|
| 诊断 baseline mae | **0.0769** | 0.0814（+5.8% 更差） | ❌ 释放后反而变差 |
| oracle_bbox 增益 | +6.4% | +10.6% | ❌ bbox 更偏离 GT |
| 匹配 GT 对象 | 90/119 | 95/119 | ✅ 多匹配 5 个 |
| bbox 中心均值 | (0.23, 0.21) | (0.25, 0.29) | ❌ GT 应 (0.5,0.5)，飘向左上 |
| 中心方差均值 | 0.0478 | 0.0883 | ✅ 更分散（坍缩确未回退） |
| cls 分布 | 100% blob | 100% blob | ➖ 仍 mode collapse |

**结论（数据驱动，已证伪"完全释放锚点"假设）：**
- **完全释放锚点（asc→0）是次优的**：对象更分散、匹配更多，但诊断 mae 反升 5.8%、bbox 偏离 GT 更大。根因：硬网格锚点提供了"大概对"的位置先验，撤掉后网络缺乏位置归纳偏置，把对象飘到**偏左上的错误分布**（cx=0.25/cy=0.29，而 GT 均匀 0.5/0.5）。
- `train mae` 的"改善"（0.0721<0.0746）只是训练分布过拟合假象；固定诊断集上 fix_anchor 更差。
- **当前最佳可用 ckpt = `runs/fix20k/last.pt`**（锚点固定，诊断 mae 0.0769 优于 fix_anchor 0.0814）。
- 修复带来的真实收益已坐实：原 20k 未修复 oracle_bbox +11.3% → fix20k +6.4%（bbox 误差减半），中心方差 0.000026→0.0478（坍缩彻底打破）。

**下一步（已采纳，见 §5.2.5）：** 锚点不应降到 0，应保留**弱先验 floor≈0.3**（`--anchor-scale 0.3` 不退火，或退火到 floor 而非 0）。此外 oracle 显示 bbox 中心仍有 +6.4% 空间，建议**对匹配对象加 cx/cy 专项 L1 监督**（让网络在锚点附近精修到 GT 中心）比继续堆步数更有效。阶段优先级：弱锚点重训 > bbox 中心专项监督 > 类别 mode collapse（cls 影响仅 +1.7%，最低优先）。

### 5.2.5 弱锚点 floor=0.3 + 匹配对象 cx/cy 专项 L1 监督（运行中）

> 目标：在 fix20k 基础上，把锚点从"完全释放（次优）"修正为"弱固定先验（floor=0.3）"，并新增 cx/cy 专项 L1 直接把匹配对象中心拉向 GT，压低 oracle_bbox（+6.4%），目标诊断 mae < 0.0769（fix20k 基线）、并改善 bbox 中心偏移。

**代码改动（本轮新增）：**
- `train.py`：新增 `--w-bbox`（默认 0.5）权重参数；`--anchor-scale` 即 floor（设为 0.3 即弱先验，不退火到 0）；日志加 `bbox=` 项监控中心 L1。
- `model/losses.py`：`matched_auxiliary_losses` 在 Hungarian 匹配对上新增 `parts["bbox"] = F.l1_loss(stack([f_cx,f_cy]), gt_bbox[cx,cy])`（仅匹配预测 slot，无匹配时为 0）；`compute_losses` 新增 `w_bbox` 参数并入 `aux_total`；`auxiliary_losses` 兼容接口同步返回（转发）。
- `model/network.py`：无需改动——`anchor_scale=0.3` 已是弱先验地板机制。

**运行（2026-09-09，task baZpC0）：**
- 命令：`python -u train.py --resume runs/fix20k/last.pt --steps 32000 --size 256 --sub-px 1 --no-grad-checkpoint --device cuda --anchor-scale 0.3 --anchor-anneal-start 20000 --anchor-anneal-end 24000 --freeze-anchor --w-bbox 0.5 --out runs/fix_bbox`
- 退火窗口 20k→24k（4000 步缓降 1.0→0.3），之后 8k 步在 asc=0.3 下自由精修；约 67min。
- 50 步冒烟已验证：`asc` 1.00→0.30 平滑、bbox L1 项正常出现（~0.18-0.25）、无 NaN/爆炸、gn 被 clip 正常。

**验证结果（已完成 2026-09-09，同 30 场景 seed=777000，三版直接可比）：**
| 指标 | fix20k（锚点 1.0） | fix_anchor（释放→0） | **fix_bbox（0.3+L1）** | 判定 |
|------|--------------------|------------------------|------------------------|------|
| 诊断 baseline mae | **0.0769** | 0.0814 | 0.0796 | ❌ fix_bbox 仍差于 fix20k |
| oracle_bbox 增益 | **+6.4%** | +10.6% | +9.2% | ❌ L1 监督反让 bbox 更偏 |
| 匹配 GT 对象 | **90/119** | 95/119 | 84/119 | ❌ 匹配数下降 |
| bbox 中心均值 | (0.23,0.21) | (0.25,0.29) | (0.23,0.25) | ❌ 仍偏左上，未趋近 (0.5,0.5) |
| 中心方差均值 | 0.0478 | 0.0883 | 0.0696 | ✅ 坍缩未回退 |

**结论（本轮证伪"弱锚点+bbox中心L1"方向）：**
- **fix_bbox 是回归**：诊断 mae 0.0796 > fix20k 0.0769，匹配数 84<90，bbox 偏移更大。cx/cy 专项 L1 没帮上忙，反而拉低匹配质量。
- **锚点强度呈单调正相关**（1.0 > 0.3 > 0）：硬网格先验是网络做 slot 消歧的关键，弱化即丢匹配。这是结构张力——**模型靠网格先验消歧，放开就回退，学不到连续位置**。
- **当前最佳可用 ckpt 仍是 `runs/fix20k/last.pt`**（锚点全固定 1.0，诊断 mae 0.0769）。fix_bbox 不提升，不采纳。
- **训练侧 ROI 已耗尽**：bbox 中心精度（oracle_bbox 卡在 +6.4%）是网格先验的伴生代价，靠调参无法根治，需架构改动（如 anchor-free 预测 + 位置嵌入）才可能突破——属大改，不值得在当前管线继续堆步数。

**下一步建议（请在 §5.2.5 结论后定）：**
1. **部署侧优先**（推荐）：`fix20k/last.pt` 已可交付，做 Rust/WASM 推理导出 + INT8 量化 + README，把科研项目变可用产品。
2. **真实图片泛化**：拿真实图标 PNG 跑 `benchmarks/infer.py`，验证泛化（训练分布=SceneGenerator 合成）。
3. 训练侧：暂停，边际收益已负。

### 5.2.6 anchor-free 架构大改：注意力质心空间先验（已完成，结论：**失败**）

> 根因（§5.2.5 结论）：bbox 中心精度（oracle_bbox +6.4%）是硬网格锚点的伴生代价——网格先验同时是 slot 消歧的解药与 grid-bias 的病根，调参绕不开。真正突破 = 去掉硬编码网格，让空间先验从图像本身来。
> 设计（用户选"注意力质心"方案）：删 `spatial_anchor`，改用 cross-attn 注意力权重算出的**质心**作为数据驱动空间先验（模型看哪、中心就在哪），bbox 中心 L1 梯度反传回注意力使其聚焦；加注意力熵正则（`w_cent`）鼓励每个 slot 看向紧凑区域，防质心扩散回中心。

**代码改动（已完成，待提交）：**
- `model/network.py`：
  - 删 `spatial_anchor` 参数与 forward 的网格偏移。
  - `DecoderLayer.forward` 返回最后一层 cross-attn 权重 `(B, num_slots, grid*grid)`。
  - `VectorNet.forward(img, centroid_scale=1.0)`：新增 `_centroid_logit(attn_w, grid, device)` —— 权重 softmax 后按网格坐标算质心 `(cx,cy)∈0..1`，转成 `_lin` 逆变换 logit 加到 bbox cx/cy；同时返回平均注意力熵 `aux["cent_entropy"]`。该质心随图像/对象变化（非固定模板），故无 grid-bias。
- `model/losses.py`：`compute_losses` 新增 `w_cent`（默认 0.05），`aux_total += w_cent * cent_entropy`；`parts` 加 `cent`。
- `train.py`：删 `--anchor-scale/--anchor-anneal-*/--freeze-anchor`，改 `--centroid-scale`（默认 1.0）与 `--w-cent`（默认 0.05）；resume 用 `load_state_dict(strict=False)` 容忍旧 ckpt 的 `spatial_anchor`，优化器状态架构不匹配时自动重置（fresh optimizer）；日志加 `cent=`。
- 参数预算：4,699,425（删 spatial_anchor 后仍达标）。

**运行（2026-09-09，task zwkWj3）：**
- 冒烟（50 步）：resume fix20k 容忍架构变化正常；`cent≈5.544`（≈ln256 最大熵）→ 注意力初始完全弥散，模型此前靠网格定位、cross-attn 未聚焦，去除锚点后质心退化到中心——**正是关键风险点，需验证能否聚焦**。
- 正式验证：`--resume runs/fix20k/last.pt --steps 24000 --size 256 --sub-px 1 --no-grad-checkpoint --device cuda --centroid-scale 1.0 --w-cent 0.15 --w-bbox 0.5 --out runs/fix_af`（新增 4k 步，约 25min）。
- 判定：跑 `diag_query`/`diag_arch`，看 (a) `cent` 熵从 5.54 明显下降（注意力聚焦）；(b) 中心方差 >0.01 且**中心分布不再钉在 (0.23,0.21) 网格点**（grid-bias 消失）；(c) oracle_bbox 从 +6.4% 回落、(d) 诊断 mae < 0.0769。

**结论（2026-09-09，task zwkWj3 完成 + 同口径诊断）：**
- `cent` 全程卡 **5.544（=ln256 最大熵）** → cross-attn 权重在 256-token 特征图上**完全均匀**，质心 = 图像几何中心 (0.5,0.5)，对 8 个 slot **零空间区分力**。`w_cent=0.15` 不起作用：注意力已到最大熵，梯度无法使其聚焦（或损失地形偏好均匀——因为其余分支也坍缩）。
- 诊断 mae 0.0758 < fix20k 0.0808，**但这是 +4k 步的副作用而非架构收益**（fix20k 多训 4k 应同等）；预测中心 `cx=0.193±0.218` 仍远离 GT `(0.499±0.155)`——grid-bias 未消失，反而退化成**中心坍缩**（比网格偏置更糟）。
- match 88/119 略优于 fix20k 72/119，主要来自 `--w-bbox 0.5` 的 cx/cy L1 监督，**非质心贡献**。
- **➜ anchor-free 注意力质心方案失败，不是突破口。** 它把网格偏置换成更糟的中心坍缩，对 mae / 定位 / 类坍缩都无改善。`runs/fix_af/last.pt` 不复用，最佳权重保留 `runs/fix20k/last.pt`。
- **代码已还原（2026-09-09，commit 见下）**：`model/network.py` / `model/losses.py` / `train.py` 从 `a3b6f41`(anchor-free) 还原回其父提交 `6a42395` 的 `spatial_anchor` 网格锚点版本（anchor-free 代码保留在 `a3b6f41` 历史，未丢失）。还原原因：anchor-free 空间先验退化均匀、比网格更糟；且最佳权重 `fix20k/last.pt` 用 `spatial_anchor` 训练，还原后可干净 resume 做下一步类头修复。HANDOFF/诊断脚本改动保留。

**战略重定（关键发现）：**
- 空间先验之争（网格 vs 注意力质心）已证明两者都**不治本**。真正被"空间坍缩"叙事掩盖的 #1 瓶颈是 **类坍缩到 blob**——见 §5.2.7。

### 5.2.7 新瓶颈：类别 mode collapse（blob 垄断）— 下一突破方向

> 诊断铁证（fix20k 与 fix_af **同口径** diag_arch，30 场景 seed=777000）：
> - **预测分布**：240/240 slot 全为 `blob`；**GT 分布**：blob 52 / polygon 26 / ellipse 21 / rect 8 / stroke 12（5 类）。
> - cls logits 熵 `0.99 / 1.61`（未数学全坍，但 argmax 恒为 blob）。
> - `oracle_cls`：fix20k **-0.8%**、fix_af **+2.2%** → 给真类标签几乎无益。
> - `oracle_cls_bbox`：fix20k **+13.9%**、fix_af **+16.5%**（天花板，说明类+位置若都对还能降 14–16% mae，但模型学不到类）。
>
> **根因（推测，待验证）：**
> 1. `dataset/renderer.py` 中 `detach().argmax()` 切断了 cls 头经渲染损失的梯度 → 类头只靠弱辅助 CE，无渲染侧激励区分 blob/ellipse/rect（256px 下外观相似）。
> 2. 类别不平衡：blob 占 GT 44%，模型直接塌到多数类；辅助 CE 权重 `w_cls` 过低。
> 3. 类头容量/特征不足以区分形状（decoder 最后层特征对类不敏感）。
>
> **下一步实验（优先级高于任何空间改动）：**
> 1. 修 `detach().argmax()`：让 cls 接收渲染梯度（或软化 argmax 用 differentiable top-k / soft argmax），使 blob vs ellipse 在渲染上可区分。
> 2. 类别平衡 CE：按逆频率重加权（polygon/ellipse/rect/stroke 全为稀有类，需大幅提权）。
> 3. 提高 `w_cls`（当前偏低）→ 观察 cls 熵与 oracle_cls 是否上升。
> 4. **验证判据**：若 `oracle_cls` 从 +2.2% 升到 >10% 且预测分布出现非 blob 类，说明类头有救、空间+类双修可推进；若仍 100% blob，则类头/特征结构需重构（如类无关几何 + 类特定外观解耦）。

### 5.2.7.1 实验 B1：类别平衡 CE + 提 w_cls（用户授权，运行中 task zXa7bx）
> 代码调研结论（落地前）：`cls` 头**确有**梯度来自 `matched_auxiliary_losses` 的 `F.cross_entropy`（losses.py:151，仅匹配对、权重 w_cls 默认 0.3）；渲染损失对类不变（slots_to_objs 硬 argmax 选类 + blob 曲线逼近任意轮廓），故渲染梯度从不惩罚"该椭圆却预测 blob"。
> 坍缩机制：渲染(1.0)+几何(0.7) 用 blob 即可匹配轮廓，cls CE(0.3) 既弱又遇类别不平衡（blob 占 44%）→ 模型学会永远预测 blob。

**代码改动（已完成，待提交）：**
- `model/losses.py`：新增模块常量 `CLS_BAL_W`（基于 generator.shape_weights 逆频率，blob 归一到 1.0 → `[1.0, 1.75, 2.33, 2.33, 2.33]`）；`matched_auxiliary_losses` / `auxiliary_losses` / `compute_losses` 加 `cls_balance: bool=True` 参数，`cls` CE 接 `weight=CLS_BAL_W.to(device)`。
- `train.py`：加 `--no-cls-balance` 开关（消融对照用）；`compute_losses` 透传 `cls_balance=not args.no_cls_balance`。
- 注：未改渲染路径（保持类硬开关）。先验证"强平衡 CE + 提权"是否足以教类头（我们有 GT 类标签，CE 本可监督类，之前是权重/平衡不够）。若仍坍缩再上 soft class mixing（让渲染对类可微）。

**运行（2026-09-09，task zXa7bx）：**
- 50 步冒烟通过：resume fix20k 干净加载（无 missing/unexpected 键）、CLS_BAL_W 设备对齐正常；params=4,699,441（含 spatial_anchor 8×2=16）。
- 正式验证：`--resume runs/fix20k/last.pt --steps 24000 --size 256 --sub-px 1 --no-grad-checkpoint --device cuda --w-cls 1.5 --out runs/fix_cls`（20k→24k 新增 4k，约 25min）。
- 判定：跑 `diag_arch` 看 (a) 预测类别分布是否出现非 blob 类（polygon/ellipse/rect/stroke）；(b) `oracle_cls` 是否从 +2.2% 升到 >10%；(c) `oracle_cls_bbox` 天花板是否抬高；(d) 诊断 mae 不显著退步（类修复不应牺牲渲染）。
- 若类分布 diversify 且 oracle_cls>10% → 类头有救，继续长训 ~32k 巩固，fix_cls 成新最佳（并顺带验证 §5.2.7 路线正确）；若仍 100% blob → 上 soft class mixing（渲染类可微）。

**结论（2026-09-09，task zXa7bx 完成 + 同口径 diag_arch）：➜ B1 失败，强平衡 CE + 提权未能打破 blob 坍缩。**
| 指标 | fix20k（基线） | fix_cls（B1: w_cls 1.5 + 平衡） | 判定 |
|------|----------------|----------------------------------|------|
| 预测类别分布 | 240/240 blob | **240/240 blob（无改善）** | ❌ 类头仍恒选 blob |
| cls logits 熵 | 0.995 | 1.190（更不决断但 argmax 仍 blob） | ❌ |
| oracle_cls 增益 | +2.3% | **+1.1%（↓ 退步）** | ❌ 给真类反而更无益 |
| oracle_bbox 增益 | +8.3% | +6.0%（↓） | ❌ |
| oracle_cls_bbox 增益 | +16.2% | +11.8%（↓） | ❌ |
| 匹配 GT 对象 | 90/119 | 82/119（↓） | ❌ |

**根因坐实**：渲染损失对类**不变**（硬 `argmax` 选类 + blob 自由多边形可逼近任意轮廓），render+geom 梯度（权重 1.0+0.7）压倒 cls CE（即便 1.5× + 平衡）→ 模型理性地选"通用匹配器 blob"。`oracle_cls` 仅 +1~2% 说明类选择对像素目标**结构性无关**——纯辅助 CE（无论多强）无法赋予类头"区分 blob/ellipse"的渲染激励。**下一步必须让渲染对类可微**：soft class mixing（§5.2.7.2）。

### 5.2.7.2 实验 C1：soft class mixing（渲染对类可微，已实现，验证训练中）

> 核心思想：每个 slot 不再硬选一类，而是渲染**全部 5 种形状变体**（blob/polygon/ellipse/rect/stroke，几何均从同一 slot 张量构造），按 `softmax(cls_logits)` 加权混合成一层再合成。这样"选椭圆 vs 选 blob"会渲染出明显不同的图，render 梯度即可反传回 cls 头，结构上打破坍缩。

**代码改动（已完成，待提交）：**
- `model/spec.py`：
  - 抽出 `_build_obj(slots_raw, f, k, cls, ftype, canvas, pad_px)`（原 `slots_to_objs` 逐 slot 构造逻辑上移，硬/软两路径共用）。
  - 新增 `slots_to_objs_soft(slots_raw, ftype_ids, bg_raw, canvas, pad_px)`：每 slot 构造 5 个候选 dict + 原始 `cls_logits = slots_raw[..., I_CLS:I_CLS+NUM_CLS]`。
  - 新增 `render_soft_objs(ren, groups, bg)`：按 softmax(cls_logits) 混合每 slot 的 5 个层（premultiplied 约定，与 SoftSVGRenderer `_render_objs` 一致），逐对象 alpha-over 合成；**逐候选 `torch.utils.checkpoint.checkpoint` 释放激活**，避免 5× 形状层同时驻留显存（4GB 卡实测从 OOM 降到可跑）。
- `train.py`：新增 `--soft-cls` 开关；`train_step` 在 `--soft-cls` 下走 `slots_to_objs_soft` + `render_soft_objs`（cls_logits 直连网络输出，梯度贯通），否则保持原硬 `slots_to_objs` 路径。
- 注：`slots_to_objs`（硬 argmax）保持不变，evaluate.py / 非 soft 训练均不受影响。

**梯度验证（50 步冒烟，已通过）：**
- `cls_logits` 接收的渲染梯度 **norm=0.17**（min −0.080 / max +0.084，有限、无 NaN）——**此前渲染梯度对 cls 头为 0，现在首次非零**，证明 soft mixing 让渲染对类可微。
- resume fix20k 干净加载，无 missing/unexpected 键；参数 4,699,441（含 spatial_anchor）。

**运行（2026-09-09，验证训练，后台 task K5n7X6 已完成）：**
- 命令：`python -u train.py --resume runs/fix20k/last.pt --steps 24000 --size 256 --sub-px 1 --no-grad-checkpoint --device cuda --w-cls 1.5 --soft-cls --out runs/fix_soft`（20k→24k 新增 4k；~1.0 it/s，~70min）。
- 训练收敛正常：末段 mae ≈ 0.070–0.075，bbox ≈ 0.185，无 NaN。

**结论（2026-09-09）：C1 失败——soft mixing 梯度通了但不足以破坍缩。**
同口径 `diag_arch`（30 场景，**注意：此时 generator 已重写为全 path 场景，旧 targets 编码器把所有 PathGeom 归为 blob，GT 分布退化为 blob:132 / 其他 0，类维度诊断仅在"全 path 数据 + 旧编码"管线内可比**）：

| 指标 | fix20k（基线） | fix_soft（soft mixing +4k） |
|---|---|---|
| 诊断 mae | 0.0641 | 0.0637（持平） |
| 匹配 | 52/132 | 48/132 |
| oracle_cls | +0.0% | **+0.0%** |
| 预测分布 | 240/240 blob | 240/240 blob |
| bbox 预测中心 | (0.162, 0.141) | (0.155, 0.121)（仍中心坍缩） |

判读：梯度通了（冒烟 norm=0.17）但 4k 步 + 渲染混合信号太弱，未能撬动 argmax 分布离开 blob；`oracle_cls` 归零也说明在"全 path"新数据下 5 类编码已无意义。**类坍缩问题被 §11 的表示重定向吸收**：新 taxonomy 里没有 blob 万能类（几何=路径段序列），5 类 CLS 将被段类型 one-hot 取代（任务 #24 targets 重编码）。fix_soft 不再继续长训，`runs/fix20k/last.pt` 仍为最佳可用权重。C1 的 soft-mixing 机制代码保留（`--soft-cls`），未来若做类无关几何+类特定外观解耦可复用。

**附带修复**：generator 重写后 `svg/serializer.py` 不识别 CMD_A 导致诊断崩溃——已补弧段序列化（rx/ry 乘边长、rot/flag 绝对、终点 bbox.map），30 场景含弧 resvg 渲染验证通过。

## 6. 关键文件索引
```
model/targets.py    常量 + encode_scene/decode_scene（Scene↔张量）
model/spec.py       slots_to_objs（硬 argmax）/ slots_to_objs_soft + render_soft_objs（soft class mixing）/ predictions_to_targets / squash_*
model/matching.py   （新增）O(n³) Hungarian slot↔GT 匹配解算器
model/network.py    VectorNet + spatial_anchor 网格空间先验（anchor-free 已还原，见 §5.2.6）
model/losses.py     matched_auxiliary_losses（匹配版，含 cls_balance 类别平衡 CE）+ spatial_diversity
benchmarks/profile_step.py  单步分段计时 + 显存峰值（新增）
benchmarks/diag_gate.py     门控诊断：valid 概率分布 / 裁剪对比（新增）
benchmarks/diag_arch.py     架构诊断：oracle ablation（新增）
benchmarks/diag_query.py    slot 退化检查：中心方差/query 相似度（新增）
train.py            训练入口（--device cuda 即 GPU 训练；暴露 --w-* 权重）
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
model/spec.py       slots_to_objs（硬 argmax）/ slots_to_objs_soft + render_soft_objs（soft class mixing）/ predictions_to_targets / squash_*
model/matching.py   slot↔GT 匈牙利匹配（§5.2.2 修复，新增）
benchmarks/diag_query.py   slot 退化检查：query/输出余弦相似度 + bbox 中心方差（§5.2.1）
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

## 11. 表示扩展路线图：Raster Graphics → Structured Vector Graphics（2026-09-09 战略转向）

> **战略定位**：项目真正目标 = 把**结构化 PNG**（图标 / Logo / UI asset / 插画）转成**可编辑 SVG**，即 *Raster Graphics → Structured Vector Graphics*，而非任意照片光栅化。表示能力应从当前 5 粗类（`rect/ellipse/polygon/blob/stroke`）逐步收敛/扩张到贴近真实 SVG 的语义树：
> ```
> path   ├─ line / quadratic / cubic / arc / compound path
> fill   ├─ solid / linearGradient / radialGradient
> stroke ├─ width / dash / linecap / linejoin
> effects├─ opacity / blur / shadow / glow
> composition ├─ groups / clipping / masking / holes / z-order
> ```
> **这同时是消解 §5.2.7 blob 坍缩的根本手段**——见下。

### 11.1 为什么这能从根上破坍缩

- 当前 `targets.py` 把 `blob`（Catmull-Rom 贝塞尔环）编码成一个**万能近似类**，ellipse/rect/polygon/stroke 的轮廓它都能拟合 → 渲染损失对"选 blob"无惩罚 → 模型理性全选 blob（§5.2.7 已用数据坐实：240/240 全 blob，oracle_cls 仅 +1~2%）。
- 把几何改成**显式路径段序列**（line/quad/cubic/arc + compound）后，**不再存在"万能类"**：椭圆就是椭圆弧、矩形就是 4 段 line+close、插画曲线就是一串 cubic。渲染是精确的，匹配在几何上有意义，坍缩吸引子消失。
- 顺带让输出直接是**可编辑 SVG**（`<path d="M..L..Q..C..A..Z"/>`），对齐产品使命。

### 11.2 现状盘点（backbone 比预期完善）

`svg/scene_graph.py` 已支持：
- `PathGeom` + `Segment`：命令 `M/L/Q/C/Z`（`CMD_Q` 已定义，但 `_CMD_POINTS` 缺 `A`）；✅ 路径段骨架在。
- `Gradient`：`linear` / `radial` + `stops`；✅ 渐变在。
- `Stroke`：`width/color/alpha`；❌ 缺 `dash/linecap/linejoin`。
- `Effects`：`shadow_dx/dy/blur/rgb/alpha`、`glow_radius/rgb/alpha`、`blur_radius`；✅ 数据模型在，但 `spec.py`/`SoftSVGRenderer` **完全没渲染 effects**（被忽略）。
- ❌ 无 `groups / clipping / masking`；`z-order` 靠 slot 顺序隐式表达。

缺口 = ①消灭 blob 万能类、几何改路径段序列；②`generator.py` 扩展产出丰富语义；③`targets.py`/`network.py` 扩展表示；④`SoftSVGRenderer` 补齐 arc/quad/dash/effects/合成；⑤`losses.py`/matching 适配新几何。

### 11.3 分阶段路线图（依用户 taxonomy 依赖顺序）

| 阶段 | 内容 | 关键改动 | 与坍缩/产品关系 |
|------|------|----------|----------------|
| **P1 几何·path** | line/quad/cubic/arc + compound（去 blob） | `scene_graph` 加 `CMD_A`；几何张量改**路径段序列**（每 slot 变长段表：段类型 one-hot + 控制点）；`generator` 产真实曲线/弧/复合路径；`spec`/渲染器扁平化 quad+cubic+arc；`targets` 重编码 | **既是地基又是坍缩根因修复** |
| **P2 fill** | solid / linear / radial（细化 stops） | 已基本具备；扩 `NUM_STOPS`、停点位置回归、多渐变 | 样式保真 |
| **P3 stroke** | width / dash / linecap / linejoin | `Stroke` 加 `dash/linecap/linejoin`；渲染器描边样式（虚线、端点、连接） | 图标/UI 描边语义 |
| **P4 effects** | opacity（已有）/ blur / shadow / glow | 渲染器实现可微 blur（可分卷积）/shadow（偏移+blur+合成）/glow（blur+加色）；`spec` 接 `Effects` | 插画/Logo 质感 |
| **P5 composition** | groups / clipping / masking / holes / z-order | 新增 `Group`/`Clip`/`Mask` 结构 + 合成通道；`generator` 产嵌套/裁切/遮罩场景 | 复杂资产层级 |

> 每阶段是**可独立训练验证**的增量；P1 优先（地基 + 破坍缩），P2/P3 可与 P1 同轮；P4/P5 工程量最大（可微滤波 + 合成通道），放后。

### 11.4 关键架构决策（2026-09-09 已确认）

- **几何表示 = path-segment 序列**（用户拍板，推荐方案）：每 slot 持有变长段表，段类型 one-hot{L,Q,C,A,Z} + 控制点；compound = 多 subpath 由 M/Z 划分。直接映射 SVG `d`，且**结构上消灭 blob 万能类**。
- **起步范围 = 五阶段全上**（用户拍板）：P1 几何 → P2 fill → P3 stroke → P4 effects → P5 composition 全实现；训练用课程式逐步启用（先几何+fill+stroke，再 effects，再 composition）。
- **对象容量**：维持 `NUM_SLOTS=8` + Hungarian 匹配（P5 groups 阶段再评估是否扩/变长）。
- 与 in-flight `fix_soft`（§5.2.7.2，task K5n7X6）关系：soft-cls 验证"渲染对类可微能否破坍缩"；若 P1 去 blob 落地，类坍缩可能自然消解，fix_soft 结论转对照参考，不必继续长训。

### 11.5 下一步（已启动，见 §11.6 + 任务 #20–#26）

1. 锁新 slot 张量契约（§11.6）→ 改 `scene_graph`（`CMD_A`/Stroke 样式/Group/Clip/Mask）+ `generator`（全 taxonomy）+ `targets`/`network`（段序列编码）+ `spec`/渲染器（扁平化 + 样式 + effects + 合成）+ `losses`/matching（段级监督）。
2. 课程式训练：先几何+fill+stroke 收敛，再启用 effects，再 composition；每阶段 4k 验证。
3. 判定：预测出现非 blob 几何、oracle 类/几何增益、mae 不退步。

### 11.6 新 slot 张量契约（数据侧 → 模型侧共同接口，任务 #20）

> 每 slot 不再有 `CLS`(5 类万能近似)，而是**显式路径段序列 + 属性块**。坐标统一在 slot 局部 bbox 归一化帧（uv∈[0,1]，与现 `I_PTS` 约定一致）。`NUM_SLOTS=8`，`N_SEG=16`（变长段表；实测 generator 单对象最大 16 段=外环 cubic + 内孔 4 弧 + M/Z，故取 16 留余量；compound 由 M 命令起新 subpath）。

**维度预算（SLOT_DIM = 280）：**
```
几何基      : valid(1) + bbox(cx,cy,w,h)(4) + nseg(1) + closed(1)        = 7
段表(16×12) : 每段 = type one-hot{M,L,Q,C,A}(5) + 坐标(7, 见下)         = 192
fill 块     : ftype one-hot{none,solid,lin,rad}(4) + rgb(3) + alpha(1)
             + gp0(2) + gp1(2) + radius(1) + stops(4×5) + nstops(1)      = 33
stroke 块   : svalid(1) + width(1) + rgb(3) + alpha(1)
             + dash(4 值 + offset=5) + linecap onehot{butt,round,sq}(3)
             + linejoin onehot{miter,round,bevel}(3)                      = 17
effects 块  : opacity(1) + blur_radius(1)
             + shadow_dx,dy,blur(3) + shadow_rgb(3) + shadow_alpha(1)
             + glow_radius(1) + glow_rgb(3) + glow_alpha(1)              = 14
composition 块: group_id onehot{G=4}(4) + has_clip(1) + clip_mode{path,mask}(2)
             + clip_ref(1) + has_mask(1) + mask_ref(1) + z(1)            = 11
合计 = 7+192+33+17+14+11 = 274  → SLOT_DIM=280（余量 6）
```
**段坐标布局（每段固定 7 个坐标槽，按类型填充，其余置 0）：**
```
M : [sx, sy, 0,0,0,0,0]
L : [ex, ey, 0,0,0,0,0]
Q : [cx, cy, ex, ey, 0,0,0]            (控制 + 终点)
C : [c1x,c1y, c2x,c2y, ex,ey, 0]
A : [rx, ry, rot, largearc, sweep, ex, ey]
```
**I_* 索引（#23 已落定，`model/targets.py` 为唯一权威）：**
```
几何基   I_VALID=0, I_BBOX=1..4(cx,cy,w,h), I_NSEG=5, I_CLOSED=6
段表     I_SEG=7 .. 198   (16 段 × 12 = one-hot{M,L,Q,C,A}(5) + 坐标(7))
fill     I_FTYPE=199(4 one-hot) I_FRGB=203(3) I_FALPHA=206 I_GP0=207(2)
         I_GP1=209(2) I_GRAD=211 I_STOPS=212(4×5) I_NSTOPS=232        → 199..232 共 34
stroke   I_SVALID=233 I_SWIDTH=234 I_SRGB=235(3) I_SALPHA=238
         I_SDASH=239(4 值 + 数量槽@243) I_SCAP=244(3) I_SJOIN=247(3)   → 233..249 共 17
effects  I_OPACITY=250 I_BLUR=251 I_SDX=252 I_SDY=253 I_SBLUR=254
         I_ESRGB=255(3) I_ESALPHA=258 I_GLOWR=259 I_EGRGB=260(3)
         I_EGALPHA=263                                                  → 250..263 共 14
compose  I_GROUP=264(4 one-hot) I_CLIP=268 I_CLIP_REF=269
         I_MASK=270 I_MASK_REF=271 I_MASK_KIND=272(lum=0/alpha=1)       → 264..274 共 11
实际使用 275 维；275..279 备用。SLOT_DIM=280。
```

**#23 实现要点（与上稿契约的偏差，以此为准）：**
- 段类型 one-hot **不含 Z**：Z 由 `I_CLOSED` 标志重建（closed=1 时每个 subpath 末尾/下一个 M 前补 Z），与 generator"subpath 全闭合或全开放"的产出约定一致。
- **对象按场景原始顺序编码，不再按面积排序**——clip.ref/mask.ref 指向原始下标，且 z-order 有语义。
- 坐标统一钳到 [0,1]（Catmull-Rom/quad 控制点可略超界）；`rot` 归一化 /360；A 段 rx/ry 为 uv 局部帧值。
- `I_SDASH` 的第 5 槽（原 offset 预留）复用为 **dash 数量**（generator 不产 offset）。
- 渐变 stops 上限 4（generator 可产 2..6，超出截断——P2 若需扩到 6，SLOT_DIM 相应 +10）。
- **验证**：`benchmarks/check_targets.py` —— 60 场景 encode→decode→re-encode **逐位精确往返**（60/60 exact）、全部通过 `validate()` + resvg 渲染；死维度均为 one-hot 布局结构性死区（seg0 恒 M、seg15 罕达等），非缺陷。

**渲染器契约（`spec.py`→`SoftSVGRenderer`）**：`slots_to_objs` 改为按段表重建 `PathGeom`（typed M/L/Q/C/A + compound），`Stroke` 带 dash/cap/join，`Effects` 接通 blur/shadow/glow（可微），`Group/Clip/Mask` 走合成通道。封闭形状（closed=1 且无显式 Z 段）自动补 Z。
```

### 11.7 模型侧落地记录（2026-09-10，#23–#27 完成）

- **#23 targets 重编码**：`model/targets.py` 全量重写至 §11.6 契约（280 维）。要点：段类型 one-hot 不含 Z（由 `I_CLOSED` 重建）；对象按场景原始顺序编码（保 z-order 与 clip/mask 引用）；坐标钳 [0,1]；rot/360；dash 数量复用 offset 槽。回归工具 `benchmarks/check_targets.py`：60 场景 encode→decode→re-encode 逐位精确 + validate + resvg 渲染全通过。
- **#24 network**：删 `cls_head`/`ftype_head`（并入 slot 向量）；单一 linear 头拆为分块头（geom 7 / seg 192 MLP-512 / fill 34 / stroke 17 / fx 14 / comp 11），按契约索引拼装；`spatial_anchor` 保留（作用于 I_BBOX）。参数 4,699,441 → **5,027,832**（3–8M 预算内）。前向输出 (B,8,280) + bg；aux={}。
- **#25 spec/渲染器**：
  - `spec.py` 全量重写：`squash_slots` 按块可微映射（one-hot 块保留 raw logits 供 CE 与渲染 argmax）；`slots_to_objs(slots_raw, bg_raw, canvas, pad_px)` 段表→可微扁平化多边形（Q/C 贝塞尔采样、A 弧端点→中心参数化）；`targets_to_raw`（GT→raw，one-hot ±4 饱和 logits）；`predictions_to_targets` 硬化。
  - `renderer.py`：`_path_subpaths` 支持 CMD_Q/CMD_A（numpy 扁平化，GT 直渲用）。
  - **关键修复**：弧角度计算 acos→**atan2**——弧起点落在椭圆 θ=0°/180° 时 dot/ln 恰为 ±1，acos 导数发散 → 梯度 NaN（torch/numpy 两侧同步改）。
  - **验证** `test`（GT→raw→可微渲染 vs SoftSVGRenderer 直渲 GT）：20 场景 mean mae **0.005**（max 0.025，曲线采样相位差），梯度全有限。
  - **范围注记**：dash/cap/join、effects(blur/shadow/glow)、clip/mask 暂不在 soft 渲染路径（GT 与预测一致忽略，辅助损失监督；resvg 导出侧已支持）——P3b/P4 待训练管线稳定后补齐。
- **#26 losses**：`_geom_block` 覆盖全契约块：段类型 CE（j<nseg 掩码）、段坐标 masked-L1（按类型占用槽数 `_COORD_N=(2,2,4,6,7)` 掩码）、nseg 软计数 L1、closed/bbox/opacity、fill(solid rgb/alpha + grad gp/radius/stops/nstops)、stroke(width/rgb/alpha/dash/ndash/cap CE/join CE)、effects 全 L1、composition BCE+ref L1。匹配代价 = bbox L1 + 段类型 NLL（向量化，无逐步 Python 循环）。`spatial_diversity` 保留。
- **#27 接线**：`train.py` 删 soft-cls/cls-balance 遗留，接新签名；`evaluate.py` 同步。**新表示与旧 ckpt 头不兼容 → 从零训练**（--resume 严格加载不变）。
- **冒烟**：50 步 GPU（256px, sub_px=1, no-ckpt）total 14.99→10.31、mae 0.69→0.37、**~3 it/s**（与旧表示持平，段扁平化无额外减速）、无 NaN。参数 5.03M。
- **下一步**：from-scratch 长训 ~24k（≈2.2h），验收判据 = 对象匹配率/段类型分布多样性/mae vs 旧表示 0.069；diag 工具需按新契约重写（旧 diag_arch 依赖 I_CLS 已失效）。

### 11.8 部署包骨架（2026-09-10 凌晨，随 #27 后落地）

- `deploy/export_weights.py`：训练 ckpt → `deploy/weights/`（`net_fp32.pt` + `net_int8.pt` + `manifest.json`）。5.03M 参数：fp32 **19.21 MB**（<20MB 硬指标达标）；INT8 动态量化（Linear）16.72 MB（conv 未量化，尺寸收益有限，进一步压缩需 conv QAT）。
- `deploy/predict.py`：独立 CPU 推理（torch+numpy+PIL）——net → predictions_to_targets → decode_scene → prune(可选) → serialize。`--bench` 跑 10 场景延迟基准。
- **2k 步中间权重实测（CPU）**：net median 27.5ms / total median **86ms**、max 108ms（fp32），INT8 持平——**<1s 硬指标大幅达标**。INT8 无速度增益（conv 主导前向），定位为"压尺寸"选项。
- 已知现象：2k 步权重预测对象 `fill=none` 且无 stroke → resvg 渲染不可见 → prune 全剪（`8→0`，空 SVG）。**非部署 bug，是训练早期未学会 ftype**；predict.py 已加空结果告警。最终权重（24k/40k）出包后需复测。
- `benchmarks/infer.py` 已修至新接口（aux 头已并入 slot 向量）。
- 待办：Rust/WASM 移植、conv QAT 全 INT8、最终权重复测延迟与剪枝收益。

### 11.9 新表示 24k 验收：PASS（2026-09-10 01:35，续训 40k 已启动）

`diag_new.py`（30 场景 seed=777000，GPU）对 `runs/newrep/last.pt`：

| 指标 | 实测 | 判据/旧基线 |
|---|---|---|
| mae_mean | **0.0541** | 判据 ≤0.085；旧表示 0.069（提升 22%） |
| 对象匹配率 | **1.000（132/132）** | 判据 ≥0.5；旧总匹配率 0.32 |
| 预测段类型 | C:485 / L:109 / A:60 / M:132 | 判据 ≥3 种非 M；旧 240/240 全 blob |
| CPU total p95 | 39.8ms（under_1s=1.0） | <1s |
| oracle_seg / nseg_gap | +0.0016 / −1.78 | 段数偏少 1.8 段 |

**判定：PASS，三项全过。blob 万能类坍缩在结构上消灭生效**——匹配满分、类型多样化、mae 显著优于旧表示。短板：Q 段预测为 0（被 C 吸收，渲染无损）、L 准确率 0.26、欠分段 1.8 段。40k 续训后台运行中（`runs/newrep_train2.log`，预计 ~02:50），06:40 晨报自动化做 40k 终诊 + 部署复测。验收详情：`runs/newrep/ACCEPTANCE_24K.md`；诊断数据：`runs/diag_new_24k.json`、`runs/eval_newrep24k.json`。

### 11.10 40k 终验 + 外观瓶颈定位（2026-09-10 03:30，晨报）

40k 结果（`diag_new` 50 场景）：mae 0.0531（24k→40k 仅 +2%，饱和）、匹配 212/212、类型 C:804/L:178/A:122/M:212、A/L/C acc .61/.34/.90、nseg_gap −2.15、Q 仍为 0（被 C 吸收）。CPU p95 ~33ms。**同配置不再加步数（收益递减，不盲训）。**

**外观瓶颈（新表示下依然存在，与表示无关，P0）：**
- 误差分解（12 场景）：对象净收益 mean **−0.011**，10/12 场景画对象比纯背景更差；背景预测 (0.16,0.15,0.17) vs GT 黑色——平凡背景都未学会。
- fill 头已激活（alpha=1.0、ftype none/solid 对半）但 rgb 学极慢（亮度 0.375）；预测对象=灰色细条（`runs/newrep/vis_seed0_*.png`）。
- 归因：几何有直接辅助监督（bbox L1+类型 NLL+坐标 L1）故学得快；**fill 颜色/覆盖只有渲染 MAE 间接梯度 → 学不动**。与旧表示"对象让 mae 比不画还高"同一病根。
- 修法（下一步 P0）：匹配对上 fill rgb/alpha 直接 L1 监督；bg 颜色监督检查（w_bg/编码）；覆盖 warmup。

**prune 升级**：`prune_scene_greedy`（贪心迭代，逐层剥冗余副本；O(N²) 渲染 N=8 时 ~0.3-0.7s）替代独立消融成为 deploy 默认。当前权重剪后为空是**正确行为**（对象无不可替代贡献）。

部署包定稿：fp32 19.21MB / INT8 16.72MB / CPU net 19-27ms。晨报：`runs/newrep/MORNING_REPORT.md`。
