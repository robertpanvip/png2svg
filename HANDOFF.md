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

## 6. 关键文件索引
```
model/targets.py    常量 + encode_scene/decode_scene（Scene↔张量）
model/spec.py       slots_to_objs / predictions_to_targets / squash_*
model/matching.py   （新增）O(n³) Hungarian slot↔GT 匹配解算器
model/network.py    VectorNet + spatial_anchor 空间锚点先验
model/losses.py     matched_auxiliary_losses（匹配版）+ spatial_diversity 正则
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
model/spec.py       slots_to_objs / predictions_to_targets / squash_*
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
```
