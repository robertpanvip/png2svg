# PNG→SVG 项目交接文档 2：Synthetic Oracle 与可验证训练闭环

> 本文不是替代 `HANDOFF.md` 的 GPU 启动说明，而是项目下一阶段的研发规范。
>
> 核心原则：**现实世界的 PNG→SVG 没有唯一标准答案；但我们可以通过自己生成 SVG→PNG，构造具有精确 Ground Truth 的 Synthetic Oracle，用它验证模型、定位错误、自动生成 Hard Cases，并形成 Generate → Train → Evaluate → Diagnose → Retrain 的闭环。**

---

## 1. 项目最终目标

项目目标是构建一个小型神经网络，将 PNG 图像转换成视觉高度一致、结构合理的 SVG。

硬约束：

- PNG→SVG 的核心必须是神经网络，而不是 Potrace、OpenCV tracing、颜色聚类等传统矢量化算法。
- 模型输出结构化 Scene Graph，不直接生成 SVG XML 字符串。
- 模型目标参数量约 3M–8M。
- 最终部署模型目标 `<20MB`，优先支持 INT8。
- CPU 推理目标 `<1s`。
- 后续目标：Rust / WASM。
- 最终评价重点是**视觉还原质量 + SVG 合理性**，不是恢复某一个唯一 SVG 字符串。

当前 baseline 已经具备：

- VectorNet：约 4.7M 参数。
- Scene Graph / slot prediction。
- Differentiable SVG Renderer。
- SVG serializer 与真实 resvg 评估链路。
- gradient / opacity / stroke / occlusion / holes / layers / Bézier 等基础表达能力。

详细历史状态见 `HANDOFF.md`。

---

## 2. 最重要的任务定义：现实世界没有唯一答案

对于真实 PNG：

```text
PNG → SVG
```

通常不存在唯一正确 SVG。

例如一个圆既可以表示为：

```svg
<circle ... />
```

也可以表示为等价的 Bézier path。

因此不能把：

```text
predicted SVG == 某个 GT SVG 字符串
```

作为最终目标。

真正的最终目标是：

```text
render(predicted_svg) ≈ input_png
```

同时希望 SVG 保持合理的结构、较低的复杂度和可编辑性。

---

## 3. Synthetic Oracle：我们拥有精确标准答案

虽然真实 PNG 没有唯一标准答案，但项目可以自己生成具有精确 Ground Truth 的训练与验证样本：

```text
GT SVG Scene
     ↓
   Renderer
     ↓
  Input PNG
     ↓
 Neural Network
     ↓
Predicted Scene
     ↓
   Renderer
     ↓
Predicted PNG
```

因为最初的 Scene 是程序生成的，所以我们知道完整 Ground Truth：

- object count
- object type
- geometry
- position
- size
- Bézier control points
- fill
- gradient
- opacity
- stroke
- effects
- z-order
- visibility
- background

这意味着 Synthetic Dataset 不依赖人工标注，标签应视为**机器生成的精确 Oracle**。

注意：Oracle 是“该 synthetic sample 的已知生成答案”，不是现实世界 PNG 的唯一答案。

---

## 4. Synthetic Dataset 必须可复现

每个样本必须能够通过 seed 和 generator/version 重新生成。

建议逻辑数据结构：

```text
sample/
├── input.png
├── ground_truth.svg
├── ground_truth_scene.json
└── metadata.json
```

`metadata.json` 至少记录：

```json
{
  "seed": 12345,
  "generator_version": "...",
  "renderer_version": "...",
  "canvas_size": 256,
  "object_count": 5,
  "features": ["gradient", "opacity", "bezier"]
}
```

不要求训练时永久保存所有 PNG；在线生成仍然可以保留。但 Benchmark / Hard Case 必须能固化和复现。

---

## 5. 双监督训练：Scene Loss + Render Loss

Synthetic 数据有精确 Scene Ground Truth，因此训练应使用双监督。

### 5.1 Scene Supervision

```text
Predicted Scene
      ↓
      VS
Ground Truth Scene
```

监督内容包括：

```text
L_type
L_geometry
L_position
L_size
L_fill
L_gradient
L_opacity
L_stroke
L_effect
L_zorder
L_visibility
```

具体 loss 应根据当前 Scene Spec 实际字段实现，不允许为了方便删除难以监督的能力。

### 5.2 Render Supervision

```text
Predicted Scene
      ↓
Differentiable Renderer
      ↓
Predicted PNG
```

然后：

```text
Predicted PNG
      VS
Input PNG
```

建议逐步包含：

```text
L_pixel
L_SSIM
L_perceptual
L_edge
L_color
L_alpha
```

### 5.3 总 Loss

概念上：

```text
L =
    λ_scene  * L_scene
  + λ_render * L_render
  + λ_complexity * L_complexity
```

权重必须通过实验确定，不允许拍脑袋固定后长期不验证。

---

## 6. 为什么不能只用 Scene Loss

Ground Truth Scene 是一种已知正确表达，但不是唯一表达。

例如：

```text
GT: circle
Prediction: equivalent Bézier path
```

可能出现：

```text
Scene Loss > 0
Render Loss ≈ 0
```

不能因此简单判定预测失败。

因此：

> **Render Loss 是最终视觉目标，Scene Loss 是强监督和结构学习工具。**

---

## 7. 为什么也不能只用 Render Loss

如果只优化：

```text
PNG → SVG → PNG
```

模型可能学出大量碎片 Path、冗余对象或异常复杂的 Scene，只为了拟合像素。

因此需要 Complexity Regularization，至少关注：

- object count
- path count
- control point count
- SVG serialized size
- 无效/冗余对象

目标不是单纯压低 SVG 复杂度，而是在视觉质量相近时偏向合理、紧凑的 Scene。

---

## 8. Evaluation 必须成为第一等公民

以后任何修改以下内容的 PR/commit：

- network
- decoder
- targets/spec
- loss
- renderer
- generator
- serializer

都必须通过固定 Benchmark 对比 before/after。

最少输出：

```text
Rendering:
  MAE
  SSIM
  PSNR

Scene:
  object count accuracy
  object type accuracy
  geometry error
  color error
  gradient error
  opacity error
  z-order accuracy

Complexity:
  predicted object count
  path count
  SVG size

Performance:
  CPU latency
  memory
```

如果只有整体 SSIM，没有 feature-level diagnosis，不允许声称“模型变强”。

---

## 9. Benchmark 必须按能力分组

建议建立固定 benchmark：

```text
benchmarks/
├── basic/
├── geometry/
├── bezier/
├── gradient/
├── transparency/
├── occlusion/
├── holes/
├── layers/
├── shadow/
├── blur/
├── glow/
└── mixed/
```

例如 gradient 还应细分：

```text
gradient/
├── linear/
├── radial/
├── multi-stop/
└── transparent-gradient/
```

Benchmark 必须固定 seed，避免每次评估因为随机数据变化而无法判断 regression。

---

## 10. Feature-level Diagnosis

评估结果不能只有：

```text
SSIM = 0.91
```

应该能回答：

```text
Feature                 Accuracy / Error
-----------------------------------------
Circle                  ...
Rectangle               ...
Bézier                  ...
Linear Gradient         ...
Radial Gradient         ...
Opacity                 ...
Z-order                 ...
Shadow                  ...
Blur                    ...
Glow                    ...
```

目标是定位：

> **模型究竟哪里不会，而不是模型整体好不好。**

---

## 11. Hard Case Mining：让数据跟着模型弱点进化

训练完成后：

```text
Model v1
   ↓
Benchmark
   ↓
Error Analysis
   ↓
发现弱项
   ↓
Generator 针对弱项生成数据
   ↓
加入训练集
   ↓
Model v2
```

例如：

```text
Gradient accuracy = 83%
Shadow accuracy   = 51%
```

下一轮不应该盲目增加全部数据，而应提高 shadow hard cases 的比例。

失败样本应保存到：

```text
hard_cases/
├── gradient/
├── bezier/
├── shadow/
├── occlusion/
└── mixed/
```

每个 hard case 都必须包含 seed / generator version，以便复现。

---

## 12. 数据生成器必须支持“针对性生成”

`SceneGenerator` 不应只是随机生成简单 Scene。

后续应该支持类似：

```text
profile=basic
profile=geometry
profile=gradient
profile=transparency
profile=occlusion
profile=shadow
profile=blur
profile=glow
profile=hard-gradient
profile=hard-shadow
profile=mixed
```

并允许控制：

- object count
- object overlap
- geometry complexity
- gradient complexity
- opacity
- effect strength
- z-order complexity
- canvas size

这样 Evaluation 才能反向驱动 Dataset。

---

## 13. Synthetic 与 Real 数据的关系

Synthetic Dataset 的作用：

> 提供精确监督、稳定 benchmark 和可诊断错误。

它不能证明模型已经泛化到现实世界。

因此后期加入真实 PNG：

```text
Real PNG
   ↓
Network
   ↓
Predicted SVG
   ↓
Renderer
   ↓
Predicted PNG
   ↓
Compare with Real PNG
```

真实 PNG 通常没有 GT Scene，因此主要使用：

```text
Render Loss
```

训练最终形成：

```text
Synthetic:
  Scene Loss + Render Loss

Real:
  Render Loss
```

这样可以利用 Synthetic Oracle 学习结构，又利用真实 PNG 缩小 domain gap。

---

## 14. 不允许通过降低任务难度来达标

以下行为禁止：

- 只生成简单 Logo。
- 只生成纯色对象。
- 删除 gradient。
- 删除复杂 Bézier。
- 删除透明度。
- 删除遮挡 / holes。
- 删除 effects。
- 把 object count 人为限制到不真实的范围。
- 只挑模型表现最好的样本计算指标。

如果某项能力还做不到，必须明确记录为 failure，而不是通过缩小测试范围隐藏问题。

---

## 15. 评估标准

阶段目标可以参考：

### Level 1：基础几何

- circle / rect / ellipse / polygon
- 基础颜色
- 基础透明度

目标：高 SSIM + 低 geometry error。

### Level 2：复杂几何

- Bézier
- 多对象
- z-order
- holes / occlusion

目标：结构恢复稳定。

### Level 3：颜色与渐变

- linear gradient
- radial gradient
- multi-stop
- transparent gradient

目标：gradient-specific benchmark 明显收敛。

### Level 4：视觉效果

- shadow
- blur
- glow
- highlight

目标：不仅恢复主体几何，也恢复主要视觉效果。

### Level 5：真实 PNG

重点评价：

- 视觉一致性
- SVG 合理性
- SVG 大小
- object 数量
- 推理速度
- 泛化能力

不要使用单一 SSIM 作为真实图片最终质量的唯一指标。

---

## 16. 模型大小与部署

当前模型约 4.7M 参数。

FP32 权重大约：

```text
4.7M × 4 bytes ≈ 18.8MB
```

因此已经接近 `<20MB` 的目标。

INT8 后理论权重规模约：

```text
≈ 4.7MB
```

所以部署阶段优先考虑：

```text
FP32 baseline
→ FP16
→ INT8
→ Rust
→ WASM
```

不要在模型结构尚未稳定时提前投入大量 Rust/WASM 工作。

---

## 17. 当前推荐优先级

严格按以下顺序推进：

```text
P0  Synthetic Oracle Benchmark
P1  固定 Benchmark + Feature-level Metrics
P2  Scene + Render 双监督训练
P3  训练到真正收敛
P4  Hard Case Mining
P5  Gradient / Shadow / Blur / Glow 强化
P6  Synthetic + Real 混合训练
P7  INT8
P8  Rust
P9  WASM
```

当前最不应该做的是：

```text
先做 Rust/WASM
先重构整个 Network
盲目增加模型参数
盲目增加随机训练数据
```

必须先用 Benchmark 证明当前瓶颈是什么。

---

## 18. 每次 Codex 修改必须回答三个问题

任何模型或数据管线修改都必须提供：

### Q1：模型变强了吗？

```text
Benchmark before
Benchmark after
```

### Q2：哪里变强？

例如：

```text
Gradient +12%
Bézier +5%
Shadow +1%
```

### Q3：有没有 regression？

例如：

```text
Basic -0.5%
Object count -2%
CPU latency +4ms
SVG size +8%
```

如果没有这些数据，不允许仅凭训练 loss 或主观图片观察声称改进。

---

## 19. 推荐的最终闭环

项目最终应形成：

```text
                 ┌─────────────────┐
                 │  SVG Generator  │
                 └────────┬────────┘
                          ↓
                 Synthetic Oracle
                          ↓
                         PNG
                          ↓
                 ┌─────────────────┐
                 │  Tiny NeuralNet │
                 └────────┬────────┘
                          ↓
                   Predicted Scene
                     /          \
                    /            \
                   ↓              ↓
           Scene Evaluation    Renderer
                   ↓              ↓
             GT comparison   Predicted PNG
                                  ↓
                            Visual Evaluation
                                  ↓
                             Error Analysis
                                  ↓
                           Hard Case Mining
                                  ↓
                          Dataset Evolution
                                  │
                                  └────→ Retrain
```

最终形成：

> **Generate → Train → Evaluate → Diagnose → Generate Hard Cases → Retrain**

这是本项目后续研发的核心闭环。

---

## 20. 给后续 Codex Agent 的工作原则

1. **先阅读 `HANDOFF.md` 和 `HANDOFF2.md`，再修改代码。**
2. 不要因为指标不好就直接扩大模型；先用 feature-level benchmark 找到瓶颈。
3. 不要删除复杂能力来提高 benchmark。
4. 不要用传统 tracing 算法替代 Neural Network 核心任务。
5. Synthetic Ground Truth 是强监督 Oracle，但不是现实世界唯一 SVG 答案。
6. Render similarity 是最终视觉目标。
7. Scene supervision 用于稳定学习结构。
8. 所有 Benchmark 必须固定 seed、可重复、可比较。
9. 新增能力必须同时增加对应的 synthetic cases 和 evaluation metrics。
10. 每次重要修改必须报告 before/after/regression。
11. 不要在模型结构未稳定前开始 Rust/WASM 移植。
12. 如果发现当前设计与本文件冲突，优先通过实验验证，而不是凭直觉修改架构。

---

## 21. 下一项具体任务

下一阶段第一任务不是继续修改 VectorNet，而是实现 **Synthetic Oracle Benchmark System**：

```text
1. 固定 benchmark seeds
2. 生成 PNG + GT Scene + GT SVG
3. 批量运行当前模型
4. 对比 Scene prediction 与 GT Scene
5. 渲染 predicted SVG
6. 对比 predicted PNG 与 input PNG
7. 输出 feature-level metrics
8. 保存失败样本和 seed
9. 输出 benchmark_report.json
10. 输出 benchmark_report.md
```

完成后再根据数据决定：

```text
Network 是否需要修改？
Loss 是否需要修改？
Generator 是否需要修改？
Renderer 是否需要修改？
还是仅仅需要更多训练？
```

**不要在没有 Benchmark 证据之前进行大规模架构重构。**
