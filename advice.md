# PNG→SVG 当前研发诊断（Advice）

> 基于 `dev` 分支当前代码与最近训练/实验方向的阶段性诊断。
>
> 核心结论：**项目整体没有走偏，但当前已经出现“围绕当前 Synthetic Generator 的可学习性不断修改 Network”的局部走偏风险。下一阶段应暂停继续堆网络，优先用训练完成后的 checkpoint + Oracle Benchmark 验证问题到底来自模型能力还是数据分布。**

---

## 1. 总体判断

当前项目的大方向仍然正确，暂不建议推翻现有架构。

当前核心链路：

```text
PNG
 ↓
CNN Encoder
 ↓
Slot Transformer
 ↓
Scene Slots
 ↓
Path Segments
 ↓
Scene Graph
 ↓
SVG
```

这一方向符合项目最初目标：由神经网络完成 PNG→SVG，模型输出结构化 Scene，而不是直接生成 SVG XML。

当前模型仍处于约 5M 参数规模，符合 3M–8M 的设计约束；CPU 推理也已经不是当前主要瓶颈。

因此：

> **不要因为当前 Appearance 指标较弱而进行大规模 Network 重构。**

---

## 2. 已经被验证正确的部分

### 2.1 Path Segment 表示

从 blob/万能轮廓表示转向显式的：

```text
M / L / Q / C / A / Z
```

是正确的战略方向。

它让模型真正学习 SVG 几何结构，而不是把所有轮廓压缩到一个不可解释的 blob 参数中。

### 2.2 Synthetic SVG→PNG→SVG 闭环

当前项目最有价值的基础设施仍然是：

```text
Synthetic SVG Scene
       ↓
    Renderer
       ↓
      PNG
       ↓
   VectorNet
       ↓
 Predicted Scene
       ↓
    Renderer
       ↓
 Predicted PNG
```

Synthetic Scene 是精确 Ground Truth，因此可以同时进行：

```text
Scene Supervision
+
Render Supervision
```

这应该继续作为整个项目的核心研发闭环。

---

## 3. 当前真正值得警惕的问题：Appearance 优化开始反客为主

最近的研发重点已经明显转向：

```text
为什么 fill RGB 学不好？
```

随后增加了：

- 高分辨率颜色分支
- 32×32 token
- raw RGBA token
- bbox center positional encoding
- Gaussian color pooling
- fill/stroke 专用 feature branch

这些方案作为**实验和诊断工具**是合理的。

但是不建议继续沿着：

```text
颜色不准
 ↓
增加颜色分支
 ↓
更高分辨率
 ↓
更多颜色 token
 ↓
更多颜色 head
 ↓
继续增加模型容量
```

无限堆叠。

原因是：当前实验已经显示，问题很可能不是“网络完全不会预测颜色”，而是**对象级颜色信息在输入图像中本身难以提取**。

---

## 4. 关键诊断：这是 Object Appearance 信息提取问题

当前实验出现了类似这样的相关性差异：

```text
bbox color correlation       ≈ 0.27
eroded bbox                  ≈ 0.42
center gaussian              ≈ 0.35
GT object mask               ≈ 0.62–0.67
```

这说明一个重要事实：

> **如果知道对象真正的 mask，颜色预测明显更容易；单纯使用 bbox / 中心区域并不能可靠得到对象真实 fill。**

而当前 Synthetic Scene 包含：

- 小对象
- overlap
- stroke
- transparency
- gradient
- occlusion
- 多层 composition

所以一个 slot 的 fill 并不是简单的：

```text
bbox → 平均 RGB
```

而更接近：

```text
slot
 ↓
where is object?
 ↓
which pixels belong to object?
 ↓
which pixels are stroke?
 ↓
which pixels are occluded?
 ↓
what is the underlying fill?
```

这已经是 object-centric appearance inference，而不是普通 RGB regression。

---

## 5. 因此现在不要继续修改 Network

建议当前训练完成后，冻结这版 Network，形成一个明确的 baseline。

然后做三个对照实验。

### Experiment A：当前完整 Generator

保持现在的复杂数据分布不变。

记录：

```text
geometry
fill
stroke
gradient
opacity
z-order
render MAE
SSIM
```

### Experiment B：降低遮挡难度，但不改变 Scene 表达能力

例如：

```text
对象面积提高
overlap 降低
极小对象减少
```

Network 完全不改。

如果 Appearance 指标明显提升，则说明主要问题来自数据分布。

### Experiment C：使用 Oracle Object Mask 做颜色诊断

仅用于实验，不作为最终模型输入。

比较：

```text
bbox color
vs
mask color
```

如果 mask 条件下颜色显著变好，则进一步证明当前瓶颈是 object localization / segmentation / occlusion，而不是颜色 head 容量不足。

---

## 6. 当前最重要的工作不是继续训练，而是 Oracle Benchmark

训练结束以后第一优先级应该是：

```text
固定 seed
 ↓
生成 GT Scene
 ↓
生成 PNG
 ↓
运行 checkpoint
 ↓
Predicted Scene
 ↓
Scene-level metrics
 ↓
Predicted SVG
 ↓
resvg
 ↓
Render metrics
```

必须能够回答：

```text
模型哪里不会？
```

而不是只得到：

```text
SSIM = 0.xx
```

建议至少统计：

### Geometry

- bbox IoU
- center error
- width/height error
- segment count error
- segment type accuracy
- control point error

### Appearance

- solid RGB error
- gradient parameter error
- opacity error
- stroke error

### Composition

- object count
- z-order accuracy
- overlap / visibility
- holes

### Rendering

- MAE
- SSIM
- PSNR
- edge similarity
- alpha similarity

### Complexity

- predicted object count
- path count
- control point count
- SVG serialized size

### Performance

- CPU latency
- memory

---

## 7. 一个非常重要的判断标准

如果出现：

```text
Geometry PASS
Appearance FAIL
```

不要直接得出：

```text
Network 太弱
```

应该继续拆解：

```text
Localization
Segmentation
Occlusion
Appearance
```

特别是：

```text
GT mask → color
```

与：

```text
predicted slot → color
```

之间的差距。

如果 GT mask 下颜色已经很好，而 predicted slot 下颜色很差，那么真正应该优化的是 object-centric representation，而不是继续扩大 color head。

---

## 8. Generator 不是越简单越好，但必须控制任务难度

这里不能走另一个极端：为了得到漂亮指标，把 Generator 简化成：

```text
纯色
单对象
无 overlap
无透明度
无 gradient
无 effects
```

这属于指标作弊。

正确做法是建立多个 difficulty/profile：

```text
basic
geometry
bezier
gradient
transparency
occlusion
shadow
blur
glow
mixed
hard-*
```

先让模型在可解释的能力维度逐级收敛，再组合成复杂场景。

最终仍然必须保留完整复杂 benchmark。

---

## 9. 当前建议的训练策略

当前正在进行的训练：

> **让它完整跑完，不要中途再修改架构。**

训练完成后保存为当前 baseline。

然后：

```text
Baseline checkpoint
       ↓
Oracle Benchmark
       ↓
Feature Diagnosis
       ↓
确定真正瓶颈
       ↓
只修改一个变量
       ↓
重新 Benchmark
       ↓
决定是否重训
```

不要同时修改：

```text
Network
+ Generator
+ Loss
+ Renderer
```

否则即使指标发生变化，也无法知道到底是什么产生了影响。

---

## 10. 当前不建议做的事情

### 不建议 1：继续盲目增加颜色网络

当前 HR color branch 可以保留作为实验，但不应该继续无限扩张。

### 不建议 2：现在扩大参数量

当前约 5M 参数已经符合项目目标，没有证据证明参数量是主要瓶颈。

### 不建议 3：现在重做整个 Transformer

当前 geometry 已经表现出明确的可学习性，没有必要因为 appearance 问题推翻整个 decoder。

### 不建议 4：现在做 Rust/WASM

模型结构尚未完全稳定，应该继续保持 Python 训练和评估闭环。

### 不建议 5：只看 SSIM

SSIM 只能说明最终 raster 相似度，不能告诉我们 SVG Scene 哪里出了问题。

---

## 11. 当前项目是否走偏？

结论：

```text
整体路线：        ✅ 没走偏
Scene 表示：      ✅ 正确
Path Segment：    ✅ 正确
Synthetic Oracle：✅ 正确
Geometry 学习：   ✅ 已证明可行
Appearance：      🟡 当前瓶颈
HR Color Branch：  🟡 可以保留实验
继续堆 Color：    🔴 不建议
Generator 对照：  🟢 强烈建议
Oracle Benchmark：🟢 当前最高优先级
```

最重要的一句话：

> **现在不要问“还要给 Network 加什么”，而应该问“当前失败究竟是 Network 的能力边界，还是 Generator 给它制造了无法可靠观察的信息”。**

---

## 12. 下一阶段推荐闭环

```text
Generate
   ↓
Train
   ↓
Evaluate
   ↓
Diagnose
   ↓
Hard Case Mining
   ↓
Targeted Generator
   ↓
Retrain
   ↓
Benchmark Regression
   ↺
```

每次实验必须记录：

```text
Generator version
Network version
Loss version
Renderer version
Seed
Checkpoint
Benchmark metrics
Failure cases
```

只有这样，这个项目才会从“不断试模型”变成一个真正可验证、可迭代的 PNG→SVG 研究/工程系统。

---

## 13. 当前执行顺序

```text
P0  当前训练跑完
P1  固化 checkpoint
P2  Oracle Benchmark
P3  Feature-level Diagnosis
P4  A/B Generator difficulty
P5  确定真正 bottleneck
P6  针对 bottleneck 修改一个变量
P7  Retrain
P8  Regression Benchmark
```

**当前不要跳到 P6。先完成 P0–P3。**
