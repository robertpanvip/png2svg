# 新表示 40k 晨报（MORNING REPORT）

日期：2026-09-10 03:30 ｜ ckpt：`runs/newrep/last.pt`（40k）｜ 结论先行：**表示重定向成功（几何/类型全达标），但外观（颜色/覆盖）通路未学会，是当前唯一硬瓶颈**——这也解释了为什么 prune 会把对象全剪掉。

---

## 一、验收数字（40k，50 场景 seed=777000）

| 指标 | 24k | **40k** | 旧表示基线（24k） |
|---|---|---|---|
| 诊断 mae | 0.0541 | **0.0531**（+2%，趋饱和） | 0.069 |
| 对象匹配率 | 132/132 (100%) | **212/212 (100%)** | 总匹配率 0.32 |
| 段类型分布 | C:485/L:109/A:60/M:132 | **C:804/L:178/A:122/M:212**（无坍缩） | 240/240 全 blob |
| seg_type_acc（A/L/C/M） | .58/.26/.86/.91 | **.61/.34/.90/.89** | — |
| nseg_gap | −1.78 | −2.15（略差） | — |
| CPU total p95 | 39.8ms | **~33ms**（net 19.2ms，under_1s=1.0） | ~27ms net |
| Q 段预测 | 0 | **0（被 C 完全吸收）** | — |

**表示层判定：PASS**。blob 万能类坍缩结构性消灭；匹配满分；类型多样化。同配置继续加步数收益递减（24k→40k 仅 +2%），**不再盲训**。

## 二、外观瓶颈（本轮最重要的发现）

诊断链路（12 场景误差分解 + 可视化 + fill 头状态检查）：

| 证据 | 数字 |
|---|---|
| 对象净收益（bg_only − full） | **mean −0.011，10/12 场景负收益**——画对象比不画还差 |
| 背景预测 | (0.16, 0.15, 0.17)，而 GT 背景=黑 (0,0,0)——连平凡背景都没学会 |
| fill 头状态 | fill_alpha 全 1.0（已激活）、ftype none/solid 各半、rgb 亮度 0.375（在学极慢） |
| 视觉对比 | GT=蓝色圆角块+黑花孔洞+品红三角；预测=灰色细条（见 `vis_seed0_*.png`） |
| prune 全剪根因 | 对象贡献≈0 或为负 → 每个对象都"不可替代性不足"→ 剪光（**行为正确，诚实反映质量**） |

**结论：几何监督（bbox L1 + 段类型 NLL + 坐标 L1）已把结构学会；像素通路（fill 颜色/覆盖）几乎只靠渲染 MAE 的间接梯度，学不动——与旧表示"对象让 mae 比不画还高"是同一病根，与表示无关。**

## 三、部署包状态（deploy/，已可用）

- 权重：fp32 **19.21MB**（<20MB 达标）/ INT8 16.72MB；CPU 推理 net **19-27ms**、总延迟 <0.5s（greedy prune）/ 85ms（independent）。
- `benchmarks/prune.py` 新增 **`prune_scene_greedy`**（贪心迭代剪枝，逐层剥冗余副本）；`deploy/predict.py` 默认启用（`--prune-mode` 可切回）。
- 诚实声明：**当前权重下 demo SVG 剪后为空**（对象像素贡献为负，剪枝如实反映）。不带 prune 的完整输出见 `demo_pred_noprune.svg`。外观瓶颈解决后 demo 即有效。

## 四、下一步（按优先级，均已写入 HANDOFF §11.10）

1. **P-Appearance（P0）**：给 fill 加直接监督——匹配对上 fill rgb/alpha 的 L1（GT 有真值），替代只靠渲染间接梯度；同时检查 bg 颜色监督（预测 0.16 vs 真值 0，w_bg=0.2 疑似过弱/编码有 bug）。
2. **P-Coverage**：对象覆盖 warmup——训练早期降低 SSIM 权重或加 fill-coverage 正则，避免"细条"局部最优。
3. **Q 段**：接受 C 吸收（渲染无损）或加 Q 场景过采样；L 准确率 0.34 同理（段型混淆矩阵分析先行）。
4. ** prune 副本问题**：贪心剪枝已兜底；训练侧可加 slot 重复惩罚（同 bbox IoU 正则）。

## 五、产物清单

- 权重：`runs/newrep/last.pt`（40k）｜ 部署权重：`deploy/weights/`（fp32/INT8/manifest）
- 诊断：`runs/diag_new_24k.json` / `runs/diag_new_40k.json` / `runs/eval_newrep40k.json` / `runs/newrep/ACCEPTANCE_24K.md`
- 可视化：`vis_seed0_gt.png` vs `vis_seed0_pred.png`（对比最强烈）、`vis_seed1_*.png`、`demo_pred_noprune.svg`
- 代码：`benchmarks/diag_new.py`（新契约诊断）、`benchmarks/prune.py`（greedy）、`deploy/*`（推理包）
