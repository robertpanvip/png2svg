# png2svg

面向 **icon** 的 PNG → SVG 转换工具（Rust 实现）。

与常见的「先二值化、再描边」思路不同，本工具**不单独做黑白轮廓**：
轮廓来自**颜色 / alpha 差异自然划分出的区域边界**——相邻像素颜色几乎一致就并入同一区域，
区域之间的边界就是轮廓。每个区域再被识别为 **纯色 / 渐变 / 阴影** 三类，并据此生成 SVG。

## 算法概览

1. **前景掩码**
   - 有透明通道：alpha ≥ 阈值即前景（阈值调低可纳入半透明阴影）。
   - 无透明通道：以四边平均色为背景，颜色距离 > 阈值即前景（`--invert` 可翻转）。

2. **区域分割（替代「黑白轮廓」）** — `src/segment.rs`
   - 对每个未访问像素做 **4 邻域漫水填充**，扩展条件是「与相邻像素的综合距离 ≤ 容差」：
     `距离 = √(RGB² + (α权重 · α差)²)`。
   - 即「遍历每个像素每个方向」：相邻色一致 → 合并为**纯色/渐变区域**；半透明像素因 α 差巨大 → 自动切出**阴影区域**。
   - 区域边界用 Moore 邻域跟踪 + 孔洞检测提取（`src/contour.rs`）：圆/椭圆拟合为**真弧**；曲线型轮廓用 **Catmull-Rom 样条**平滑为密集曲线；直线/直角轮廓保留多边形；`evenodd` 保留内孔。

3. **三类识别** — `src/gradient.rs` + `src/main.rs`
   - 对每个区域做**最小二乘平面拟合** `c = a + b·x + c·y`（每通道独立）。
   - 判定顺序：
     - **阴影**：区域平均 alpha 低于 `--shadow-alpha` → 渲染时降低不透明度，画在**最底层**；
     - **渐变**：平均 R² 高 **且** 整片颜色变化量（梯度幅值 × 投影跨度）> `--solid-eps`；
     - **纯色**：其余兜底，取区域均值色。
   - 说明：这里用「R² + 整片颜色变化量」判渐变，而非「每像素梯度幅值」——后者会把
     每像素只变一点点、但整片跨度很大的**真实平缓渐变**误判成纯色。

4. **SVG 渲染** — `src/svg.rs`
   - 渐变区域 → `<linearGradient>` + `fill="url(#g)"`；
   - 纯色区域 → `fill="rgb(...)"`；
   - 阴影区域 → 同上但带 `fill-opacity`，且绘制顺序排在最前（底层）。

## 构建

```bash
cd png2svg
cargo build --release
# 二进制位于 target/release/png2svg(.exe)
```

> 国内网络建议在 `~/.cargo/config.toml` 配置 rsproxy 等镜像加速依赖下载。

## 用法

```bash
# 转换单个图标
png2svg input.png -o output.svg

# 生成测试图标（含渐变环 + 纯色块 + 半透明阴影）
png2svg gen-test test_icon.png

# 转换测试图标，查看三类识别效果
png2svg test_icon.png -o out.svg
```

### 可调参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tolerance N` | 28 | 颜色 + alpha 分割容差；越大区域越「粗」，越小越细碎 |
| `--alpha N` | 8 | 前景 alpha 阈值（0–255）；调低可纳入更淡的阴影 |
| `--bg-thresh N` | 40 | 无 alpha 时的背景距离阈值 |
| `--shadow-alpha N` | 230 | 区域平均 alpha 低于此值判为阴影 |
| `--invert` | 关 | 无 alpha 时翻转前景 / 背景 |
| `--solid-eps N` | 8 | 判定为渐变的最小整片颜色变化量 |
| `--r2 N` | 0.6 | 判定为渐变的最小 R² |
| `--simplify N` | 0.5 | 轮廓简化精度（越小点越密） |
| `--no-smooth` | 关 | 关闭曲线平滑，回退为纯多边形（配合小 `--simplify` 可得极密多边形） |
| `--circ-tol N` | 0.06 | 圆拟合容许误差/半径（越小越严格） |
| `--ell-tol N` | 0.06 | 椭圆拟合容许误差（越小越严格） |

## 项目结构

```
src/raster.rs    图像加载（RGBA）、前景掩码辅助
src/segment.rs   按颜色 + alpha 漫水填充分割区域
src/contour.rs   Moore 边界跟踪 + 孔洞检测 + RDP 简化
src/gradient.rs  最小二乘平面拟合与纯色/渐变判定
src/svg.rs       SVG 文档构建（渐变/纯色/阴影三类）
src/main.rs      CLI 串联三步流程，含 gen-test
```

## 示例输出

`gen-test` 生成的图标含 4 个区域，转换后：
- 渐变环 → `<linearGradient>`（含内孔，`evenodd` 保留）；
- 绿块 / 红块 → 纯色填充；
- 半透明椭圆 → `fill-opacity` 降低不透明度，绘制在底层。

## License

MIT（如需要，可后续补充 LICENSE 文件）。
