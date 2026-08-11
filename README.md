# png2svg

PNG → SVG 转换工具，面向图标场景：按颜色 / alpha 区分轮廓，自动把每个区域识别为
**纯色 / 渐变 / 阴影**，输出简洁的 SVG（真圆/真椭圆弧 + 贝塞尔曲线 + 渐变）。

本项目是一个 **IntelliJ 平台插件** 的多模块 Gradle 工程：

```
png2svg/
├── settings.gradle.kts        # 根：多模块，include("core", "plugin")
├── gradle.properties
├── gradle/wrapper/            # Gradle 9.4.1 wrapper（复制自参考工程）
├── core/                      # 转换内核（core 模块）
│   ├── Cargo.toml             # Rust 工程（实际算法在这里）
│   ├── src/...                # Rust 源码：contour / gradient / segment / svg ...
│   ├── tests/                 # 25 icon + 30 回归往返测试套件
│   ├── build.gradle.kts       # Kotlin/JVM 模块
│   └── src/main/kotlin/com/pan/png2svg/Png2SvgConverter.kt
│       └── src/main/resources/native/<os>/png2svg[.exe]   # 随 jar 分发的原生二进制
└── plugin/                    # IntelliJ 插件模块（plugin 模块）
    ├── build.gradle.kts       # IntelliJ Platform Gradle Plugin 2.18.1
    └── src/main/
        ├── kotlin/com/pan/png2svg/idea/ConvertToSvgAction.kt
        └── resources/META-INF/plugin.xml
```

## 模块职责

- **core**：Rust 写成的 `png2svg` CLI（真正的转换算法）+ 一个 Kotlin 包装层
  `Png2SvgConverter`，负责从 classpath 抽取原生二进制、进程调用、返回 SVG。
  原生二进制以资源形式打包进 core 的 jar，按平台分目录：
  `core/src/main/resources/native/{windows,linux,macos}/`。
- **plugin**：IntelliJ 平台插件，依赖 `core`。在 Project View / 编辑器标签页右键
  PNG 文件提供 **Convert to SVG** 动作，在同目录生成同名 `.svg` 并打开。

## 开发 / 构建

需要：JDK 21、Rust 工具链（仅改 core 时需要）、IntelliJ IDEA 2025.1+（或 EAP）。

```bash
# 1) 重新编译 Rust 内核（可选，仅当改了 core 的 Rust 代码）
cd core && cargo build --release
#    把产物拷到资源目录（Windows 示例）：
#    cp target/release/png2svg.exe src/main/resources/native/windows/

# 2) 在 IDEA 中打开本工程，运行 Run/Debug Configuration
#    “Run IDE with Plugin”（插件模块自带 .run 配置）
#    或在终端：
./gradlew :plugin:buildPlugin      # 产出可安装的 zip
./gradlew :plugin:runIde           # 启动带插件的 IDE 沙箱
```

## 使用

在 IDEA 里右键任意 `.png` 文件 → **Convert to SVG** → 同目录生成 `<原名>.svg` 并自动打开。

## 跨平台

目前资源目录已放置 `windows/png2svg.exe`（Release 构建）。要在 Linux / macOS 分发，
需分别编译对应平台二进制并放到 `core/src/main/resources/native/linux/`、`macos/`。
缺失平台时会给出明确报错而不是静默失败。
