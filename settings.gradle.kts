rootProject.name = "png2svg"

// 多模块结构：
//   core   —— Rust 写的 PNG→SVG 转换内核（CLI 二进制），以及 Kotlin 包装层
//   plugin —— IntelliJ 平台插件，提供“右键 PNG → 转 SVG”动作，依赖 core
include("core")
include("plugin")
