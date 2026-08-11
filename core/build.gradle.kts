plugins {
    // Kotlin/JVM 编译包装层（Png2SvgConverter.kt）—— 多模块下 Kotlin 子模块需显式 apply
    id("org.jetbrains.kotlin.jvm") version "2.4.0"
    // IntelliJ 平台“模块”插件：让本模块作为“插件内容模块”被主插件自动打包进 lib/modules/。
    // 注意：不能用普通的 java + kotlin.jvm 当子模块，否则主插件不会把它打进分发包，
    // 运行时插件类加载器找不到其中的类（NoClassDefFoundError）。
    id("org.jetbrains.intellij.platform.module")
}

group = "com.pan"
version = "0.1.0"

repositories {
    mavenCentral()
    intellijPlatform {
        defaultRepositories()
    }
}

dependencies {
    // gradle.properties 关闭了默认 stdlib 注入，这里显式引入
    implementation(kotlin("stdlib"))
}

// 主插件（plugin 模块）声明了目标平台；core 作为内容模块继承该目标，无需单独声明 intellijPlatform {} target。
// JVM 21 与目标平台（IDEA 2025.1 / sinceBuild 251）保持一致。
kotlin {
    jvmToolchain(21)
}

// core 同时包含 Rust 源码（core/Cargo.toml，Cargo 自带 src/ 布局，被 Gradle 源集忽略）与 Kotlin 包装层。
// Rust 二进制放在 core/src/main/resources/native/<os>/ 下随模块 jar 分发；重新编译：
// 在 core/ 执行 `cargo build --release`，把产物拷到对应目录。
// 二进制资源默认按字节原样拷贝，无需额外过滤配置。
