import org.jetbrains.intellij.platform.gradle.extensions.intellijPlatform

// 集中声明 IntelliJ Platform Gradle Plugin 版本，让所有子模块（含 module 子插件）
// 可以无版本号 apply，避免“already on classpath with an unknown version”冲突。
plugins {
    id("org.jetbrains.intellij.platform.settings") version "2.18.1"
}

// 在 settings 层统一管理依赖仓库，子模块的 build.gradle.kts 不再各自声明 repositories。
// FAIL_ON_PROJECT_REPOS：若有子模块仍声明 repositories 则直接报错，保证一致性。
dependencyResolutionManagement {
    repositoriesMode = RepositoriesMode.FAIL_ON_PROJECT_REPOS
    repositories {
        mavenCentral()
        intellijPlatform {
            defaultRepositories()
        }
    }
}

rootProject.name = "png2svg"

// 多模块结构：
//   core   —— Rust 写的 PNG→SVG 转换内核（CLI 二进制），以及 Kotlin 包装层（作为插件内容模块）
//   plugin —— IntelliJ 平台插件，提供“右键 PNG → 转 SVG”动作，依赖 core
include("core")
include("plugin")
