import org.jetbrains.intellij.platform.gradle.TestFrameworkType

plugins {
    id("java")
    id("org.jetbrains.kotlin.jvm") version "2.4.0"
    // 版本由 settings.gradle.kts 的 org.jetbrains.intellij.platform.settings 统一托管
    id("org.jetbrains.intellij.platform")
}

group = "com.pan"
// 发布版本跟随 git tag（workflow 传 -PpluginVersion=1.2.3）；本地构建默认 SNAPSHOT
version = providers.gradleProperty("pluginVersion").getOrElse("0.1.0-SNAPSHOT")

// 依赖仓库已在 settings.gradle.kts 的 dependencyResolutionManagement 中统一声明

dependencies {
    intellijPlatform {
        jetbrainsRuntime()

        // 基础 IDEA Community 即可（本插件只处理图片文件，无需 Ultimate/前端插件）
        intellijIdeaCommunity("LATEST-EAP-SNAPSHOT") {
            useInstaller = false
        }

        // 单元测试用平台测试框架
        testFramework(TestFrameworkType.Platform)

        // 关键：用 pluginModule(...) 把 core 作为“插件内容模块”打包进 lib/modules/。
        // 仅写 implementation(project(":core")) 时，IJ 平台插件不会把 core.jar 打进插件包，
        // 运行时插件类加载器找不到 Png2SvgConverter → NoClassDefFoundError。
        pluginModule(implementation(project(":core")))
    }

    testImplementation("junit:junit:4.13.2")
}

intellijPlatform {
    pluginConfiguration {
        ideaVersion {
            sinceBuild = "251"
        }
        changeNotes = """
            0.1.0 初始版本：右键 PNG 文件一键转换为 SVG（由 Rust core 实现）
        """.trimIndent()
    }

    // 发布到 JetBrains 插件市场（release 工作流设置 JETBRAINS_MARKETPLACE_TOKEN 后生效）
    // 2.x DSL：token/channels 是 intellijPlatform.publishing 扩展属性，作为 publishPlugin 任务的默认值。
    // 注意 publishPlugin 本身是任务，不能直接写在 intellijPlatform {} 下（否则报 receiver type mismatch）。
    publishing {
        token.set(providers.environmentVariable("JETBRAINS_MARKETPLACE_TOKEN").orElse(""))
        channels.set(listOf(providers.gradleProperty("pluginChannel").getOrElse("default")))
    }
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

tasks {
    withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
        }
    }
    test {
        useJUnit()
    }

    // 打包产物命名为 png2svg.zip（覆盖默认的 <projectName>-<version>.zip）
    buildPlugin {
        archiveFileName.set("png2svg.zip")
    }
}
