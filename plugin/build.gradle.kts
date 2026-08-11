import org.jetbrains.intellij.platform.gradle.TestFrameworkType

plugins {
    id("java")
    id("org.jetbrains.kotlin.jvm") version "2.4.0"
    id("org.jetbrains.intellij.platform") version "2.18.1"
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
    intellijPlatform {
        jetbrainsRuntime()

        // 基础 IDEA Community 即可（本插件只处理图片文件，无需 Ultimate/前端插件）
        intellijIdeaCommunity("LATEST-EAP-SNAPSHOT") {
            useInstaller = false
        }

        // 单元测试用平台测试框架
        testFramework(TestFrameworkType.Platform)
    }

    // 复用 core 的 Kotlin 包装层与原生二进制
    implementation(project(":core"))

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
}
