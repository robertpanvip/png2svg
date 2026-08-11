plugins {
    id("java")
    id("org.jetbrains.kotlin.jvm") version "2.4.0"
}

group = "com.pan"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    // gradle.properties 关闭了默认 stdlib 注入，这里显式引入
    implementation(kotlin("stdlib"))
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
    }
}

// core 同时包含 Rust 源码（core/Cargo.toml）与 Kotlin 包装层。
// Rust 二进制放在 core/src/main/resources/native/<os>/ 下随 jar 分发；
// 重新编译：在 core/ 执行 `cargo build --release`，把产物拷到对应目录。
// 二进制资源默认按字节原样拷贝，无需额外过滤配置。
