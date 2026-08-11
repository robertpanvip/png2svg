package com.pan.png2svg

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.util.concurrent.atomic.AtomicReference

/**
 * png2svg 的 JVM 包装层。
 *
 * 实际的 PNG→SVG 转换由 Rust 写成的原生二进制完成（见 core/Cargo.toml）。
 * 该二进制以资源形式打进 core 的 jar（core/src/main/resources/native/<os>/png2svg[.exe]），
 * 运行时首次抽取到用户缓存目录，之后复用。转换通过进程调用实现。
 */
object Png2SvgConverter {

    private val extracted = AtomicReference<Path?>(null)

    /** 抽取并返回原生二进制路径（只抽取一次）。 */
    @Synchronized
    fun binaryPath(): Path {
        extracted.get()?.let { if (Files.exists(it)) return it }

        val os = currentOs()
        val exeName = if (os == "windows") "png2svg.exe" else "png2svg"
        val resource = "/native/$os/$exeName"

        val stream = Png2SvgConverter::class.java.getResourceAsStream(resource)
            ?: throw IllegalStateException(
                "未找到 $os 平台的 png2svg 原生二进制。请将编译好的二进制放到 " +
                    "core/src/main/resources/native/$os/$exeName 后重新构建。"
            )

        val dir = Files.createDirectories(
            Paths.get(System.getProperty("user.home"), ".png2svg", "bin")
        )
        val exe = dir.resolve(exeName)
        stream.use { Files.copy(it, exe, StandardCopyOption.REPLACE_EXISTING) }
        if (os != "windows") {
            exe.toFile().setExecutable(true)
        }
        extracted.set(exe)
        return exe
    }

    /**
     * 将 [inputPng] 转换并把 SVG 写入 [outputSvg]。
     * 直接调用 `png2svg <input> -o <output>`。
     */
    fun convert(inputPng: Path, outputSvg: Path) {
        val cmd = listOf(binaryPath().toString(), inputPng.toString(), "-o", outputSvg.toString())
        val proc = ProcessBuilder(cmd).redirectErrorStream(true).start()
        val out = proc.inputStream.bufferedReader().readText()
        val code = proc.waitFor()
        if (code != 0) {
            throw IllegalStateException("png2svg 执行失败（退出码 $code）：$out")
        }
    }

    /**
     * 将 [inputPng] 转换并返回 SVG 文本（写到临时文件再读取）。
     */
    fun convert(inputPng: Path): String {
        val tmp = Files.createTempFile("png2svg-", ".svg")
        try {
            convert(inputPng, tmp)
            return Files.readString(tmp)
        } finally {
            runCatching { Files.deleteIfExists(tmp) }
        }
    }

    /**
     * 将 PNG 字节直接转换并返回 SVG 文本（写到临时 PNG 再调用原生二进制）。
     */
    fun convert(pngBytes: ByteArray): String {
        val tmpPng = Files.createTempFile("png2svg-in-", ".png")
        try {
            Files.write(tmpPng, pngBytes)
            return convert(tmpPng)
        } finally {
            runCatching { Files.deleteIfExists(tmpPng) }
        }
    }

    private fun currentOs(): String {
        val name = System.getProperty("os.name").lowercase()
        return when {
            name.contains("win") -> "windows"
            name.contains("mac") || name.contains("darwin") -> "macos"
            else -> "linux"
        }
    }
}
