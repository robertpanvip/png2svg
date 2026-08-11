package com.pan.png2svg.idea

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.Messages
import com.intellij.openapi.vfs.LocalFileSystem
import com.intellij.openapi.vfs.VirtualFile
import com.pan.png2svg.Png2SvgConverter
import java.nio.file.Paths

/**
 * 在 Project View / 编辑器标签页右键菜单中，对选中的 PNG 文件执行
 * “Convert to SVG”：调用 core 的 png2svg 二进制，在同目录生成同名 .svg 并打开。
 */
class ConvertToSvgAction : AnAction() {

    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val file = e.getData(CommonDataKeys.VIRTUAL_FILE) ?: return
        if (!isPng(file)) return

        val inputPath = Paths.get(file.path)
        val outputPath = inputPath.parent.resolve(
            inputPath.fileName.toString().removeSuffix(".png").removeSuffix(".PNG") + ".svg"
        )

        try {
            Png2SvgConverter.convert(inputPath, outputPath)
        } catch (ex: Exception) {
            Messages.showErrorDialog(
                project,
                "PNG 转 SVG 失败：${ex.message}",
                "png2svg"
            )
            return
        }

        val outVf: VirtualFile =
            LocalFileSystem.getInstance().refreshAndFindFileByNioFile(outputPath) ?: return
        FileEditorManager.getInstance(project).openFile(outVf, true)
    }

    override fun update(e: AnActionEvent) {
        val file = e.getData(CommonDataKeys.VIRTUAL_FILE)
        e.presentation.isEnabledAndVisible = file != null && isPng(file)
    }

    override fun getActionUpdateThread(): ActionUpdateThread = ActionUpdateThread.BGT

    private fun isPng(file: VirtualFile): Boolean =
        file.isInLocalFileSystem && file.extension.equals("png", ignoreCase = true)
}
