"""写文件工具，负责按权限约束把内容写入目标文件。"""

from typing import Any

from app.permissions import PermissionManager
from app.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, Any]:
    """
    校验 write_file 的输入，并转成统一结构。
    """

    # 输入必须是字典，后面才能安全读取字段
    if not isinstance(input_data, dict):
        raise ValueError("write_file 输入必须是一个字典，包含 path 和 content 字段。")

    # 读取 path 字段
    path = input_data.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    # 读取 content 字段
    content = input_data.get("content")
    if not isinstance(content, str):
        raise ValueError("content 必须是字符串")

    # overwrite 是可选字段，默认不覆盖已有文件
    overwrite = input_data.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值")

    # 返回规范化后的输入，后续 runner 直接使用
    return {
        "path": path.strip(),
        "content": content,
        "overwrite": overwrite,
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """
    把内容写入目标文件。
    """

    # 用当前工作目录作为权限边界
    permission_manager = PermissionManager(context.cwd)

    # 取出校验后的输入
    raw_path = validated_input["path"]
    content = validated_input["content"]
    overwrite = validated_input["overwrite"]

    # 检查路径是否越界，并解析成绝对路径
    target_path = permission_manager.ensure_path_access(raw_path)

    # 如果目标已存在且不是文件，直接拒绝
    if target_path.exists() and not target_path.is_file():
        return ToolResult(
            ok=False,
            output=f"目标不是文件，无法写入：{raw_path}",
            error="TARGET_NOT_FILE",
            meta={
                "path": raw_path,
            },
        )

    # 如果目标文件已存在，但本次不允许覆盖，直接返回失败
    if target_path.exists() and not overwrite:
        return ToolResult(
            ok=False,
            output=f"文件已存在，且 overwrite=False：{raw_path}",
            error="FILE_ALREADY_EXISTS",
            meta={
                "path": raw_path,
                "overwrite": overwrite,
            },
        )

    # 如果父目录不存在，这一版先明确报错，让模型先调用建目录工具
    parent_dir = target_path.parent
    if not parent_dir.exists():
        return ToolResult(
            ok=False,
            output=f"父目录不存在，请先创建目录：{parent_dir}",
            error="PARENT_DIR_NOT_FOUND",
            meta={
                "path": raw_path,
                "parent_dir": str(parent_dir),
            },
        )

    try:
        # 使用 utf-8 写入，保证和现有工具编码风格一致
        target_path.write_text(content, encoding="utf-8")
    except Exception as error:
        # 写入异常统一转成失败结果，避免主循环崩溃
        return ToolResult(
            ok=False,
            output=f"写入文件失败：{error}",
            error="WRITE_FILE_FAILED",
            meta={
                "path": raw_path,
            },
        )

    # 返回成功结果，并带一些基础元信息方便后续调试
    return ToolResult(
        ok=True,
        output=f"文件写入成功：{raw_path}",
        meta={
            "path": raw_path,
            "bytes": len(content.encode('utf-8')),
            "chars": len(content),
            "overwrite": overwrite,
        },
    )


write_file_tool = ToolDefinition(
    name="write_file",
    description="把文本内容写入指定文件，可选择是否覆盖已有文件",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要写入的文件路径，必须在工作目录内",
            },
            "content": {
                "type": "string",
                "description": "要写入文件的完整文本内容",
            },
            "overwrite": {
                "type": "boolean",
                "description": "是否允许覆盖已有文件，默认 false",
            },
        },
        "required": ["path", "content"],
    },
)
