from typing import Any

from app.agent.permissions import PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, Any]:
    """
    校验 edit_file 的输入，并转成统一结构。
    """

    # 输入必须是字典，后面才能安全读取字段
    if not isinstance(input_data, dict):
        raise ValueError("edit_file 输入必须是一个字典，包含 path、old_text 和 new_text 字段。")

    # 读取 path 字段
    path = input_data.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    # 读取 old_text 字段
    old_text = input_data.get("old_text")
    if not isinstance(old_text, str):
        raise ValueError("old_text 必须是字符串")

    # old_text 不能为空，否则会导致替换逻辑失控
    if not old_text:
        raise ValueError("old_text 不能为空字符串")

    # 读取 new_text 字段
    new_text = input_data.get("new_text")
    if not isinstance(new_text, str):
        raise ValueError("new_text 必须是字符串")

    # replace_all 是可选字段，默认只替换第一处
    replace_all = input_data.get("replace_all", False)
    if not isinstance(replace_all, bool):
        raise ValueError("replace_all 必须是布尔值")

    # 返回规范化后的输入，后续 runner 直接使用
    return {
        "path": path.strip(),
        "old_text": old_text,
        "new_text": new_text,
        "replace_all": replace_all,
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """
    在已有文件内容中把 old_text 替换成 new_text。
    """

    # 用当前工作目录作为权限边界
    permission_manager = PermissionManager(context.cwd)

    # 取出校验后的输入
    raw_path = validated_input["path"]
    old_text = validated_input["old_text"]
    new_text = validated_input["new_text"]
    replace_all = validated_input["replace_all"]

    # 检查路径是否越界，并解析成绝对路径
    target_path = permission_manager.ensure_path_access(raw_path)

    # 目标不存在时，不能编辑
    if not target_path.exists():
        return ToolResult(
            ok=False,
            output=f"文件不存在，无法编辑：{raw_path}",
            error="FILE_NOT_FOUND",
            meta={
                "path": raw_path,
            },
        )

    # 目标不是文件时，不能编辑
    if not target_path.is_file():
        return ToolResult(
            ok=False,
            output=f"目标不是文件，无法编辑：{raw_path}",
            error="TARGET_NOT_FILE",
            meta={
                "path": raw_path,
            },
        )

    try:
        # 先读取原文件内容，后面基于原内容做替换
        content = target_path.read_text(encoding="utf-8")
    except Exception as error:
        # 读取失败时直接返回错误结果
        return ToolResult(
            ok=False,
            output=f"读取文件失败：{error}",
            error="READ_FILE_FAILED",
            meta={
                "path": raw_path,
            },
        )

    # 原内容里找不到 old_text 时，直接返回失败
    if old_text not in content:
        return ToolResult(
            ok=False,
            output="未找到要替换的旧文本，文件未修改。",
            error="OLD_TEXT_NOT_FOUND",
            meta={
                "path": raw_path,
                "replace_all": replace_all,
            },
        )

    # 统计旧文本在文件中出现了多少次，便于返回给模型
    occurrence_count = content.count(old_text)

    # 按 replace_all 决定替换一处还是全部替换
    if replace_all:
        new_content = content.replace(old_text, new_text)
        replaced_count = occurrence_count
    else:
        new_content = content.replace(old_text, new_text, 1)
        replaced_count = 1

    try:
        # 把替换后的完整内容写回文件
        target_path.write_text(new_content, encoding="utf-8")
    except Exception as error:
        # 写回失败时返回统一错误结果
        return ToolResult(
            ok=False,
            output=f"写回文件失败：{error}",
            error="EDIT_FILE_FAILED",
            meta={
                "path": raw_path,
            },
        )

    # 返回成功结果，并附带替换统计信息
    return ToolResult(
        ok=True,
        output=f"文件编辑成功：{raw_path}",
        meta={
            "path": raw_path,
            "replace_all": replace_all,
            "replaced_count": replaced_count,
            "occurrence_count": occurrence_count,
            "old_length": len(old_text),
            "new_length": len(new_text),
        },
    )


edit_file_tool = ToolDefinition(
    name="edit_file",
    description="在已有文件中把指定旧文本替换为新文本，可选择只替换一处或全部替换",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要编辑的文件路径，必须在工作目录内",
            },
            "old_text": {
                "type": "string",
                "description": "文件中需要被替换的旧文本，不能为空",
            },
            "new_text": {
                "type": "string",
                "description": "替换后的新文本",
            },
            "replace_all": {
                "type": "boolean",
                "description": "是否替换全部匹配项，默认 false 表示只替换第一处",
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
)