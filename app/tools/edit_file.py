"""文件编辑工具，按精确字符串匹配替换文件内容。
   参考 Claude Code FileEditTool 语义实现：old_string/new_string 协议、stale-check、diff 摘要。"""

from pathlib import Path
from typing import Any

from app.agent.permissions import PathAccessStatus, PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, Any]:
    """校验 edit_file 输入参数。

       新字段 old_string / new_string + file_path，兼容旧 old_text / new_text / path。
    """
    if not isinstance(input_data, dict):
        raise ValueError("edit_file 输入必须是字典，包含 file_path、old_string 和 new_string 字段。")

    # ── file_path 为主字段，兼容旧 path ──
    raw_path = input_data.get("file_path") or input_data.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("file_path 必须是非空字符串。")

    # ── old_string 为主字段，兼容旧 old_text ──
    old_string = input_data.get("old_string") or input_data.get("old_text")
    if not isinstance(old_string, str):
        raise ValueError("old_string 必须是字符串。")
    if not old_string:
        # 允许空字符串：表示创建新文件（对不存在的文件）或清空文件内容
        pass

    # ── new_string 为主字段，兼容旧 new_text ──
    new_string = input_data.get("new_string") or input_data.get("new_text")
    if not isinstance(new_string, str):
        raise ValueError("new_string 必须是字符串。")

    # ── 新旧内容相同时拒绝（无变化编辑）──
    if old_string == new_string:
        raise ValueError("无变化：old_string 和 new_string 完全相同，不需要编辑。")

    # ── replace_all: 是否替换全部匹配项 ──
    replace_all = bool(input_data.get("replace_all", False))

    return {
        "file_path": raw_path.strip(),
        "old_string": old_string,
        "new_string": new_string,
        "replace_all": replace_all,
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """在已有文件中把 old_string 精确替换为 new_string。"""
    permission_manager = PermissionManager(
        context.cwd,
        additional_workspaces={Path(p) for p in context.additional_workspaces},
        permanent_workspaces={Path(p) for p in context.permanent_workspaces},
    )

    raw_path = validated_input["file_path"]
    old_string = validated_input["old_string"]
    new_string = validated_input["new_string"]
    replace_all = validated_input["replace_all"]

    # ── 路径权限检查 ──
    check = permission_manager.check_path_access(raw_path)
    if check.status == PathAccessStatus.OUTSIDE_WORKSPACE:
        return ToolResult(
            ok=False,
            output=f"目标路径不在工作目录范围内：{raw_path}",
            error="WORKSPACE_ACCESS_REQUIRED",
            meta={
                "file_path": raw_path,
                "resolved_path": str(check.resolved_path) if check.resolved_path else raw_path,
                "action_key": f"workspace::{raw_path}",
                "reason": check.message,
            },
        )
    target_path = check.resolved_path

    # ── 目标不存在 → 如果 old_string 为空则创建新文件 ──
    if not target_path.exists():
        if old_string == "":
            return _create_new_file(target_path, raw_path, new_string)
        return ToolResult(
            ok=False,
            output=f"文件不存在，无法编辑：{raw_path}\n如果要创建新文件，将 old_string 设为空字符串。",
            error="FILE_NOT_FOUND",
            meta={"file_path": raw_path},
        )

    if not target_path.is_file():
        return ToolResult(
            ok=False,
            output=f"目标不是文件，无法编辑：{raw_path}",
            error="TARGET_NOT_FILE",
            meta={"file_path": raw_path},
        )

    # ── 读取原文件内容 ──
    try:
        original_content = target_path.read_text(encoding="utf-8")
    except Exception as exc:
        return ToolResult(
            ok=False,
            output=f"读取文件失败：{exc}",
            error="READ_FILE_FAILED",
            meta={"file_path": raw_path},
        )

    # ── 精确匹配 old_string ──
    if old_string not in original_content:
        return ToolResult(
            ok=False,
            output=f"未找到要替换的文本，文件未修改。\nold_string: {old_string[:200]}{'...' if len(old_string) > 200 else ''}",
            error="OLD_STRING_NOT_FOUND",
            meta={
                "file_path": raw_path,
                "replace_all": replace_all,
                "old_length": len(old_string),
                "file_size": len(original_content),
            },
        )

    # ── 统计出现次数并执行替换 ──
    occurrence_count = original_content.count(old_string)

    if replace_all:
        new_content = original_content.replace(old_string, new_string)
        replaced_count = occurrence_count
    else:
        new_content = original_content.replace(old_string, new_string, 1)
        replaced_count = 1

        # 多匹配但未设置 replace_all 时提醒
        if occurrence_count > 1:
            return ToolResult(
                ok=False,
                output=(
                    f"发现 {occurrence_count} 处匹配，但 replace_all 为 false。"
                    f"要替换全部请设 replace_all=true。"
                    f"要仅替换一处请提供更多上下文使 old_string 唯一。"
                ),
                error="MULTIPLE_MATCHES",
                meta={
                    "file_path": raw_path,
                    "occurrence_count": occurrence_count,
                },
            )

    # ── 写回文件 ──
    try:
        target_path.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        return ToolResult(
            ok=False,
            output=f"写回文件失败：{exc}",
            error="EDIT_FILE_FAILED",
            meta={"file_path": raw_path},
        )

    # ── 构建 diff 摘要（参考 Claude Code FileEditTool）──
    diff_summary = _build_diff_summary(old_string, new_string, replaced_count)

    return ToolResult(
        ok=True,
        output=f"文件编辑成功：{raw_path}\n{diff_summary}",
        meta={
            "file_path": raw_path,
            "replace_all": replace_all,
            "replaced_count": replaced_count,
            "occurrence_count": occurrence_count,
            "old_length": len(old_string),
            "new_length": len(new_string),
        },
    )


def _create_new_file(target_path: Path, raw_path: str, new_string: str) -> ToolResult:
    """old_string 为空且文件不存在时创建新文件（对齐 Claude Code FileEditTool 行为）。"""
    parent_dir = target_path.parent
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult(
                ok=False,
                output=f"无法创建父目录：{parent_dir} — {exc}",
                error="PARENT_DIR_CREATE_FAILED",
                meta={"file_path": raw_path},
            )
    try:
        target_path.write_text(new_string, encoding="utf-8")
    except Exception as exc:
        return ToolResult(
            ok=False,
            output=f"创建文件失败：{exc}",
            error="CREATE_FILE_FAILED",
            meta={"file_path": raw_path},
        )
    return ToolResult(
        ok=True,
        output=f"文件创建成功：{raw_path}\n+{len(new_string.splitlines())} 行",
        meta={"file_path": raw_path, "action": "create", "chars": len(new_string)},
    )


def _build_diff_summary(old: str, new: str, count: int) -> str:
    """构建最小 diff 摘要，帮助模型确认变更效果。"""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    removed = len(old_lines)
    added = len(new_lines)

    parts = []
    if count > 1:
        parts.append(f"替换了 {count} 处匹配。")
    if old_lines and new_lines:
        if removed == added:
            parts.append(f"修改了 {removed} 行。")
        else:
            parts.append(f"-{removed} 行，+{added} 行。")

    # 给出 new_string 的前 3 行作为快速预览
    preview = "\n".join(new_lines[:3])
    if len(new_lines) > 3:
        preview += "..."
    if preview.strip():
        parts.append(f"预览: {preview}")

    return "\n".join(parts)


# ── 注册工具 ──
edit_file_tool = ToolDefinition(
    name="edit_file",
    description="在已有文件中把指定旧文本替换为新文本，支持精确替换和全局替换。old_string 为空且文件不存在时可创建新文件。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要编辑的文件绝对路径。",
            },
            "old_string": {
                "type": "string",
                "description": "文件中需要被替换的文本，必须精确匹配。设为空字符串可创建新文件（当文件不存在时）。",
            },
            "new_string": {
                "type": "string",
                "description": "替换后的新文本内容。",
            },
            "replace_all": {
                "type": "boolean",
                "description": "是否替换全部匹配项，默认 false（只替换第一处）。多匹配时必须设为 true 才能替换全部。",
            },
            "path": {
                "type": "string",
                "description": "(已弃用) 请使用 file_path。",
            },
            "old_text": {
                "type": "string",
                "description": "(已弃用) 请使用 old_string。",
            },
            "new_text": {
                "type": "string",
                "description": "(已弃用) 请使用 new_string。",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    },
)
