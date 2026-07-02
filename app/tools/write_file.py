"""写文件工具，支持自动创建父目录和 create/update 语义。
   参考 Claude Code FileWriteTool 语义实现。"""

from pathlib import Path
from typing import Any

from app.agent.permissions import PathAccessStatus, PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, Any]:
    """校验 write_file 输入参数。

       新主字段 file_path（兼容旧 path），content 必填。
    """
    if not isinstance(input_data, dict):
        raise ValueError("write_file 输入必须是字典，包含 file_path 和 content 字段。")

    # ── file_path 为主字段，兼容旧 path ──
    raw_path = input_data.get("file_path") or input_data.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("file_path 必须是非空字符串。")

    # ── content 必填 ──
    content = input_data.get("content")
    if not isinstance(content, str):
        raise ValueError("content 必须是字符串。")

    return {
        "file_path": raw_path.strip(),
        "content": content,
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """把内容写入目标文件，自动创建缺失的父目录。

       返回 create（新建）或 update（更新）语义，
       覆盖前做 stale-check 防止覆盖外部修改。
    """
    permission_manager = PermissionManager(
        context.cwd,
        additional_workspaces={Path(p) for p in context.additional_workspaces},
        permanent_workspaces={Path(p) for p in context.permanent_workspaces},
    )

    raw_path = validated_input["file_path"]
    content = validated_input["content"]

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
    existed = target_path.exists()

    # ── 如果目标已存在但不是文件，拒绝 ──
    if existed and not target_path.is_file():
        return ToolResult(
            ok=False,
            output=f"目标不是文件，无法写入：{raw_path}",
            error="TARGET_NOT_FILE",
            meta={"file_path": raw_path},
        )

    # ── 自动创建父目录（对齐 Claude Code FileWriteTool 行为）──
    parent_dir = target_path.parent
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult(
                ok=False,
                output=f"无法创建父目录：{parent_dir} — {exc}",
                error="PARENT_DIR_CREATE_FAILED",
                meta={"file_path": raw_path, "parent_dir": str(parent_dir)},
            )

    # ── 覆盖前 stale-check：防止模型基于旧 read 版本覆盖掉外部最新修改 ──
    if existed:
        try:
            old_content = target_path.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(
                ok=False,
                output=f"读取原文件失败：{exc}",
                error="READ_BEFORE_WRITE_FAILED",
                meta={"file_path": raw_path},
            )

        # 原内容和要写入的内容完全相同时，视为无操作
        if old_content == content:
            return ToolResult(
                ok=True,
                output=f"文件内容未变化，无需写入：{raw_path}",
                meta={
                    "file_path": raw_path,
                    "bytes": len(content.encode("utf-8")),
                    "chars": len(content),
                    "action": "update",
                    "unchanged": True,
                },
            )

    # ── 执行写入（utf-8）──
    try:
        target_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return ToolResult(
            ok=False,
            output=f"写入文件失败：{exc}",
            error="WRITE_FILE_FAILED",
            meta={"file_path": raw_path},
        )

    action = "create" if not existed else "update"
    output_msg = f"文件{'创建' if not existed else '更新'}成功：{raw_path}"

    return ToolResult(
        ok=True,
        output=output_msg,
        meta={
            "file_path": raw_path,
            "bytes": len(content.encode("utf-8")),
            "chars": len(content),
            "action": action,
        },
    )


# ── 注册工具 ──
write_file_tool = ToolDefinition(
    name="write_file",
    description="把文本内容写入指定文件，自动创建缺失的父目录。已存在的文件将被覆盖。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要写入的文件绝对路径（必须是绝对路径，不能是相对路径）。",
            },
            "content": {
                "type": "string",
                "description": "要写入文件的完整文本内容。",
            },
            "path": {
                "type": "string",
                "description": "(已弃用) 请使用 file_path。",
            },
        },
        "required": ["file_path", "content"],
    },
)
