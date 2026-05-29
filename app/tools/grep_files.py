from __future__ import annotations

from typing import Any

from app.permissions import PermissionManager
from app.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# 搜索类工具不一定要做 offset/limit 分页，
# 但必须把“总命中数”和“当前返回多少条”明确告诉模型。
DEFAULT_MAX_MATCHES = 200
MAX_MAX_MATCHES = 1_000


def _validate(input_data: Any) -> dict[str, int | str]:
    """校验 grep_files 输入，并支持 max_matches 控制返回条数。"""
    if not isinstance(input_data, dict):
        raise ValueError("grep_files 输入必须是一个字典，包含 pattern 和 path 字段。")

    pattern = input_data.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern 必须是非空字符串")

    path = input_data.get("path", ".")
    if not isinstance(path, str):
        raise ValueError("path 必须是字符串")

    raw_max_matches = input_data.get("max_matches", DEFAULT_MAX_MATCHES)
    try:
        max_matches = int(raw_max_matches)
    except (TypeError, ValueError) as error:
        raise ValueError("max_matches 必须是整数") from error

    if max_matches < 1 or max_matches > MAX_MAX_MATCHES:
        raise ValueError(f"max_matches 必须在 1 到 {MAX_MAX_MATCHES} 之间")

    return {
        "pattern": pattern.strip(),
        "path": path.strip(),
        "max_matches": max_matches,
    }


def _run(validated_input: dict[str, int | str], context: ToolContext) -> ToolResult:
    """在目录内递归搜索文本，并返回显式的截断说明。"""
    permission_manager = PermissionManager(context.cwd)

    pattern = str(validated_input["pattern"])
    raw_path = str(validated_input["path"])
    max_matches = int(validated_input["max_matches"])
    target_path = permission_manager.ensure_path_access(raw_path)

    if not target_path.exists():
        return ToolResult(
            ok=False,
            output=f"路径不存在：{raw_path}",
        )

    if not target_path.is_dir():
        return ToolResult(
            ok=False,
            output=f"目标不是目录：{raw_path}",
        )

    matches: list[str] = []
    total_matches = 0
    truncated = False

    for file_path in target_path.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            # 保持现有宽松行为：读不了的文件直接跳过，避免一次搜索整体失败。
            continue

        relative_path = file_path.relative_to(target_path)
        for line_num, line in enumerate(content.splitlines(), start=1):
            if pattern not in line:
                continue

            total_matches += 1
            if len(matches) < max_matches:
                matches.append(f"{relative_path}:{line_num}: {line}")
            else:
                truncated = True

    if total_matches == 0:
        return ToolResult(
            ok=True,
            output=(
                f"PATTERN: {pattern}\n"
                f"ROOT: {raw_path}\n"
                "TOTAL_MATCHES: 0\n"
                "RETURNED_MATCHES: 0\n"
                "TRUNCATED: no\n\n"
                "没有找到匹配的内容。"
            ),
            meta={
                "pattern": pattern,
                "search_root": raw_path,
                "total_matches": 0,
                "returned_matches": 0,
                "truncated": False,
            },
        )

    header_lines = [
        f"PATTERN: {pattern}",
        f"ROOT: {raw_path}",
        f"TOTAL_MATCHES: {total_matches}",
        f"RETURNED_MATCHES: {len(matches)}",
        f"TRUNCATED: {'yes' if truncated else 'no'}",
        "",
    ]
    output = "\n".join(header_lines + matches)

    return ToolResult(
        ok=True,
        output=output,
        meta={
            "pattern": pattern,
            "search_root": raw_path,
            "total_matches": total_matches,
            "returned_matches": len(matches),
            "truncated": truncated,
        },
    )


grep_files_tool = ToolDefinition(
    name="grep_files",
    description="在指定目录中搜索包含目标文本的文件内容，并显式返回命中统计。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "要搜索的文本内容，必填。",
            },
            "path": {
                "type": "string",
                "description": "要搜索的目录路径，选填，默认为当前目录。",
            },
            "max_matches": {
                "type": "integer",
                "description": f"最多返回多少条匹配，默认 {DEFAULT_MAX_MATCHES}，最大 {MAX_MAX_MATCHES}。",
            },
        },
        "required": ["pattern"],
    },
)
