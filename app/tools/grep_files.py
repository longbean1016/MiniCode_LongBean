from __future__ import annotations

from pathlib import Path
from typing import Any

from app.permissions import PermissionManager
from app.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# 搜索类工具不一定要做 offset/limit 分页，
# 但必须把“总命中数”和“当前返回多少条”明确告诉模型。
DEFAULT_MAX_MATCHES = 200
MAX_MAX_MATCHES = 1_000
MAX_MATCH_LINE_CHARS = 240
MAX_OUTPUT_CHARS = 12_000
_BLOCKED_INTERNAL_DIRS = {".cache", "cache", ".sessions", "sessions", ".context_state", "context_state"}


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
    normalized_path = _to_workspace_relative_path(target_path, context.cwd)

    # 默认不允许把内部上下文目录作为 grep 根目录，避免模型全文检索历史缓存后反复打转。
    blocked_reason = _match_blocked_internal_path(normalized_path)
    if blocked_reason is not None:
        return ToolResult(
            ok=False,
            output=(
                f"默认不允许搜索内部上下文目录：{raw_path}\n"
                f"原因：{blocked_reason}\n"
                "请优先对源码、配置或业务目录进行搜索。"
            ),
            error="SEARCH_POLICY_BLOCKED",
            meta={
                "path": raw_path,
                "normalized_path": normalized_path,
                "blocked_reason": blocked_reason,
            },
        )

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
    content_clipped = False
    output_budget_hit = False
    current_output_chars = 0

    for file_path in target_path.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(target_path)
        # 遍历普通目录时，也跳过内部上下文目录下面的文件，避免把缓存和状态当源码搜索。
        if _contains_internal_context_part(relative_path):
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            # 保持现有宽松行为：读不了的文件直接跳过，避免一次搜索整体失败。
            continue

        for line_num, line in enumerate(content.splitlines(), start=1):
            if pattern not in line:
                continue

            total_matches += 1
            if len(matches) < max_matches:
                clipped_line, line_was_clipped = _clip_match_line(line)
                rendered_line = f"{relative_path}:{line_num}: {clipped_line}"
                projected_chars = current_output_chars + len(rendered_line) + 1

                # 先在工具层做总字符预算，避免超长 grep 结果全压到 registry 才被动收尾。
                if projected_chars > MAX_OUTPUT_CHARS and matches:
                    output_budget_hit = True
                    truncated = True
                    continue

                matches.append(rendered_line)
                current_output_chars = projected_chars
                if line_was_clipped:
                    content_clipped = True
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
                "TRUNCATED: no\n"
                "CONTENT_CLIPPED: no\n"
                "OUTPUT_BUDGET_HIT: no\n\n"
                "没有找到匹配的内容。"
            ),
            meta={
                "pattern": pattern,
                "search_root": raw_path,
                "total_matches": 0,
                "returned_matches": 0,
                "truncated": False,
                "content_clipped": False,
                "output_budget_hit": False,
            },
        )

    header_lines = [
        f"PATTERN: {pattern}",
        f"ROOT: {raw_path}",
        f"TOTAL_MATCHES: {total_matches}",
        f"RETURNED_MATCHES: {len(matches)}",
        f"TRUNCATED: {'yes' if truncated else 'no'}",
        f"CONTENT_CLIPPED: {'yes' if content_clipped else 'no'}",
        f"OUTPUT_BUDGET_HIT: {'yes' if output_budget_hit else 'no'}",
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
            "content_clipped": content_clipped,
            "output_budget_hit": output_budget_hit,
        },
    )


def _to_workspace_relative_path(target_path: Path, cwd: str) -> str:
    """把绝对路径转成工作区相对路径，便于统一做目录策略判断。"""
    workspace_root = Path(cwd).resolve()
    try:
        relative_path = target_path.resolve().relative_to(workspace_root)
    except ValueError:
        return target_path.resolve().as_posix()
    return relative_path.as_posix()


def _match_blocked_internal_path(relative_path: str) -> str | None:
    """判断 grep 根目录是否命中了内部上下文目录。"""
    normalized = relative_path.replace("\\", "/").lstrip("./")
    if not normalized:
        return None

    first_part = normalized.split("/", 1)[0].lower()
    if first_part in _BLOCKED_INTERNAL_DIRS:
        return "这是内部上下文目录，默认不作为全文搜索入口。"
    return None


def _contains_internal_context_part(relative_path: Path) -> bool:
    """判断递归到的文件是否位于内部上下文目录下。"""
    return any(part.lower() in _BLOCKED_INTERNAL_DIRS for part in relative_path.parts)


def _clip_match_line(line: str) -> tuple[str, bool]:
    """裁剪超长命中行，避免单条大日志直接撑爆搜索结果。"""
    if len(line) <= MAX_MATCH_LINE_CHARS:
        return line, False

    keep = max(80, MAX_MATCH_LINE_CHARS - 18)
    return f"{line[:keep]} ...[单行已截断]", True


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
