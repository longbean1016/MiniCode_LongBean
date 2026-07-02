"""glob 文件名模式匹配工具，按通配符查找文件并返回匹配列表。
   参考 Claude Code GlobTool 语义实现。"""

from pathlib import Path
from typing import Any

from app.agent.permissions import PathAccessStatus, PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# 默认最大返回文件数，避免结果过多撑爆上下文
DEFAULT_MAX_RESULTS = 100
# 单次输出字符预算上限
MAX_OUTPUT_CHARS = 10_000


def _validate(input_data: Any) -> dict[str, str]:
    """校验 glob_files 输入参数，确保 pattern 非空、path 可选。"""
    if not isinstance(input_data, dict):
        raise ValueError("glob_files 输入必须是字典，包含 pattern 字段。")

    pattern = input_data.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern 必须是非空字符串。")

    # path 可选，默认从当前工作目录搜索
    search_path = input_data.get("path")
    if search_path is not None and (not isinstance(search_path, str) or not search_path.strip()):
        raise ValueError("path 如果提供，必须是非空字符串。")

    return {
        "pattern": pattern.strip(),
        "path": search_path.strip() if search_path else "",
    }


def _glob_files(
    pattern: str,
    root_dir: Path,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> tuple[list[str], bool]:
    """在 root_dir 下递归执行 glob 匹配，返回 (匹配文件相对路径列表, 是否截断)。"""
    matches: list[str] = []
    for file_path in root_dir.rglob(pattern):
        if not file_path.is_file():
            continue
        try:
            relative = file_path.relative_to(root_dir).as_posix()
        except ValueError:
            relative = file_path.as_posix()
        matches.append(relative)

    # 按修改时间降序排列，最近修改的文件排在前面（参考 Claude Code GlobTool 行为）
    matches.sort(
        key=lambda p: (root_dir / p).stat().st_mtime if (root_dir / p).exists() else 0,
        reverse=True,
    )

    truncated = len(matches) > max_results
    if truncated:
        matches = matches[:max_results]

    return matches, truncated


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """在指定目录下按 glob 模式搜索文件，返回匹配列表和截断标记。"""
    pattern = validated_input["pattern"]
    raw_path = validated_input.get("path", "") or "."

    # 权限管理器 — 校验搜索根目录是否在工作区范围内
    permission_manager = PermissionManager(
        context.cwd,
        additional_workspaces={Path(p) for p in context.additional_workspaces},
        permanent_workspaces={Path(p) for p in context.permanent_workspaces},
    )

    check = permission_manager.check_path_access(raw_path)
    if check.status == PathAccessStatus.OUTSIDE_WORKSPACE:
        return ToolResult(
            ok=False,
            output=f"目标路径不在工作目录范围内：{raw_path}",
            error="WORKSPACE_ACCESS_REQUIRED",
            meta={
                "path": raw_path,
                "action_key": f"workspace::{raw_path}",
                "reason": check.message,
            },
        )

    target_path = check.resolved_path
    if not target_path.exists():
        return ToolResult(
            ok=False,
            output=f"路径不存在：{raw_path}",
            meta={"path": raw_path},
        )

    if not target_path.is_dir():
        return ToolResult(
            ok=False,
            output=f"目标不是目录：{raw_path}\nglob_files 只能搜索目录。",
            error="GLOB_EXPECTS_DIRECTORY",
            meta={"path": raw_path},
        )

    # 执行 glob 匹配
    filenames, truncated = _glob_files(pattern, target_path)

    if not filenames:
        return ToolResult(
            ok=True,
            output=f"未找到匹配 '{pattern}' 的文件。",
            meta={
                "pattern": pattern,
                "search_root": raw_path,
                "num_files": 0,
                "truncated": False,
            },
        )

    # 构建输出，同时做字符级截断保护，避免单个工具结果撑爆上下文
    output_lines: list[str] = []
    current_chars = 0
    for fname in filenames:
        if current_chars + len(fname) + 1 > MAX_OUTPUT_CHARS:
            truncated = True
            break
        output_lines.append(fname)
        current_chars += len(fname) + 1

    output = "\n".join(output_lines)
    if truncated:
        output += f"\n\n(结果已截断。考虑使用更具体的 path 或 pattern 缩小范围。)"

    return ToolResult(
        ok=True,
        output=output,
        meta={
            "pattern": pattern,
            "search_root": raw_path,
            "num_files": len(filenames),
            "truncated": truncated,
        },
    )


# 按 MiniCode 现有 ToolDefinition 模式注册工具
glob_files_tool = ToolDefinition(
    name="glob_files",
    description="按文件名通配符模式快速查找文件。支持 **/*.py 等 glob 语法，返回匹配文件列表。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "要匹配的 glob 模式，例如 '**/*.py' 或 'src/**/*.ts'。",
            },
            "path": {
                "type": "string",
                "description": "搜索根目录。不填默认使用当前工作目录。不要填 'undefined' 或 'null'，不填即可。",
            },
        },
        "required": ["pattern"],
    },
)
