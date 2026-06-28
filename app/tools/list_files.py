"""列目录工具，负责输出目录结构并控制返回规模。"""

from pathlib import Path
from typing import Any

from app.agent.permissions import PathAccessStatus, PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

DEFAULT_MAX_ENTRIES = 200
MAX_MAX_ENTRIES = 1_000
MAX_ENTRY_LINE_CHARS = 180
MAX_OUTPUT_CHARS = 8_000
_BLOCKED_INTERNAL_DIRS = {".cache", "cache", ".sessions", "sessions", ".context_state", "context_state"}


def _validate(input_data: Any) -> dict[str, int | str]:
    """
    校验并规范化工具输入。

    支持的输入格式：
    1. {"path": "app"}
    2. {} 或 None，默认使用当前目录 "."
    """
    if input_data is None:
        return {"path": ".", "max_entries": DEFAULT_MAX_ENTRIES}

    if not isinstance(input_data, dict):
        raise ValueError("输入必须是一个字典，包含 'path' 键")

    path = input_data.get("path", ".")
    if not isinstance(path, str):
        raise ValueError("路径必须是一个字符串")

    raw_max_entries = input_data.get("max_entries", DEFAULT_MAX_ENTRIES)
    try:
        max_entries = int(raw_max_entries)
    except (TypeError, ValueError) as error:
        raise ValueError("max_entries 必须是整数") from error

    if max_entries < 1 or max_entries > MAX_MAX_ENTRIES:
        raise ValueError(f"max_entries 必须在 1 到 {MAX_MAX_ENTRIES} 之间")

    return {"path": path, "max_entries": max_entries}


def _run(validated_input: dict[str, int | str], context: ToolContext) -> ToolResult:
    """
    执行列出文件工具。

    功能：
    1. 检查目标路径是否在允许的工作目录内
    2. 判断路径是否存在
    3. 如果是文件，直接返回文件信息
    4. 如果是目录，返回目录下的文件和子目录列表
    """

    # 第一步：创建权限管理器
    # 这里把 context.cwd 当作当前允许操作的工作根目录
    permission_manager = PermissionManager(
        context.cwd,
        additional_workspaces={Path(p) for p in context.additional_workspaces},
        permanent_workspaces={Path(p) for p in context.permanent_workspaces},
    )

    # 第二步：拿到用户想查看的原始路径
    raw_path = validated_input["path"]

    # 第三步：权限管理器检查路径访问是否合法
    check = permission_manager.check_path_access(raw_path)
    if check.status == PathAccessStatus.OUTSIDE_WORKSPACE:
        return ToolResult(
            ok=False,
            output=f"目标路径不在工作目录范围内：{raw_path}",
            error="WORKSPACE_ACCESS_REQUIRED",
            meta={
                "path": raw_path,
                "resolved_path": str(check.resolved_path) if check.resolved_path else raw_path,
                "action_key": f"workspace::{raw_path}",
                "reason": check.message,
            },
        )
    target_path = check.resolved_path
    max_entries = int(validated_input["max_entries"])
    normalized_path = _to_workspace_relative_path(target_path, context.cwd)

    # 默认不允许把内部上下文目录再喂回模型，避免 agent 看到这些目录后继续空转。
    blocked_reason = _match_blocked_internal_path(normalized_path)
    if blocked_reason is not None:
        return ToolResult(
            ok=False,
            output=(
                f"默认不允许列出内部上下文目录：{raw_path}\n"
                f"原因：{blocked_reason}\n"
                "请优先查看正常源码、配置或业务目录。"
            ),
            error="LIST_POLICY_BLOCKED",
            meta={
                "path": str(raw_path),
                "normalized_path": normalized_path,
                "blocked_reason": blocked_reason,
            },
        )

    # 第四步：检查路径是否存在
    if not target_path.exists():
        return ToolResult(
            ok=False,
            output=f"路径不存在: {target_path}"
        )

    # 第五步：如果目标本身就是文件，
    # 那就没必要再迭代目录了，直接告诉调用方这是一个文件
    if target_path.is_file():
        return ToolResult(
            ok=True,
            output=(
                f"ROOT: {raw_path}\n"
                "TOTAL_ENTRIES: 1\n"
                "RETURNED_ENTRIES: 1\n"
                "TRUNCATED: no\n\n"
                f"file {target_path.name}"
            ),
            meta={
                "search_root": raw_path,
                "total_entries": 1,
                "returned_entries": 1,
                "truncated": False,
            },
        )

    # 第六步：如果是目录，就读取目录下所有子项
    # 用名字的小写排序，保证输出稳定，便于测试和调试

    entries = sorted(target_path.iterdir(), key=lambda item: item.name.lower())

    # 第七步：如果目录是空的，返回一个明确提示
    if not entries:
        return ToolResult(
            ok=True,
            output=(
                f"ROOT: {raw_path}\n"
                "TOTAL_ENTRIES: 0\n"
                "RETURNED_ENTRIES: 0\n"
                "TRUNCATED: no\n\n"
                "(empty)"
            ),
            meta={
                "search_root": raw_path,
                "total_entries": 0,
                "returned_entries": 0,
                "truncated": False,
            },
        )

    # 第八步：把每个子项转换成文本行
    # 目录前面标记 dir，文件前面标记 file
    lines: list[str] = []
    omitted_internal_entries = 0
    output_budget_hit = False

    for entry in entries:
        # 目录工具默认把内部上下文目录藏起来，减少模型继续钻进 .cache/.sessions 这类路径。
        if _is_internal_context_name(entry.name):
            omitted_internal_entries += 1
            continue

        prefix = "dir " if entry.is_dir() else "file"
        line = _clip_entry_line(f"{prefix} {entry.name}")
        lines.append(line)

    returned_lines: list[str] = []
    current_chars = 0
    for line in lines[:max_entries]:
        projected_chars = current_chars + len(line) + 1
        # 在工具层先做总字符预算，避免大量长文件名直接把上下文撑肥。
        if projected_chars > MAX_OUTPUT_CHARS and returned_lines:
            output_budget_hit = True
            break
        returned_lines.append(line)
        current_chars = projected_chars

    truncated = len(lines) > len(returned_lines)
    header_lines = [
        f"ROOT: {raw_path}",
        f"TOTAL_ENTRIES: {len(lines)}",
        f"RETURNED_ENTRIES: {len(returned_lines)}",
        f"TRUNCATED: {'yes' if truncated else 'no'}",
        f"OMITTED_INTERNAL_ENTRIES: {omitted_internal_entries}",
        f"OUTPUT_BUDGET_HIT: {'yes' if output_budget_hit else 'no'}",
        "",
    ]

    # 第九步：把结果拼成一个多行字符串返回。
    # 这里先做一次工具自身限额，避免目录很大时原始输出直接失控。
    return ToolResult(
        ok=True,
        output="\n".join(header_lines + returned_lines),
        meta={
            "search_root": raw_path,
            "total_entries": len(lines),
            "returned_entries": len(returned_lines),
            "truncated": truncated,
            "omitted_internal_entries": omitted_internal_entries,
            "output_budget_hit": output_budget_hit,
        },
    )


def _to_workspace_relative_path(target_path: Path, cwd: str) -> str:
    """把绝对路径转成工作区相对路径，便于做内部目录识别。"""
    workspace_root = Path(cwd).resolve()
    try:
        relative_path = target_path.resolve().relative_to(workspace_root)
    except ValueError:
        return target_path.resolve().as_posix()
    return relative_path.as_posix()


def _match_blocked_internal_path(relative_path: str) -> str | None:
    """判断当前列目录请求是否落在内部上下文目录下。"""
    normalized = relative_path.replace("\\", "/").lstrip("./")
    if not normalized:
        return None

    first_part = normalized.split("/", 1)[0].lower()
    if first_part in _BLOCKED_INTERNAL_DIRS:
        return "这是内部上下文目录，默认不作为分析入口。"
    return None


def _is_internal_context_name(name: str) -> bool:
    """判断目录项名字是否属于内部上下文产物。"""
    return name.strip().lower() in _BLOCKED_INTERNAL_DIRS


def _clip_entry_line(text: str) -> str:
    """裁剪过长目录项，避免单个超长文件名污染整体输出。"""
    if len(text) <= MAX_ENTRY_LINE_CHARS:
        return text
    keep = max(40, MAX_ENTRY_LINE_CHARS - 24)
    return f"{text[:keep]} ...[文件名过长已截断]"

# 第十步：把上面的校验函数和执行函数组装成一个正式工具定义
# 后面 ToolRegistry 注册的就是这个对象
list_files_tool = ToolDefinition(
    name="list_files",
    description="列出指定目录下的文件和文件夹",
    validator=_validate,
    runner=_run, # type: ignore
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要列出的目录路径，默认当前目录",
            },
            "max_entries": {
                "type": "integer",
                "description": f"最多返回多少条目录项，默认 {DEFAULT_MAX_ENTRIES}，最大 {MAX_MAX_ENTRIES}。",
            }
        },
        "required": [],
    },
)
