"""grep 文本搜索工具，支持目录/单文件搜索、glob 过滤和多种输出模式。
   参考 Claude Code GrepTool 语义实现，优先使用 ripgrep，不可用时回退 Python 原生扫描。"""

from pathlib import Path
import subprocess
from typing import Any

from app.agent.permissions import PathAccessStatus, PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# ── 搜索参数默认值（对齐 Claude Code GrepTool） ──
DEFAULT_HEAD_LIMIT = 250          # 默认最多返回行数/文件数
MAX_HEAD_LIMIT = 10_000           # 允许的上限（0 表示不限）
MAX_MATCH_LINE_CHARS = 240        # 单行最大展示字符数
MAX_OUTPUT_CHARS = 16_000         # 输出总字符预算
# 排除的内部目录和 VCS 目录 — 避免把缓存/状态文件当作源码搜索
_BLOCKED_INTERNAL_DIRS = {
    ".cache", "cache", ".sessions", "sessions", ".context_state", "context_state",
    ".git", ".svn", ".hg", ".bzr",
}
# ripgrep 额外排除规则
_RG_EXCLUDE_GLOBS = (
    "!.git/**",
    "!.memory/**",
    "!.qdrant_storage/**",
    "!.pytest_cache/**",
    "!tmp/**",
    "!__pycache__/**",
    "!debug.log",
)


def _validate(input_data: Any) -> dict[str, Any]:
    """校验 grep_files 输入参数（对齐 Claude Code GrepTool input_schema）。

       支持：pattern / path / glob / output_mode / head_limit / offset / -i / -n
    """
    if not isinstance(input_data, dict):
        raise ValueError("grep_files 输入必须是字典，包含 pattern 字段。")

    # ── pattern（必填） ──
    pattern = input_data.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern 必须是非空字符串。")

    # ── path（可选，默认当前目录）──
    raw_path = input_data.get("path")
    if raw_path is not None and not isinstance(raw_path, str):
        raise ValueError("path 必须是字符串。")
    path = raw_path.strip() if isinstance(raw_path, str) and raw_path.strip() else "."

    # ── glob 文件过滤（可选）──
    glob_filter = input_data.get("glob")
    if glob_filter is not None and not isinstance(glob_filter, str):
        raise ValueError("glob 必须是字符串。")

    # ── output_mode: content | files_with_matches | count ──
    output_mode = input_data.get("output_mode", "files_with_matches")
    if output_mode not in ("content", "files_with_matches", "count"):
        raise ValueError("output_mode 必须是 'content'、'files_with_matches' 或 'count'。")

    # ── head_limit（可选，默认 250，0=不限）──
    raw_head = input_data.get("head_limit")
    head_limit = DEFAULT_HEAD_LIMIT
    if raw_head is not None:
        try:
            head_limit = int(raw_head)
        except (TypeError, ValueError):
            pass
    if head_limit < 0 or head_limit > MAX_HEAD_LIMIT:
        raise ValueError(f"head_limit 必须在 0 到 {MAX_HEAD_LIMIT} 之间。")

    # ── offset（可选，默认 0）──
    raw_offset = input_data.get("offset", 0)
    try:
        offset = int(raw_offset)
    except (TypeError, ValueError):
        offset = 0
    if offset < 0:
        raise ValueError("offset 必须 >= 0。")

    # ── -i: 大小写不敏感 ──
    case_insensitive = bool(input_data.get("-i", False))

    # ── -n: 显示行号（content 模式下默认 true）──
    show_line_numbers = bool(input_data.get("-n", True))

    return {
        "pattern": pattern.strip(),
        "path": path,
        "glob": glob_filter.strip() if glob_filter else None,
        "output_mode": output_mode,
        "head_limit": head_limit,
        "offset": offset,
        "case_insensitive": case_insensitive,
        "show_line_numbers": show_line_numbers,
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """在目录/文件内搜索匹配文本，支持 content / files_with_matches / count 三种输出模式。"""
    permission_manager = PermissionManager(
        context.cwd,
        additional_workspaces={Path(p) for p in context.additional_workspaces},
        permanent_workspaces={Path(p) for p in context.permanent_workspaces},
    )

    pattern = validated_input["pattern"]
    raw_path = validated_input["path"]
    glob_filter = validated_input["glob"]
    output_mode = validated_input["output_mode"]
    head_limit = validated_input["head_limit"]
    offset = validated_input["offset"]
    case_insensitive = validated_input["case_insensitive"]
    show_line_numbers = validated_input["show_line_numbers"]

    # 路径权限检查
    check = permission_manager.check_path_access(raw_path)
    if check.status == PathAccessStatus.OUTSIDE_WORKSPACE:
        return ToolResult(
            ok=False,
            output=f"目标路径不在工作目录范围内：{raw_path}",
            error="WORKSPACE_ACCESS_REQUIRED",
            meta={"path": raw_path, "action_key": f"workspace::{raw_path}", "reason": check.message},
        )
    target_path = check.resolved_path

    if not target_path.exists():
        return ToolResult(ok=False, output=f"路径不存在：{raw_path}")

    # ── 支持单文件搜索（Claude Code GrepTool 的新增能力）──
    if target_path.is_file():
        return _search_single_file(
            pattern=pattern, file_path=target_path, raw_path=raw_path,
            output_mode=output_mode, head_limit=head_limit, offset=offset,
            case_insensitive=case_insensitive, show_line_numbers=show_line_numbers,
        )

    if not target_path.is_dir():
        return ToolResult(ok=False, output=f"目标不是目录也不是文件：{raw_path}",
                          error="SEARCH_INVALID_PATH")

    # ── 目录搜索：优先走 ripgrep ──
    rg_result = _run_with_rg(
        pattern=pattern, raw_path=raw_path, target_path=target_path,
        cwd=context.cwd, output_mode=output_mode, head_limit=head_limit,
        offset=offset, case_insensitive=case_insensitive, glob_filter=glob_filter,
        show_line_numbers=show_line_numbers,
    )
    if rg_result is not None:
        return rg_result

    # ── ripgrep 不可用时用 Python 回退 ──
    return _run_with_python_scan(
        pattern=pattern, raw_path=raw_path, target_path=target_path,
        output_mode=output_mode, head_limit=head_limit, offset=offset,
        case_insensitive=case_insensitive, show_line_numbers=show_line_numbers,
    )


# ── 单文件搜索 ──

def _search_single_file(
    *, pattern: str, file_path: Path, raw_path: str, output_mode: str,
    head_limit: int, offset: int, case_insensitive: bool, show_line_numbers: bool,
) -> ToolResult:
    """对单个文件做内容搜索（Claude Code GrepTool 支持 path 指向文件）。"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return ToolResult(ok=False, output=f"无法读取文件：{raw_path} — {exc}",
                          error="FILE_READ_FAILED")

    lines = content.splitlines()
    matched_lines: list[str] = []
    search_pattern = pattern.lower() if case_insensitive else pattern

    for line_num, line in enumerate(lines, start=1):
        target = line.lower() if case_insensitive else line
        if search_pattern in target:
            clipped, _ = _clip_match_line(line)
            prefix = f"{raw_path}:{line_num}: " if show_line_numbers else f"{raw_path}: "
            matched_lines.append(f"{prefix}{clipped}")

    return _format_final_result(
        pattern=pattern, raw_path=raw_path, matched_lines=matched_lines,
        output_mode=output_mode, head_limit=head_limit, offset=offset,
        search_engine="python",
    )


# ── ripgrep 优先搜索 ──

def _run_with_rg(
    *, pattern: str, raw_path: str, target_path: Path, cwd: str,
    output_mode: str, head_limit: int, offset: int,
    case_insensitive: bool, glob_filter: str | None, show_line_numbers: bool,
) -> ToolResult | None:
    """使用 ripgrep 搜索；不可用时返回 None 让调用方回退 Python 扫描。"""
    args = ["rg", "--hidden", "--no-heading", "--color", "never", "--no-messages"]

    # 排除 VCS 和内部上下文目录
    for vcs_dir in [".git", ".svn", ".hg"]:
        args.extend(["--glob", f"!{vcs_dir}"])
    for exc in _RG_EXCLUDE_GLOBS:
        args.extend(["--glob", exc])

    # 大小写不敏感
    if case_insensitive:
        args.append("-i")

    # 输出模式（对齐 Claude Code GrepTool）
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")
    elif output_mode == "content" and show_line_numbers:
        args.append("-n")

    # glob 过滤
    if glob_filter:
        for g in glob_filter.replace(",", " ").split():
            if g.strip():
                args.extend(["--glob", g.strip()])

    # 限制列宽，避免 base64/压缩 js 污染结果
    args.extend(["--max-columns", "500"])

    # 确保 pattern 不以 - 开头导致 rg 把它当成选项
    if pattern.startswith("-"):
        args.extend(["-e", pattern, raw_path])
    else:
        args.extend(["--", pattern, raw_path])

    try:
        result = subprocess.run(
            args, cwd=cwd, text=True, capture_output=True,
            timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    # rg: 0=有匹配, 1=无匹配, 2+=错误
    if result.returncode not in {0, 1}:
        return None

    matched_lines: list[str] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        line, _ = _normalize_rg_line(raw_line, target_path, cwd)
        matched_lines.append(line)

    return _format_final_result(
        pattern=pattern, raw_path=raw_path, matched_lines=matched_lines,
        output_mode=output_mode, head_limit=head_limit, offset=offset,
        search_engine="rg",
    )


# ── Python 回退扫描 ──

def _run_with_python_scan(
    *, pattern: str, raw_path: str, target_path: Path,
    output_mode: str, head_limit: int, offset: int,
    case_insensitive: bool, show_line_numbers: bool,
) -> ToolResult:
    """Python 原生文件扫描回退方案（rg 不可用时使用）。"""
    matched_lines: list[str] = []
    search_pattern = pattern.lower() if case_insensitive else pattern

    for file_path in target_path.rglob("*"):
        if not file_path.is_file():
            continue
        # 跳过内部目录下的文件
        if any(p.lower() in _BLOCKED_INTERNAL_DIRS for p in file_path.parts):
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        rel_path = str(file_path.relative_to(target_path))
        for line_num, line in enumerate(content.splitlines(), start=1):
            target = line.lower() if case_insensitive else line
            if search_pattern in target:
                clipped, _ = _clip_match_line(line)
                prefix = f"{rel_path}:{line_num}: " if show_line_numbers else f"{rel_path}: "
                matched_lines.append(f"{prefix}{clipped}")

    return _format_final_result(
        pattern=pattern, raw_path=raw_path, matched_lines=matched_lines,
        output_mode=output_mode, head_limit=head_limit, offset=offset,
        search_engine="python",
    )


# ── 结果格式化 ──

def _format_final_result(
    *, pattern: str, raw_path: str, matched_lines: list[str],
    output_mode: str, head_limit: int, offset: int, search_engine: str,
) -> ToolResult:
    """统一格式化搜索结果，应用 head_limit + offset 截断 + 字符预算保护。"""
    total_matches = len(matched_lines)

    # 统计匹配文件数（用于 files_with_matches 和 count 模式）
    files_set: set[str] = set()
    for line in matched_lines:
        if ":" in line:
            files_set.add(line.split(":", 1)[0])

    # 应用 offset
    if offset > 0 and offset < len(matched_lines):
        matched_lines = matched_lines[offset:]

    # 应用 head_limit（0 = 不限）
    limited = False
    if head_limit > 0 and len(matched_lines) > head_limit:
        matched_lines = matched_lines[:head_limit]
        limited = True

    # 构建输出头
    lines_after = len(matched_lines)
    header_parts = [
        f"PATTERN: {pattern}",
        f"ROOT: {raw_path}",
        f"MODE: {output_mode}",
        f"TOTAL_MATCHES: {total_matches}",
        f"RETURNED: {lines_after}",
        f"ENGINE: {search_engine}",
    ]
    if limited:
        header_parts.append(f"(head_limit={head_limit}, offset={offset})")
    header = "\n".join(header_parts) + "\n\n"

    # 字符预算保护
    output_chars = len(header)
    result_lines: list[str] = []
    truncated = limited
    for line in matched_lines:
        proj = output_chars + len(line) + 1
        if proj > MAX_OUTPUT_CHARS and result_lines:
            truncated = True
            break
        result_lines.append(line)
        output_chars = proj

    output = header + "\n".join(result_lines)
    if truncated:
        output += "\n\n(结果已截断，缩小搜索范围可获取更多结果)"

    return ToolResult(
        ok=True,
        output=output,
        meta={
            "pattern": pattern,
            "search_root": raw_path,
            "total_matches": total_matches,
            "returned_matches": lines_after,
            "truncated": truncated,
            "output_mode": output_mode,
            "num_files": len(files_set),
            "search_engine": search_engine,
        },
    )


# ── 辅助函数 ──

def _normalize_rg_line(raw_line: str, target_path: Path, cwd: str) -> tuple[str, bool]:
    """把 ripgrep 输出的绝对路径转为相对于搜索根目录的路径。"""
    if ":" not in raw_line:
        return raw_line, False
    path_part, rest = raw_line.split(":", 1)
    rg_path = Path(path_part)
    if not rg_path.is_absolute():
        rg_path = Path(cwd) / rg_path
    try:
        display = rg_path.resolve().relative_to(target_path.resolve()).as_posix()
    except ValueError:
        display = path_part.replace("\\", "/")
    return f"{display}:{rest}", False


def _clip_match_line(line: str) -> tuple[str, bool]:
    """裁剪超长命中行，避免单行大日志直接撑爆搜索结果。"""
    if len(line) <= MAX_MATCH_LINE_CHARS:
        return line, False
    keep = max(80, MAX_MATCH_LINE_CHARS - 18)
    return f"{line[:keep]} ...[单行已截断]", True


# ── 注册工具 ──
grep_files_tool = ToolDefinition(
    name="grep_files",
    description=(
        "在文件内容中搜索正则表达式匹配。"
        "支持目录和单文件、glob 文件过滤、三种输出模式（content/files_with_matches/count），"
        "支持 head_limit + offset 分页和 -i 大小写不敏感。"
    ),
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "要搜索的正则表达式模式。",
            },
            "path": {
                "type": "string",
                "description": "要搜索的文件或目录路径。默认当前工作目录。支持指向单个文件。",
            },
            "glob": {
                "type": "string",
                "description": "glob 模式过滤文件名，例如 '*.py' 或 '*.{ts,tsx}'（映射为 rg --glob）。",
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "输出模式：content 显示匹配行，files_with_matches 只显示文件路径，count 显示匹配计数。默认 files_with_matches。",
            },
            "head_limit": {
                "type": "integer",
                "description": f"最多返回多少条结果，相当于 '| head -N'。默认 {DEFAULT_HEAD_LIMIT}，设为 0 表示不限。",
            },
            "offset": {
                "type": "integer",
                "description": "跳过前 N 条结果，相当于 '| tail -n +N'。默认 0。",
            },
            "-i": {
                "type": "boolean",
                "description": "大小写不敏感搜索（rg -i）。默认 false。",
            },
            "-n": {
                "type": "boolean",
                "description": "显示行号（rg -n）。仅在 output_mode=content 时有效。默认 true。",
            },
        },
        "required": ["pattern"],
    },
)
