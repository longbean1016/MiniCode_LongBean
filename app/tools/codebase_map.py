from __future__ import annotations

from collections import Counter
from typing import Any

from app.tooling import ToolDefinition
from app.tools._code_nav_common import (
    format_symbol_record,
    iter_python_files,
    parse_python_symbols,
    resolve_safe_path,
    to_relative_display,
)
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, Any]:
    """
    校验 codebase_map 的输入。

    path:
        可选，要概览的目录，默认当前工作目录。
    max_files:
        可选，最多挑多少个 Python 文件做符号摘要，避免输出过长。
    """

    if input_data is None:
        return {"path": ".", "max_files": 8}

    if not isinstance(input_data, dict):
        raise ValueError("codebase_map 输入必须是字典。")

    path = input_data.get("path", ".")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    max_files = input_data.get("max_files", 8)
    if not isinstance(max_files, int) or max_files <= 0:
        raise ValueError("max_files 必须是正整数")

    return {
        "path": path.strip(),
        "max_files": max_files,
    }


def _pick_focus_files(python_files: list, root_path) -> list:
    """
    从所有 Python 文件里选出更值得优先展示的文件。

    这里不是随便截前几个，而是优先挑：
    1. 常见入口文件
    2. 非测试文件
    3. 路径更短、通常更核心的文件
    """

    preferred_names = {
        "main.py",
        "__init__.py",
        "app.py",
        "cli.py",
        "server.py",
        "model.py",
        "config.py",
    }

    def sort_key(file_path) -> tuple[int, int, int, str]:
        relative = to_relative_display(file_path, root_path)
        name_score = 0 if file_path.name in preferred_names else 1
        test_score = 1 if "test" in relative.lower() else 0
        depth_score = len(file_path.parts)
        return (name_score, test_score, depth_score, relative.lower())

    return sorted(python_files, key=sort_key)


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """
    生成“代码库第一层地图”。

    这个工具的目标不是替代 read_file，而是先给模型一份初步导航：
    - 目录里大概有什么
    - Python 文件分布如何
    - 哪些文件像入口
    - 每个重点文件里大致定义了什么
    """

    resolved = resolve_safe_path(validated_input["path"], context.cwd)
    if not resolved.abs_path.exists():
        return ToolResult(ok=False, output=f"路径不存在：{validated_input['path']}")
    if not resolved.abs_path.is_dir():
        return ToolResult(ok=False, output="codebase_map 需要传入目录路径。")

    entries = sorted(list(resolved.abs_path.iterdir()), key=lambda item: item.name.lower())
    dirs = [entry for entry in entries if entry.is_dir()]
    files = [entry for entry in entries if entry.is_file()]
    python_files = iter_python_files(resolved.abs_path)

    suffix_counter = Counter(file_path.suffix or "(无后缀)" for file_path in files)
    entry_candidates: list[str] = []
    for candidate_name in [
        "app/main.py",
        "main.py",
        "manage.py",
        "run.py",
        "README.md",
        "requirements.txt",
        "longbean.toml",
        "pyproject.toml",
    ]:
        if (resolved.abs_path / candidate_name).exists():
            entry_candidates.append(candidate_name)

    lines = [
        f"目录: {resolved.display_path}",
        f"子目录数: {len(dirs)}",
        f"顶层文件数: {len(files)}",
        f"Python 文件总数: {len(python_files)}",
        "",
        "顶层目录:",
        *([entry.name for entry in dirs[:20]] or ["(无)"]),
        "",
        "顶层文件:",
        *([entry.name for entry in files[:20]] or ["(无)"]),
        "",
        "顶层文件后缀统计:",
        *([f"{suffix}: {count}" for suffix, count in sorted(suffix_counter.items())] or ["(无)"]),
        "",
        "入口候选:",
        *(entry_candidates or ["(未识别到常见入口文件)"]),
        "",
        "重点 Python 文件与主要符号:",
    ]

    focus_files = _pick_focus_files(python_files, resolved.abs_path)[: validated_input["max_files"]]
    if not focus_files:
        lines.append("(没有可分析的 Python 文件)")
        return ToolResult(ok=True, output="\n".join(lines))

    for file_path in focus_files:
        display_path = to_relative_display(file_path, resolved.abs_path)
        symbols, error = parse_python_symbols(file_path)
        if error:
            lines.append(f"- {display_path}")
            lines.append(f"  解析失败: {error}")
            continue

        lines.append(f"- {display_path}")
        if not symbols:
            lines.append("  (未提取到主要符号)")
            continue

        for symbol in symbols[:8]:
            formatted = format_symbol_record(display_path, symbol)
            # 这里去掉重复的文件名前缀，避免组合输出过于啰嗦。
            prefix = f"{display_path}:{symbol.line} "
            compact = formatted
            if formatted.startswith(prefix):
                compact = formatted[len(prefix):]
            lines.append(f"  - L{symbol.line} {compact}")

        if len(symbols) > 8:
            lines.append(f"  - ... 省略其余 {len(symbols) - 8} 个符号")

    return ToolResult(ok=True, output="\n".join(lines))


codebase_map_tool = ToolDefinition(
    name="codebase_map",
    description="生成代码库第一层地图，组合目录概览和重点 Python 文件符号摘要。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要概览的目录路径，默认当前工作目录。",
            },
            "max_files": {
                "type": "integer",
                "description": "最多展示多少个重点 Python 文件，默认 8。",
            },
        },
        "required": [],
    },
)
