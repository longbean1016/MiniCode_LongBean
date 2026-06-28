from __future__ import annotations

from typing import Any

"""符号查找工具，用于扫描项目中的函数、类和变量定义。"""

from pathlib import Path

from app.agent.tooling import ToolDefinition
from app.tools._code_nav_common import (
    format_symbol_record,
    iter_python_files,
    parse_python_symbols,
    resolve_safe_path,
    to_relative_display,
    workspace_access_denied_result,
)
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, str]:
    """
    校验 find_symbols 的输入。

    path:
        必填，要扫描的目录或 Python 文件。
    symbol_query:
        可选，用于按名称过滤符号。
    """

    if not isinstance(input_data, dict):
        raise ValueError("find_symbols 输入必须是字典，且至少包含 path 字段。")

    path = input_data.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    symbol_query = input_data.get("symbol_query", "")
    if not isinstance(symbol_query, str):
        raise ValueError("symbol_query 必须是字符串")

    return {
        "path": path.strip(),
        "symbol_query": symbol_query.strip(),
    }


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """
    扫描 Python 符号定义。

    当前版本会提取：
    - class
    - function
    - method
    - variable
    - import / import_from
    """

    resolved = resolve_safe_path(
        validated_input["path"],
        context.cwd,
        additional_workspaces={Path(p) for p in context.additional_workspaces},
        permanent_workspaces={Path(p) for p in context.permanent_workspaces},
    )
    if resolved is None:
        return workspace_access_denied_result(validated_input["path"])
    if not resolved.abs_path.exists():
        return ToolResult(ok=False, output=f"路径不存在：{validated_input['path']}")

    python_files = iter_python_files(resolved.abs_path)
    if not python_files:
        return ToolResult(ok=True, output="没有找到可分析的 Python 文件。")

    query = validated_input["symbol_query"].lower()
    root_for_display = resolved.abs_path if resolved.abs_path.is_dir() else resolved.abs_path.parent
    lines: list[str] = []
    parse_errors: list[str] = []

    for file_path in python_files:
        symbols, error = parse_python_symbols(file_path)
        if error:
            parse_errors.append(f"{file_path.name}: {error}")
            continue

        display_path = to_relative_display(file_path, root_for_display)
        for symbol in symbols:
            if query and query not in symbol.name.lower():
                continue
            lines.append(format_symbol_record(display_path, symbol))

    if not lines and not parse_errors:
        return ToolResult(ok=True, output="没有找到匹配的符号。")

    if parse_errors:
        lines.append("")
        lines.append("解析告警：")
        lines.extend(parse_errors[:20])

    return ToolResult(ok=True, output="\n".join(lines).strip())


find_symbols_tool = ToolDefinition(
    name="find_symbols",
    description="扫描 Python 文件中的类、函数、方法、变量和导入符号定义。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要扫描的目录或 Python 文件路径。",
            },
            "symbol_query": {
                "type": "string",
                "description": "可选，按子串过滤符号名。",
            },
        },
        "required": ["path"],
    },
)
