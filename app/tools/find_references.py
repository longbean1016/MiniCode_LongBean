from __future__ import annotations

import re
from typing import Any

"""引用查找工具，用于搜索符号在项目中的调用或引用位置。"""

from app.tooling import ToolDefinition
from app.tools._code_nav_common import (
    iter_python_files,
    read_text_file,
    resolve_safe_path,
    to_relative_display,
)
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, str]:
    """
    校验 find_references 的输入。

    symbol:
        必填，要查找的标识符名称。
    path:
        可选，查找范围，默认当前目录。
    """

    if not isinstance(input_data, dict):
        raise ValueError("find_references 输入必须是字典。")

    symbol = input_data.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol 必须是非空字符串")

    path = input_data.get("path", ".")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    return {
        "symbol": symbol.strip(),
        "path": path.strip(),
    }


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """
    查找某个 Python 标识符的引用位置。

    这里先使用“带标识符边界的文本匹配”，
    这样比普通子串搜索更稳一些，能够减少误报。
    """

    resolved = resolve_safe_path(validated_input["path"], context.cwd)
    if not resolved.abs_path.exists():
        return ToolResult(ok=False, output=f"路径不存在：{validated_input['path']}")

    python_files = iter_python_files(resolved.abs_path)
    if not python_files:
        return ToolResult(ok=True, output="没有找到可分析的 Python 文件。")

    symbol = validated_input["symbol"]
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")
    root_for_display = resolved.abs_path if resolved.abs_path.is_dir() else resolved.abs_path.parent
    matches: list[str] = []

    for file_path in python_files:
        display_path = to_relative_display(file_path, root_for_display)
        for line_number, line in enumerate(read_text_file(file_path).splitlines(), start=1):
            if pattern.search(line):
                matches.append(f"{display_path}:{line_number}: {line.strip()}")

    if not matches:
        return ToolResult(ok=True, output="没有找到匹配的引用。")

    return ToolResult(ok=True, output="\n".join(matches))


find_references_tool = ToolDefinition(
    name="find_references",
    description="在 Python 文件中查找某个符号名称的引用位置。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "要查找引用的符号名。",
            },
            "path": {
                "type": "string",
                "description": "可选，查找范围，默认当前工作目录。",
            },
        },
        "required": ["symbol"],
    },
)
