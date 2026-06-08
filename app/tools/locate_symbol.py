from __future__ import annotations

import re
from typing import Any

"""符号定位工具，用于快速定位指定标识符的定义位置。"""

from app.agent.tooling import ToolDefinition
from app.tools._code_nav_common import (
    build_function_signature,
    iter_python_files,
    parse_python_symbols,
    read_text_file,
    resolve_safe_path,
    to_relative_display,
)
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, Any]:
    """
    校验 locate_symbol 的输入。

    symbol:
        必填，要定位的符号名。
    path:
        可选，查找范围，默认当前工作目录。
    max_references:
        可选，最多展示多少条引用，避免输出过长。
    """

    if not isinstance(input_data, dict):
        raise ValueError("locate_symbol 输入必须是字典。")

    symbol = input_data.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol 必须是非空字符串")

    path = input_data.get("path", ".")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    max_references = input_data.get("max_references", 20)
    if not isinstance(max_references, int) or max_references <= 0:
        raise ValueError("max_references 必须是正整数")

    return {
        "symbol": symbol.strip(),
        "path": path.strip(),
        "max_references": max_references,
    }


def _build_symbol_context(file_path, symbol_name: str) -> list[str]:
    """
    为命中的定义构建一小段结构化上下文。

    这里不直接塞整文件内容，而是给出：
    - 这是 class/function/method/variable 的哪一种
    - 所在位置
    - 如果是函数/方法，补一份签名
    """

    source = read_text_file(file_path)
    try:
        import ast

        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    contexts: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == symbol_name:
            contexts.append(f"class {node.name} @L{node.lineno}")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    contexts.append(f"  - method {child.name}{build_function_signature(child)} @L{child.lineno}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol_name:
            contexts.append(f"function {node.name}{build_function_signature(node)} @L{node.lineno}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == symbol_name:
                    contexts.append(f"variable {symbol_name} @L{target.lineno}")
        elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == symbol_name:
            contexts.append(f"variable {symbol_name} @L{node.target.lineno}")

    return contexts


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """
    组合“找定义 + 看周边结构 + 找引用”三步，直接返回定位结果。

    这个工具适合回答：
    - 某个函数/类定义在哪里
    - 这个符号在什么文件里
    - 它周边结构是什么
    - 大概有哪些引用点
    """

    resolved = resolve_safe_path(validated_input["path"], context.cwd)
    if not resolved.abs_path.exists():
        return ToolResult(ok=False, output=f"路径不存在：{validated_input['path']}")

    python_files = iter_python_files(resolved.abs_path)
    if not python_files:
        return ToolResult(ok=True, output="没有找到可分析的 Python 文件。")

    symbol = validated_input["symbol"]
    root_for_display = resolved.abs_path if resolved.abs_path.is_dir() else resolved.abs_path.parent
    symbol_lower = symbol.lower()
    definition_lines: list[str] = []
    context_lines: list[str] = []
    reference_lines: list[str] = []
    reference_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")

    for file_path in python_files:
        display_path = to_relative_display(file_path, root_for_display)
        symbols, error = parse_python_symbols(file_path)
        if not error:
            matched_definitions = [
                item
                for item in symbols
                if item.name.lower() == symbol_lower
                and item.kind not in {"import", "import_from"}
            ]
            for item in matched_definitions:
                parent_part = f" parent={item.parent}" if item.parent else ""
                signature_part = f" {item.signature}" if item.signature else ""
                definition_lines.append(
                    f"{display_path}:{item.line}: {item.kind} {item.name}{signature_part}{parent_part}".rstrip()
                )

            if matched_definitions:
                contexts = _build_symbol_context(file_path, symbol)
                if contexts:
                    context_lines.append(f"[{display_path}]")
                    context_lines.extend(contexts[:12])

        for line_number, line in enumerate(read_text_file(file_path).splitlines(), start=1):
            if reference_pattern.search(line):
                reference_lines.append(f"{display_path}:{line_number}: {line.strip()}")

    if not definition_lines and not reference_lines:
        return ToolResult(ok=True, output="没有找到该符号的定义或引用。")

    output_lines = [f"符号: {symbol}", ""]

    output_lines.append("定义位置:")
    output_lines.extend(definition_lines or ["(未找到精确同名定义)"])

    output_lines.append("")
    output_lines.append("定义周边结构:")
    output_lines.extend(context_lines or ["(未生成额外结构摘要)"])

    output_lines.append("")
    output_lines.append("引用位置:")
    if reference_lines:
        output_lines.extend(reference_lines[: validated_input["max_references"]])
        remain = len(reference_lines) - validated_input["max_references"]
        if remain > 0:
            output_lines.append(f"... 其余省略 {remain} 条引用")
    else:
        output_lines.append("(未找到引用)")

    return ToolResult(ok=True, output="\n".join(output_lines))


locate_symbol_tool = ToolDefinition(
    name="locate_symbol",
    description="组合查找符号定义、周边结构和引用位置，用于快速定位改动点。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "要定位的符号名。",
            },
            "path": {
                "type": "string",
                "description": "可选，查找范围，默认当前工作目录。",
            },
            "max_references": {
                "type": "integer",
                "description": "最多展示多少条引用，默认 20。",
            },
        },
        "required": ["symbol"],
    },
)
