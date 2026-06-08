from __future__ import annotations

import ast
from typing import Any

"""文件概览工具，提取单个源码文件的导入、函数和类结构。"""

from app.agent.tooling import ToolDefinition
from app.tools._code_nav_common import (
    build_function_signature,
    read_text_file,
    resolve_safe_path,
    shorten_doc,
)
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, str]:
    """
    校验 file_overview 的输入。
    """

    if not isinstance(input_data, dict):
        raise ValueError("file_overview 输入必须是字典。")

    path = input_data.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    return {"path": path.strip()}


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """
    生成单文件结构摘要。

    如果是 Python 文件，就基于 AST 给出结构信息。
    如果不是 Python 文件，就返回基础文本统计。
    """

    resolved = resolve_safe_path(validated_input["path"], context.cwd)
    if not resolved.abs_path.exists():
        return ToolResult(ok=False, output=f"文件不存在：{validated_input['path']}")
    if not resolved.abs_path.is_file():
        return ToolResult(ok=False, output="file_overview 需要传入文件路径。")

    content = read_text_file(resolved.abs_path)
    raw_lines = content.splitlines()
    output_lines = [
        f"文件: {resolved.display_path}",
        f"扩展名: {resolved.abs_path.suffix or '(无后缀)'}",
        f"总行数: {len(raw_lines)}",
        f"总字符数: {len(content)}",
    ]

    if resolved.abs_path.suffix != ".py":
        first_non_empty = next((line.strip() for line in raw_lines if line.strip()), "")
        output_lines.append(f"首个非空行: {first_non_empty or '(无)'}")
        return ToolResult(ok=True, output="\n".join(output_lines))

    try:
        tree = ast.parse(content, filename=str(resolved.abs_path))
    except SyntaxError as error:
        output_lines.append(f"语法解析失败: {error}")
        return ToolResult(ok=False, output="\n".join(output_lines))

    module_doc = shorten_doc(ast.get_docstring(tree))
    if module_doc:
        output_lines.append(f"模块文档: {module_doc}")

    imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for alias in node.names:
                imports.append(f"{module_name}.{alias.asname or alias.name}".strip("."))
        elif isinstance(node, ast.ClassDef):
            method_count = sum(
                1 for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            classes.append(f"{node.name} @L{node.lineno} methods={method_count}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(f"{node.name}{build_function_signature(node)} @L{node.lineno}")

    output_lines.extend(
        [
            "",
            "导入:",
            *(imports or ["(无)"]),
            "",
            "类:",
            *(classes or ["(无)"]),
            "",
            "函数:",
            *(functions or ["(无)"]),
        ]
    )

    return ToolResult(ok=True, output="\n".join(output_lines))


file_overview_tool = ToolDefinition(
    name="file_overview",
    description="生成单个文件的结构摘要；Python 文件返回类、函数、导入等信息。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要概览的文件路径。",
            },
        },
        "required": ["path"],
    },
)
