from __future__ import annotations

import ast
from typing import Any

"""AST 信息工具，输出源码的结构化语法树摘要。"""

from pathlib import Path

from app.agent.tooling import ToolDefinition
from app.tools._code_nav_common import (
    build_function_signature,
    read_text_file,
    resolve_safe_path,
    shorten_doc,
    workspace_access_denied_result,
)
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, str]:
    """
    校验 get_ast_info 的输入。
    """

    if not isinstance(input_data, dict):
        raise ValueError("get_ast_info 输入必须是字典。")

    path = input_data.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    return {"path": path.strip()}


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """
    返回单个 Python 文件的 AST 结构摘要。
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
        return ToolResult(ok=False, output=f"文件不存在：{validated_input['path']}")
    if not resolved.abs_path.is_file():
        return ToolResult(ok=False, output="get_ast_info 只支持单个 Python 文件。")
    if resolved.abs_path.suffix != ".py":
        return ToolResult(ok=False, output="get_ast_info 目前只支持 .py 文件。")

    source = read_text_file(resolved.abs_path)
    try:
        tree = ast.parse(source, filename=str(resolved.abs_path))
    except SyntaxError as error:
        return ToolResult(ok=False, output=f"语法解析失败：{error}")

    imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []
    variables: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for alias in node.names:
                imports.append(f"{module_name}.{alias.asname or alias.name}".strip("."))
        elif isinstance(node, ast.ClassDef):
            class_header = f"class {node.name} @L{node.lineno}"
            class_doc = shorten_doc(ast.get_docstring(node))
            if class_doc:
                class_header += f" - {class_doc}"

            method_lines: list[str] = [class_header]
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_lines.append(
                        f"  - method {child.name}{build_function_signature(child)} @L{child.lineno}"
                    )
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            method_lines.append(f"  - class_variable {target.id} @L{target.lineno}")
                elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    method_lines.append(f"  - class_variable {child.target.id} @L{child.target.lineno}")
            classes.append("\n".join(method_lines))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_line = f"function {node.name}{build_function_signature(node)} @L{node.lineno}"
            function_doc = shorten_doc(ast.get_docstring(node))
            if function_doc:
                function_line += f" - {function_doc}"
            functions.append(function_line)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.append(f"variable {target.id} @L{target.lineno}")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            variables.append(f"variable {node.target.id} @L{node.target.lineno}")

    output_lines = [
        f"文件: {resolved.display_path}",
        f"总行数: {len(source.splitlines())}",
        f"顶层节点数: {len(tree.body)}",
    ]

    module_doc = shorten_doc(ast.get_docstring(tree))
    if module_doc:
        output_lines.append(f"模块文档: {module_doc}")

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
            "",
            "模块级变量:",
            *(variables or ["(无)"]),
        ]
    )

    return ToolResult(ok=True, output="\n".join(output_lines))


get_ast_info_tool = ToolDefinition(
    name="get_ast_info",
    description="返回单个 Python 文件的 AST 结构摘要，包括导入、类、函数和变量。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要分析的 Python 文件路径。",
            },
        },
        "required": ["path"],
    },
)
