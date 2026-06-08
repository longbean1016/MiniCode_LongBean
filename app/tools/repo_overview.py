from __future__ import annotations

from collections import Counter
from typing import Any

"""仓库概览工具，用于生成项目目录和核心文件的整体摘要。"""

from app.agent.tooling import ToolDefinition
from app.tools._code_nav_common import iter_python_files, resolve_safe_path
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, str]:
    """
    校验 repo_overview 的输入。
    """

    if input_data is None:
        return {"path": "."}

    if not isinstance(input_data, dict):
        raise ValueError("repo_overview 输入必须是字典。")

    path = input_data.get("path", ".")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    return {"path": path.strip()}


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """
    生成仓库或目录级别的结构摘要。
    """

    resolved = resolve_safe_path(validated_input["path"], context.cwd)
    if not resolved.abs_path.exists():
        return ToolResult(ok=False, output=f"路径不存在：{validated_input['path']}")
    if not resolved.abs_path.is_dir():
        return ToolResult(ok=False, output="repo_overview 需要传入目录路径。")

    entries = sorted(list(resolved.abs_path.iterdir()), key=lambda item: item.name.lower())
    dirs = [entry for entry in entries if entry.is_dir()]
    files = [entry for entry in entries if entry.is_file()]
    python_files = iter_python_files(resolved.abs_path)
    test_files = [file_path for file_path in python_files if "test" in file_path.name.lower()]

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

    output_lines = [
        f"目录: {resolved.display_path}",
        f"子目录数: {len(dirs)}",
        f"文件数: {len(files)}",
        f"Python 文件数: {len(python_files)}",
        f"测试文件数: {len(test_files)}",
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
    ]

    return ToolResult(ok=True, output="\n".join(output_lines))


repo_overview_tool = ToolDefinition(
    name="repo_overview",
    description="生成目录或仓库级别的结构摘要，包括顶层目录、文件、Python 文件和入口候选。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要概览的目录路径，默认当前工作目录。",
            },
        },
        "required": [],
    },
)
