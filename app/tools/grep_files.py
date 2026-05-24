
from typing import Any

from app.permissions import PermissionManager
from app.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, str]:
    """
    校验 grep_files 的输入。
    第一版要求必须有 pattern，path 可以省略，默认当前目录。
    """
    # 输入必须是字典，后面才能安全取出 pattern 和 path 字段
    if not isinstance(input_data, dict):
        raise ValueError("grep_files 输入必须是一个字典，包含 pattern 和 path 字段。")

    # pattern 是必须的，用来表示要搜索的文本
    pattern = input_data.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern 必须是非空字符串")

    # path 是可选的，默认值是当前目录 "."
    path = input_data.get("path", ".")
    if not isinstance(path, str):
        raise ValueError("path 必须是字符串")

    # 返回规范化后的输入
    return {
        "pattern": pattern.strip(),
        "path": path.strip(),
    }


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """
    在指定目录下递归搜索文本内容。
    """
    # 创建权限管理器，限制工具只能在工作目录内访问文件
    permission_manager = PermissionManager(context.cwd)

    # 取出搜索关键字和目标目录
    pattern = validated_input["pattern"]
    raw_path = validated_input["path"]

    # 检查路径是否合法，并解析成绝对路径
    target_path = permission_manager.ensure_path_access(raw_path)

    # 路径不存在时，直接返回失败结果
    if not target_path.exists():
        return ToolResult(
            ok=False,
            output=f"路径不存在：{raw_path}",
        )

    # 第一版只允许搜索目录，不处理单文件搜索
    if not target_path.is_dir():
        return ToolResult(
            ok=False,
            output=f"目标不是目录：{raw_path}",
        )

    matches: list[str] = []

    # 递归搜索目录下的所有文件
    for file_path in target_path.rglob("*"):
        # 跳过目录，只处理文件
        if not file_path.is_file():
            continue

        # 读取文件内容，第一版先固定使用 utf-8 编码
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            # 读取文件失败时，先跳过该文件，避免中断整个搜索
            continue

        # 按行遍历，查找包含搜索关键字的行
        for line_num, line in enumerate(content.splitlines(), start=1):
            if pattern in line:
                # 找到匹配时，记录相对路径、行号和行内容
                relative_path = file_path.relative_to(target_path)
                matches.append(f"{relative_path}:{line_num}: {line}")

    # 如果没有找到任何匹配行，返回一个明确提示
    if not matches:
        return ToolResult(
            ok=True,
            output="没有找到匹配的内容",
        )

    # 第一版先限制返回前 200 条，避免输出过长
    return ToolResult(
        ok=True,
        output="\n".join(matches[:200]),
    )


# 把 grep_files 工具注册成统一定义
grep_files_tool = ToolDefinition(
    name="grep_files",
    description="在指定目录中搜索包含目标文本的文件内容",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "要搜索的文本内容，必填",
            },
            "path": {
                "type": "string",
                "description": "要搜索的目录路径，选填，默认为当前目录",
            },
        },
        "required": ["pattern"],
    }
)
