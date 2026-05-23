
from typing import Any

from app.permissions import PermissionManager
from app.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, str]:
    """
    校验 read_file 的输入。
    第一版要求必须传入 path，且 path 必须是非空字符串。
    """
    # 输入必须是字典，后面才能安全取出 path 字段
    if not isinstance(input_data, dict):
        raise ValueError("read_file 输入必须是一个字典，包含 path 字段。")

    # 读取 path 字段
    path = input_data.get("path")

    # path 字段必须存在，且必须是非空字符串
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    # 返回规范化后的输入
    return {"path": path.strip()}


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """
    读取目标文件内容。
    """
    # 创建权限管理器，限制工具只能在工作目录内访问文件
    permission_manager = PermissionManager(context.cwd)

    # 取出用户输入的原始路径
    raw_path = validated_input["path"]

    # 检查路径是否越界，并解析成绝对路径
    target_path = permission_manager.ensure_path_access(raw_path)

    # 路径不存在时，直接返回失败结果
    if not target_path.exists():
        return ToolResult(
            ok=False,
            output=f"文件不存在：{raw_path}",
        )

    # 如果目标不是文件，也直接返回失败结果
    if not target_path.is_file():
        return ToolResult(
            ok=False,
            output=f"目标不是文件：{raw_path}",
        )

    # 读取文件内容，第一版先固定使用 utf-8 编码
    content = target_path.read_text(encoding="utf-8")

    # 把读取结果作为成功输出返回
    return ToolResult(
        ok=True,
        output=content,
    )


# 把校验函数和执行函数组装成为正式工具
read_file_tool = ToolDefinition(
    name="read_file",
    description="读取指定文件的内容",
    validator=_validate,
    runner=_run,
)
