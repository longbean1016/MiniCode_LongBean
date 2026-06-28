from pathlib import Path
from typing import Any

from app.agent.permissions import PathAccessStatus, PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, Any]:
    """
    校验 make_dirs 的输入，并转成统一结构。
    """

    # 输入必须是字典，后面才能安全读取字段
    if not isinstance(input_data, dict):
        raise ValueError("make_dirs 输入必须是一个字典，包含 path 字段。")

    # 读取 path 字段
    path = input_data.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")

    # parents 表示是否递归创建父目录，默认开启
    parents = input_data.get("parents", True)
    if not isinstance(parents, bool):
        raise ValueError("parents 必须是布尔值")

    # exist_ok 表示目录已存在时是否视为成功，默认开启
    exist_ok = input_data.get("exist_ok", True)
    if not isinstance(exist_ok, bool):
        raise ValueError("exist_ok 必须是布尔值")

    # 返回规范化后的输入
    return {
        "path": path.strip(),
        "parents": parents,
        "exist_ok": exist_ok,
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """
    创建目标目录。
    """

    # 用当前工作目录作为权限边界，包含额外工作目录
    permission_manager = PermissionManager(
        context.cwd,
        additional_workspaces={Path(p) for p in context.additional_workspaces},
        permanent_workspaces={Path(p) for p in context.permanent_workspaces},
    )

    # 取出校验后的输入
    raw_path = validated_input["path"]
    parents = validated_input["parents"]
    exist_ok = validated_input["exist_ok"]

    # 检查路径是否越界，并解析成绝对路径
    check = permission_manager.check_path_access(raw_path)
    if check.status == PathAccessStatus.OUTSIDE_WORKSPACE:
        return ToolResult(
            ok=False,
            output=f"目标路径不在工作目录范围内：{raw_path}",
            error="WORKSPACE_ACCESS_REQUIRED",
            meta={
                "path": raw_path,
                "resolved_path": str(check.resolved_path) if check.resolved_path else raw_path,
                "action_key": f"workspace::{raw_path}",
                "reason": check.message,
            },
        )
    target_path = check.resolved_path

    # 如果目标已存在且不是目录，直接返回失败
    if target_path.exists() and not target_path.is_dir():
        return ToolResult(
            ok=False,
            output=f"目标已存在且不是目录，无法创建目录：{raw_path}",
            error="TARGET_NOT_DIRECTORY",
            meta={
                "path": raw_path,
            },
        )

    try:
        # 按参数决定是否递归创建，以及已存在时是否报错
        target_path.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as error:
        # 创建失败时返回统一错误结果
        return ToolResult(
            ok=False,
            output=f"创建目录失败：{error}",
            error="MAKE_DIRS_FAILED",
            meta={
                "path": raw_path,
                "parents": parents,
                "exist_ok": exist_ok,
            },
        )

    # 返回成功结果
    return ToolResult(
        ok=True,
        output=f"目录创建成功：{raw_path}",
        meta={
            "path": raw_path,
            "parents": parents,
            "exist_ok": exist_ok,
        },
    )


make_dirs_tool = ToolDefinition(
    name="make_dirs",
    description="创建指定目录，可选择递归创建父目录，并允许目录已存在",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要创建的目录路径，必须在工作目录内",
            },
            "parents": {
                "type": "boolean",
                "description": "是否递归创建父目录，默认 true",
            },
            "exist_ok": {
                "type": "boolean",
                "description": "目录已存在时是否视为成功，默认 true",
            },
        },
        "required": ["path"],
    },
)