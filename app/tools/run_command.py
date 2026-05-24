
import subprocess
from typing import Any

from app.permissions import PermissionManager
from app.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, str]:
    """
    校验 run_command 的输入。
    第一版要求必须传入 command，且 command 必须是非空字符串。
    """
    # 输入必须是字典，后面才能安全取出 command 字段
    if not isinstance(input_data, dict):
        raise ValueError("run_command 输入必须是一个字典，包含 command 字段。")

    # 读取 command 字段
    command = input_data.get("command")

    # command 必须是非空字符串
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command 必须是非空字符串")

    # 返回规范化后的输入
    return {"command": command.strip()}


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """
    执行命令并返回输出结果。
    """
    # 创建权限管理器，用于检查命令是否属于危险命令
    permission_manager = PermissionManager(context.cwd)

    # 取出用户输入的原始命令
    raw_command = validated_input["command"]

    # 权限管理器检查命令是否合法
    permission_manager.ensure_command_allowed(raw_command)

    try:
        # 在工作目录中执行命令，并捕获标准输出和标准错误
        result = subprocess.run(
            raw_command,
            cwd=context.cwd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception as error:
        # 如果命令执行过程本身出错，返回失败结果
        return ToolResult(
            ok=False,
            output=f"命令执行失败: {error}",
        )

    # 拼接标准输出和标准错误，方便调用方查看完整结果
    output_parts: list[str] = []

    if result.stdout:
        output_parts.append(f"标准输出:\n{result.stdout.strip()}")

    if result.stderr:
        output_parts.append(f"标准错误:\n{result.stderr.strip()}")

    output = "\n".join(part for part in output_parts if part)

    # 返回码为 0 说明命令执行成功，否则视为失败
    return ToolResult(
        ok=(result.returncode == 0),
        output=output or "命令执行完成，但没有输出。",
    )


# 把 run_command 工具注册成统一定义
run_command_tool = ToolDefinition(
    name="run_command",
    description="执行一条命令并返回输出结果",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的命令，必须是非空字符串",
            }
        },
        "required": ["command"],
    }
)
