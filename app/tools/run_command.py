
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
    """执行命令并返回结果，包含权限检查、超时控制和输出截断。"""

    # 创建权限管理器，用于检查命令是否属于危险命令
    permission_manager = PermissionManager(context.cwd)

    # 取出用户输入的原始命令
    raw_command = validated_input["command"]

     # 1) 命令安全检查（危险命令黑名单）
    permission_manager.ensure_command_allowed(raw_command)

    try:
        # 2) 执行命令（加超时，防止卡死）
        result = subprocess.run(
            raw_command,
            cwd=context.cwd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # 编码异常时替换，避免抛错中断
            timeout=permission_manager.get_command_timeout(),
        )
    except subprocess.TimeoutExpired:
        # 3) 超时错误统一返回
        return ToolResult(
            ok=False,
            output=f"命令执行超时：超过 {permission_manager.get_command_timeout()} 秒",
        )
    except Exception as error:
        # 其他执行异常统一兜底
        return ToolResult(
            ok=False,
            output=f"命令执行失败: {error}",
        )

    # 组装 stdout/stderr
    output_parts: list[str] = []
    if result.stdout:
        output_parts.append(f"标准输出:\n{result.stdout.strip()}")
    if result.stderr:
        output_parts.append(f"标准错误:\n{result.stderr.strip()}")

    output = "\n\n".join(output_parts).strip()
    if not output:
        output = "命令执行完成，但没有输出。"

    # 4) 截断超长输出，防止污染上下文
    output = permission_manager.truncate_output(output)

    return ToolResult(
        ok=(result.returncode == 0),
        output=output,
    )


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
                "description": "要执行的命令（非空字符串）",
            }
        },
        "required": ["command"],
    },
)
