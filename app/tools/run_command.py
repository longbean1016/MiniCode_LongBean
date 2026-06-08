
import subprocess
from typing import Any

"""命令执行工具，负责受控运行终端命令并返回截断后的输出。"""

from app.agent.permissions import PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, str]:
    """校验 run_command 的输入。"""

    # run_command 工具要求输入必须是一个字典，
    # 这样后面才能安全地读取 command 字段
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

    # 创建权限管理器：
    # 这里把当前工具上下文中的 cwd 作为工作区根目录，
    # 后续权限判断、命令超时、输出截断都通过它统一处理
    permission_manager = PermissionManager(context.cwd)

    # 取出用户输入的原始命令
    raw_command = validated_input["command"]

    # 先做权限判断，而不是直接执行命令
    # 这里会返回三种结果：
    # allow = 允许执行
    # ask = 需要用户授权
    # deny = 直接拒绝

    decision = permission_manager.check_command_permission(
        raw_command,
        approved_actions=context.approved_actions,
    )
    # 如果是绝对拒绝，就直接返回失败结果
    # 不再继续执行 subprocess.run
    if decision.status=="deny":
        return ToolResult(
            ok=False,
            output=f"命令被拒绝：{decision.reason}",
            error="PERMISSION_DENIED",
            meta={
                # 原始命令，便于上层记录和展示
                "command": raw_command,
                # 给用户或日志看的拒绝原因
                "reason": decision.reason,
                # 命中的规则，用于排查是哪条规则拦住了
                "rule": decision.rule,
                # 这次动作的唯一标识
                "action_key": decision.action_key,
            }
        )
    
    # 如果命令属于“高风险但可授权”，这里不直接执行，
    # 而是返回一个特殊错误码，让上层主循环去询问用户
    if decision.status=="ask":
        return ToolResult(
            ok=False,
            output="该命令需要用户授权之后才能执行。",
            error="PERMISSION_REQUIRED",
            meta={
                # 原始命令内容，后面授权提示要展示给用户
                "command": raw_command,
                # 命中高风险规则的原因说明
                "reason": decision.reason,
                # 具体命中的正则规则
                "rule": decision.rule,
                # 当前动作唯一键，用户批准后可以加入 approved_actions
                "action_key": decision.action_key,
            }
        )
    

    try:
        # 真正执行命令。
        # shell=True：允许直接执行命令字符串
        # cwd=context.cwd：命令默认在当前工作区下执行
        # capture_output=True：同时捕获 stdout 和 stderr
        # text=True：输出按文本处理，而不是 bytes
        # encoding/errors：尽量避免因为编码问题导致命令直接崩掉
        # timeout：防止命令长时间阻塞
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
        # 命令超时时，返回统一错误结果
        return ToolResult(
            ok=False,
            output=f"命令执行超时：超过 {permission_manager.get_command_timeout()} 秒",
            error="COMMAND_TIMEOUT",
            meta={
                "command": raw_command,
            },
        )
    except Exception as error:
        # 兜底捕获其他执行异常，例如系统层执行失败
        return ToolResult(
            ok=False,
            output=f"命令执行失败: {error}",
            error="COMMAND_EXEC_FAILED",
            meta={
                "command": raw_command,
            },
        )

     # 用一个列表统一收集 stdout 和 stderr
    # 这样最终输出结构会更清晰
    output_parts: list[str] = []

    # 如果有标准输出，就加上“标准输出”标题再放进去
    if result.stdout:
        output_parts.append(f"标准输出:\n{result.stdout.strip()}")

    # 如果有标准错误，就加上“标准错误”标题再放进去
    if result.stderr:
        output_parts.append(f"标准错误:\n{result.stderr.strip()}")

    # 把 stdout / stderr 统一拼成一个最终输出字符串
    output = "\n\n".join(output_parts).strip()

    # 如果命令确实执行了，但没有任何输出，给一个明确提示
    if not output:
        output = "命令执行完成，但没有输出。"

    # 对超长输出做统一截断，避免把上下文撑爆
    output = permission_manager.truncate_output(output)

    # 返回标准化结果：
    # ok 根据进程退出码判断
    # returncode == 0 一般表示执行成功
    # 非 0 表示命令执行了，但结果是失败状态
    return ToolResult(
        ok=(result.returncode == 0),
        output=output,
        error=None if result.returncode == 0 else "COMMAND_NON_ZERO_EXIT",
        meta={
            # 原始命令内容
            "command": raw_command,
            # 子进程退出码，便于调试
            "returncode": result.returncode,
            # 当前动作键，后续日志或授权复用时可能会用到
            "action_key": decision.action_key,
        },
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
                "description": "要执行的命令，必须是非空字符串",
            }
        },
        "required": ["command"],
    },
)
