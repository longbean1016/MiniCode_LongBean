"""命令执行工具，支持 Bash/PowerShell 分路、超时控制和命令安全分类。
   参考 Claude Code BashTool / PowerShellTool 语义实现。"""

import platform
import subprocess
from typing import Any

from app.agent.permissions import PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# ── 默认 / 最大超时（毫秒）──
DEFAULT_TIMEOUT_MS = 120_000   # 2 分钟
MAX_TIMEOUT_MS = 600_000       # 10 分钟
# 命令输出最大字符数
MAX_OUTPUT_CHARS = 30_000

# ── 只读命令集合（参考 Claude Code BashTool）──
_READ_ONLY_COMMANDS = {
    "cat", "head", "tail", "less", "more",
    "ls", "dir", "tree", "du",
    "find", "grep", "rg", "ag", "ack", "locate", "which", "whereis",
    "wc", "stat", "file", "strings",
    "echo", "printf", "date", "time",
    "git", "gh",  # git/gh 子命令再单独判断
    "npm", "yarn", "pnpm",  # 包管理器有只读子命令
    "python", "python3", "node",  # 脚本引擎
}
# ── 高风险命令关键词 ──
_HIGH_RISK_KEYWORDS = {
    "rm", "del", "rmdir", "rd", "erase",
    "format", "fdisk", "diskpart",
    ">", ">>",  # 输出重定向
    "|",  # 管道（总是需要检查）
    "sudo", "su", "runas",
    "chmod", "chown", "cacls", "icacls",
    "kill", "taskkill", "pkill",
    "shutdown", "reboot", "restart",
}
# ── Windows PowerShell 特有的高风险动词 ──
_PS_HIGH_RISK_VERBS = {
    "Remove-Item", "Stop-Process", "Stop-Service",
    "Set-ExecutionPolicy", "Clear-Content",
    "Disable-", "Uninstall-", "Unregister-",
}


def _validate(input_data: Any) -> dict[str, Any]:
    """校验 run_command 输入参数。

       支持 command / shell / timeout_ms / description（对齐 Claude Code BashTool）。
    """
    if not isinstance(input_data, dict):
        raise ValueError("run_command 输入必须是字典，包含 command 字段。")

    # ── command 必填 ──
    command = input_data.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command 必须是非空字符串。")

    # ── shell: auto / bash / powershell ──
    shell = input_data.get("shell", "auto")
    if shell not in ("auto", "bash", "powershell"):
        raise ValueError("shell 必须是 'auto'、'bash' 或 'powershell'。")

    # ── timeout_ms: 可选超时（毫秒）──
    raw_timeout = input_data.get("timeout_ms")
    timeout_ms = DEFAULT_TIMEOUT_MS
    if raw_timeout is not None:
        try:
            timeout_ms = int(raw_timeout)
        except (TypeError, ValueError):
            pass
    if timeout_ms < 1000 or timeout_ms > MAX_TIMEOUT_MS:
        raise ValueError(f"timeout_ms 必须在 1000 到 {MAX_TIMEOUT_MS} 之间。")

    # ── description: 命令用途简述（可选，帮助权限判断）──
    description = input_data.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError("description 必须是字符串。")

    return {
        "command": command.strip(),
        "shell": shell,
        "timeout_ms": timeout_ms,
        "description": description.strip() if description else "",
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """执行命令并返回结果，包含权限检查、超时控制和输出截断。"""
    permission_manager = PermissionManager(context.cwd)
    raw_command = validated_input["command"]
    shell = validated_input["shell"]
    timeout_ms = validated_input["timeout_ms"]
    description = validated_input.get("description", "")

    # ── 命令安全分类（只读 / 高风险）──
    risk_level = _classify_command_risk(raw_command)

    # ── 权限判断 ──
    decision = permission_manager.check_command_permission(
        raw_command,
        approved_actions=context.approved_actions,
    )
    if decision.status == "deny":
        return ToolResult(
            ok=False,
            output=f"命令被拒绝：{decision.reason}",
            error="PERMISSION_DENIED",
            meta={
                "command": raw_command,
                "reason": decision.reason,
                "rule": decision.rule,
                "action_key": decision.action_key,
                "risk_level": risk_level,
            },
        )

    if decision.status == "ask":
        return ToolResult(
            ok=False,
            output="该命令需要用户授权之后才能执行。",
            error="PERMISSION_REQUIRED",
            meta={
                "command": raw_command,
                "reason": decision.reason,
                "rule": decision.rule,
                "action_key": decision.action_key,
                "risk_level": risk_level,
            },
        )

    # ── 确定实际使用的 shell ──
    actual_shell = _resolve_shell(shell)
    use_shell = actual_shell in ("powershell", "cmd")
    timeout_seconds = timeout_ms / 1000.0

    try:
        if actual_shell == "powershell":
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", raw_command],
                cwd=context.cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        elif actual_shell == "cmd":
            result = subprocess.run(
                ["cmd.exe", "/c", raw_command],
                cwd=context.cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        else:
            result = subprocess.run(
                raw_command,
                cwd=context.cwd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            output=f"命令执行超时：超过 {timeout_seconds:.0f} 秒。",
            error="COMMAND_TIMEOUT",
            meta={"command": raw_command, "timeout_ms": timeout_ms},
        )
    except Exception as exc:
        return ToolResult(
            ok=False,
            output=f"命令执行失败：{exc}",
            error="COMMAND_EXEC_FAILED",
            meta={"command": raw_command},
        )

    # ── 构建输出 ──
    output_parts: list[str] = []
    if result.stdout:
        output_parts.append(f"标准输出:\n{result.stdout.strip()}")
    if result.stderr:
        output_parts.append(f"标准错误:\n{result.stderr.strip()}")

    output = "\n\n".join(output_parts).strip()
    if not output:
        output = "命令执行完成，但没有输出。"

    # ── 截断保护 ──
    output = permission_manager.truncate_output(output)

    return ToolResult(
        ok=(result.returncode == 0),
        output=output,
        error=None if result.returncode == 0 else f"COMMAND_EXIT_{result.returncode}",
        meta={
            "command": raw_command,
            "shell": actual_shell,
            "returncode": result.returncode,
            "timeout_ms": timeout_ms,
            "risk_level": risk_level,
            "action_key": decision.action_key,
        },
    )


def _resolve_shell(shell: str) -> str:
    """根据 auto 模式和当前平台决定使用哪个 shell。"""
    if shell == "powershell":
        return "powershell"
    if shell == "bash":
        return "bash"
    if shell == "auto":
        is_windows = platform.system() == "Windows"
        return "powershell" if is_windows else "bash"
    return "bash"


def _classify_command_risk(command: str) -> str:
    """对命令做安全分类：read_only / caution / high_risk。

       参考 Claude Code BashTool 的 isSearchOrReadBashCommand 语义。"""
    lowered = command.lower()
    # 取第一个词作为主命令
    parts = command.strip().split()
    if not parts:
        return "caution"

    main_cmd = parts[0].lower()
    # 去掉路径前缀，只取命令名
    if "/" in main_cmd or "\\" in main_cmd:
        main_cmd = main_cmd.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    # 高风险关键词检查
    for kw in _HIGH_RISK_KEYWORDS:
        if kw.lower() in lowered:
            return "high_risk"

    # PowerShell 高风险动词检查
    for verb in _PS_HIGH_RISK_VERBS:
        if lowered.startswith(verb.lower()):
            return "high_risk"

    # 只读命令（不带破坏性参数时）
    if main_cmd in _READ_ONLY_COMMANDS:
        # git/gh 的子命令检查
        if main_cmd in ("git", "gh"):
            dangerous_subcmds = {"push", "commit", "merge", "rebase", "reset", "rm", "clean"}
            subcmd = parts[1] if len(parts) > 1 else ""
            if subcmd in dangerous_subcmds:
                return "high_risk"
            return "read_only"
        # npm/yarn 的子命令检查
        if main_cmd in ("npm", "yarn", "pnpm"):
            read_subcmds = {"ls", "list", "view", "info", "outdated", "why", "explain", "audit"}
            subcmd = parts[1] if len(parts) > 1 else ""
            if subcmd in read_subcmds:
                return "read_only"
            return "caution"
        return "read_only"

    return "caution"


# ── 注册工具 ──
run_command_tool = ToolDefinition(
    name="run_command",
    description="执行系统命令并返回输出结果。支持 Bash/PowerShell 自动选择、超时控制和命令安全分类。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的命令。",
            },
            "shell": {
                "type": "string",
                "enum": ["auto", "bash", "powershell"],
                "description": '使用的 shell 类型。auto 表示根据平台自动选择。默认 "auto"。',
            },
            "timeout_ms": {
                "type": "integer",
                "description": f"命令超时时间（毫秒）。默认 {DEFAULT_TIMEOUT_MS}，最大 {MAX_TIMEOUT_MS}。",
            },
            "description": {
                "type": "string",
                "description": "命令用途的简短描述，有助于权限判断和日志记录。",
            },
        },
        "required": ["command"],
    },
)
