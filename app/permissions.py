

"""权限控制模块，负责约束工具访问范围和高风险操作审批。"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal


@dataclass(slots=True)
class PermissionDecision:
    """表示一次权限检查的结果。"""

    # 权限检查结论：
    # allow = 直接放行
    # ask = 需要用户授权
    # deny = 直接拒绝
    status: Literal["allow", "ask", "deny"]

    # 给上层看的原因说明，例如“命中高风险规则”
    reason: str = ""

    # 命中的具体规则，便于调试和日志记录
    rule: str = ""

    # 当前动作的唯一标识，用于“用户批准后再次执行时直接放行”
    action_key: str = ""


class PermissionManager:
    """工具权限管理器：负责路径、命令和输出安全边界。"""

    def __init__(
            self,
            workspace_root: str,
            command_timeout_seconds:int=15,
            max_output_chars:int=8000,
            ) -> None:
        
        # 统一使用绝对路径，避免相对路径绕过校验
        self._workspace_root = Path(workspace_root).resolve()

        # 命令执行超时时间（秒）
        self.command_timeout_seconds=command_timeout_seconds

         # 工具输出最大字符数，超出后截断
        self.max_output_chars = max_output_chars
        
       # 绝对禁止的危险命令：
        # 命中这些规则后，不给授权机会，直接拒绝
        self._deny_patterns = [
            r"\bformat\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bmkfs\b",
            r"\bdd\b",
            r"\bpoweroff\b",
            r"\bhalt\b",
            r"\bchown\b",
            r"\bchmod\s+777\b",
        ]

        # 需要用户授权的高风险命令：
        # 命中这些规则后，不直接执行，而是先 ask
        self._ask_patterns = [
            r"\brm\b",
            r"\bdel\b",
            r"\brmdir\b",
        ]



    def _resolve_path(self,target_path:str)-> Path: # type: ignore
        """
        将目标路径转成绝对路径。

        如果传入的是相对路径，就按 workspace_root 进行拼接。
        如果传入的是绝对路径，就直接解析。
        """
         # 先把传入字符串转成 Path 对象
        path = Path(target_path)

        # 如果传入的是相对路径，就默认以 workspace_root 为基准拼接
        if not path.is_absolute():
            path = self._workspace_root / path

        # resolve() 会把路径规范化成绝对路径
        return path.resolve()

    

    def ensure_path_access(self,target_path:str)-> Path:
        """
        检查目标路径是否在允许的工作目录范围内。
        越界路径仍然直接拒绝。
        """
        # 先把目标路径解析成标准绝对路径
        resolved_path = self._resolve_path(target_path)

        try:
            # relative_to() 的意思是：
            # 尝试判断 resolved_path 能不能视为 workspace_root 的子路径
            # 如果不能，说明它越界到了工作区外
            resolved_path.relative_to(self._workspace_root)
        except ValueError as error:
            # 越界访问直接拒绝，不给授权机会
            raise PermissionError(
                f"访问被拒绝：{resolved_path} 超出工作目录范围"
            ) from error

        # 合法时返回解析后的绝对路径，后续工具直接使用
        return resolved_path
    

    def check_command_permission(
        self,
        command: str,
        approved_actions: set[str] | None = None,
    ) -> PermissionDecision:
        """检查命令权限，返回 allow / ask / deny。"""

        # 去掉首尾空格并转成小写，方便统一匹配规则
        normalized_command = command.strip().lower()

        # 空命令没有执行意义，直接拒绝
        if not normalized_command:
            return PermissionDecision(
                status="deny",
                reason="命令不能为空",
                rule="EMPTY_COMMAND!",
            )
        # 用规范化后的命令生成一个动作唯一键
        # 后面如果用户批准过同一条命令，就可以靠这个键直接放行
        action_key = f"run_command::{normalized_command}"

        # 如果当前命令已经在“已批准动作集合”里，本次直接允许执行
        if approved_actions and action_key in approved_actions:
            return PermissionDecision(
                status="allow",
                reason="该命令已在当前会话中获得授权",
                rule="APPROVED_ACTION!",
                action_key=action_key
            )
        
        # 先检查绝对禁止规则
        # 一旦命中，就不允许申请授权，直接 deny
        for pattern in self._deny_patterns:
            if re.search(pattern,normalized_command):
                return PermissionDecision(
                    status="deny",
                    reason=f"命中绝对禁止规则: {pattern}",
                    rule=pattern,
                    action_key=action_key
                )
        # 再检查高风险规则
        # 命中后进入 ask，由上层询问用户是否批准
        for pattern in self._ask_patterns:
            if re.search(pattern,normalized_command):
                return PermissionDecision(
                    status="ask",
                    reason=f"命中高风险规则: {pattern}",
                    rule=pattern,
                    action_key=action_key
                )
        # 没命中任何规则时，默认允许执行
        return PermissionDecision(
            status="allow",
            action_key=action_key,
        )

    def get_command_timeout(self) -> int:
        """返回命令超时秒数。"""
        return self.command_timeout_seconds

    def truncate_output(self, text: str) -> str:
        """按最大字符数截断输出，避免超长响应。"""
        safe_text = text if isinstance(text, str) else str(text)
        if len(safe_text) <= self.max_output_chars:
            return safe_text

        clipped = safe_text[: self.max_output_chars]
        remain = len(safe_text) - self.max_output_chars
        return (
            f"{clipped}\n\n"
            f"[输出已截断：超出 {remain} 个字符，最大保留 {self.max_output_chars} 字符]"
        )
