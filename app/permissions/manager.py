"""权限控制管理器：集成规则引擎、命令安全分类和路径访问控制。
   对标 Claude Code 的 bashToolHasPermission 权限检查流程。

   检查链：
   1. exact-match: 精确匹配整个命令
   2. deny-rules:  命中 deny 规则 → 直接拒绝
   3. ask-rules:   命中 ask 规则 → 提示授权
   4. path-constraints: 输出重定向到工作区外 → 提示授权
   5. allow-rules: 命中 allow 规则 → 直接放行
   6. read-only-check: 纯只读命令 → 自动放行
   7. passthrough: 都不命中 → 默认询问用户
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from app.permissions.command_safety import (
    RiskLevel,
    classify_command_risk,
    has_command_injection_risk,
)
from app.permissions.rules import (
    PermissionResult as RulePermissionResult,
    PermissionRule,
    PermissionRuleEngine,
)
from app.permissions.settings import load_rules, save_rules


# ── 路径访问状态 ──

class PathAccessStatus(Enum):
    """路径检查结果状态。"""
    ALLOWED = "allowed"
    OUTSIDE_WORKSPACE = "outside_workspace"


@dataclass(slots=True)
class PathCheckResult:
    """表示一次路径访问检查的结果。

       status=ALLOWED 时，resolved_path 为解析后的绝对路径。
       status=OUTSIDE_WORKSPACE 时，message 包含提示信息。
    """
    status: PathAccessStatus
    resolved_path: Path | None = None
    message: str = ""


@dataclass(slots=True)
class PermissionDecision:
    """权限检查结果（向后兼容旧 API）。

       对标 Claude Code PermissionResult：
       - status: allow / ask / deny
       - reason: 给上层看的原因说明
       - rule: 命中的具体规则（调试用）
       - action_key: 动作唯一标识（用于"批准后重试"）
       - suggestions: 规则建议列表（用户批准后可保存的规则）
    """
    status: Literal["allow", "ask", "deny"]
    reason: str = ""
    rule: str = ""
    action_key: str = ""
    suggestions: list[PermissionRule] | None = None


class PermissionManager:
    """工具权限管理器：集成规则引擎 + 命令安全分类 + 路径访问控制 + 规则持久化。

       对标 Claude Code 的 bashToolHasPermission + checkPathConstraints + checkReadOnlyConstraints。
    """

    def __init__(
        self,
        workspace_root: str,
        additional_workspaces: set[Path] | None = None,
        permanent_workspaces: set[Path] | None = None,
        command_timeout_seconds: int = 15,
        max_output_chars: int = 8000,
    ) -> None:
        # ── 工作目录 ──
        self._workspace_root = Path(workspace_root).resolve()
        self._additional_workspaces: set[Path] = additional_workspaces or set()
        self._permanent_workspaces: set[Path] = permanent_workspaces or set()

        # ── 命令执行参数 ──
        self.command_timeout_seconds = command_timeout_seconds
        self.max_output_chars = max_output_chars

        # ── 新权限引擎 ──
        self._rule_engine = PermissionRuleEngine()

        # ── 从 .bean/settings.json 加载持久化规则 ──
        try:
            persistent_rules = load_rules(str(self._workspace_root))
            self._rule_engine.add_rules(persistent_rules)
        except Exception:
            pass

    # ── 工作目录管理 ──

    @property
    def additional_workspaces(self) -> frozenset[Path]:
        """返回会话级额外工作目录（只读）。"""
        return frozenset(self._additional_workspaces)

    @property
    def permanent_workspaces(self) -> frozenset[Path]:
        """返回永久额外工作目录（只读）。"""
        return frozenset(self._permanent_workspaces)

    @property
    def all_workspaces(self) -> list[Path]:
        """返回所有工作目录：root > permanent > additional。"""
        return (
            [self._workspace_root]
            + sorted(self._permanent_workspaces, key=str)
            + sorted(self._additional_workspaces, key=str)
        )

    def add_workspace(self, path: str, *, permanent: bool = False) -> Path:
        """将路径加入工作目录集合。"""
        resolved = Path(path).resolve()
        if permanent:
            self._permanent_workspaces.add(resolved)
        else:
            self._additional_workspaces.add(resolved)
        return resolved

    def get_permanent_workspace_paths(self) -> list[str]:
        """获取所有永久工作目录路径字符串。"""
        return sorted(str(p) for p in self._permanent_workspaces)

    # ── 规则管理 ──

    def add_session_rule(self, rule: PermissionRule) -> None:
        """添加一条会话级权限规则。"""
        self._rule_engine.add_rule(rule)

    def get_rule_engine(self) -> PermissionRuleEngine:
        """获取底层的规则引擎实例，供外部查询/调试。"""
        return self._rule_engine

    # ── 规则持久化 ──

    def persist_session_rules(self) -> None:
        """将当前所有规则（包括会话级新增的）持久化到 .bean/settings.json。"""
        try:
            save_rules(
                self._rule_engine._rules,  # type: ignore[arg-type]
                str(self._workspace_root),
            )
        except Exception:
            pass

    # ── 路径解析 ──

    def _resolve_path(self, target_path: str) -> Path:
        """将目标路径转成绝对路径。"""
        path = Path(target_path)
        if not path.is_absolute():
            path = self._workspace_root / path
        return path.resolve()

    # ── 路径访问检查 ──

    def check_path_access(self, target_path: str) -> PathCheckResult:
        """检查目标路径是否在允许的工作目录范围内。"""
        resolved_path = self._resolve_path(target_path)

        for ws in self.all_workspaces:
            try:
                resolved_path.relative_to(ws)
                return PathCheckResult(
                    status=PathAccessStatus.ALLOWED,
                    resolved_path=resolved_path,
                )
            except ValueError:
                continue

        return PathCheckResult(
            status=PathAccessStatus.OUTSIDE_WORKSPACE,
            resolved_path=resolved_path,
            message=(
                f"目标路径不在当前工作目录范围内：{target_path}\n"
                f"解析后路径：{resolved_path}\n"
                f"当前工作目录：{self._workspace_root}"
            ),
        )

    def ensure_path_access(self, target_path: str) -> Path:
        """检查目标路径是否在工作目录范围内（越界直接拒绝，向后兼容）。"""
        resolved_path = self._resolve_path(target_path)
        try:
            resolved_path.relative_to(self._workspace_root)
        except ValueError as error:
            raise PermissionError(
                f"访问被拒绝：{resolved_path} 超出工作目录范围"
            ) from error
        return resolved_path

    # ── 命令权限检查（对标 Claude Code bashToolHasPermission）──

    def check_command_permission(
        self,
        command: str,
        approved_actions: set[str] | None = None,
    ) -> PermissionDecision:
        """检查命令权限，走完整的 7 步检查链。

           Args:
               command: 要检查的命令文本
               approved_actions: 当前会话中已批准的动作键集合

           Returns:
               PermissionDecision: allow / ask / deny
        """
        normalized = command.strip()
        if not normalized:
            return PermissionDecision(
                status="deny",
                reason="命令不能为空",
                rule="EMPTY_COMMAND",
            )

        action_key = f"run_command::{normalized}"

        # ── 已批准动作直接放行 ──
        if approved_actions and action_key in approved_actions:
            return PermissionDecision(
                status="allow",
                reason="该命令已在当前会话中获得授权",
                rule="APPROVED_ACTION",
                action_key=action_key,
                suggestions=self._generate_suggestions(normalized),
            )

        # ── 命令注入检测 ──
        if has_command_injection_risk(normalized):
            return PermissionDecision(
                status="ask",
                reason="命令包含潜在的注入/替换模式，需要用户确认",
                rule="INJECTION_RISK",
                action_key=action_key,
            )

        # ── 第1-3步：规则引擎检查（exact → deny → ask）──
        rule_result = self._rule_engine.check("run_command", normalized)

        if rule_result.behavior == "deny":
            matched = rule_result.matched_rule
            return PermissionDecision(
                status="deny",
                reason=rule_result.message,
                rule=matched.pattern if matched else "DENY_RULE",
                action_key=action_key,
            )
        if rule_result.behavior == "ask":
            matched = rule_result.matched_rule
            return PermissionDecision(
                status="ask",
                reason=rule_result.message,
                rule=matched.pattern if matched else "ASK_RULE",
                action_key=action_key,
            )

        # ── 第4步：命令安全分类 ──
        risk = classify_command_risk(normalized)

        if risk == "critical":
            return PermissionDecision(
                status="deny",
                reason=f"命令 '{normalized.split()[0]}' 具有极高危险性，已被禁止执行",
                rule="CRITICAL_COMMAND",
                action_key=action_key,
            )
        if risk == "high_risk":
            return PermissionDecision(
                status="ask",
                reason=f"命令 '{normalized.split()[0]}' 为高风险操作，需要用户授权",
                rule="HIGH_RISK_COMMAND",
                action_key=action_key,
                suggestions=self._generate_suggestions(normalized),
            )

        # ── 第5步：allow 规则 ──
        if rule_result.behavior == "allow":
            matched = rule_result.matched_rule
            return PermissionDecision(
                status="allow",
                reason=rule_result.message,
                rule=matched.pattern if matched else "ALLOW_RULE",
                action_key=action_key,
            )

        # ── 第6步：只读命令自动放行 ──
        if risk == "read_only":
            return PermissionDecision(
                status="allow",
                reason="该命令为只读操作，自动放行",
                rule="READ_ONLY_COMMAND",
                action_key=action_key,
            )

        # ── 第7步：passthrough → 默认询问 ──
        return PermissionDecision(
            status="ask",
            reason="命令未命中任何授权规则，需要用户确认",
            rule="PASSTHROUGH",
            action_key=action_key,
            suggestions=self._generate_suggestions(normalized),
        )

    def _generate_suggestions(self, command: str) -> list[PermissionRule] | None:
        """生成规则建议（对标 Claude Code suggestionForExactCommand）。

           用户批准后可以保存这些规则，下次同模式命令不再询问。
        """
        engine = self._rule_engine
        return engine.suggest_rules("run_command", command, behavior="allow")

    # ── 输出治理 ──

    def get_command_timeout(self) -> int:
        """返回命令超时秒数。"""
        return self.command_timeout_seconds

    def truncate_output(self, text: str) -> str:
        """按最大字符数截断输出，避免超长响应。"""
        safe_text = text if isinstance(text, str) else str(text)
        if len(safe_text) <= self.max_output_chars:
            return safe_text
        clipped = safe_text[:self.max_output_chars]
        remain = len(safe_text) - self.max_output_chars
        return (
            f"{clipped}\n\n"
            f"[输出已截断：超出 {remain} 个字符，"
            f"最大保留 {self.max_output_chars} 字符]"
        )
