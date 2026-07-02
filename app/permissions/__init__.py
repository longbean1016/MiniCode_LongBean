"""权限管理模块：统一规则引擎、命令安全分类和规则持久化。
   模块结构：
   - rules.py: 权限规则引擎（allow/deny/ask × exact/prefix/wildcard）
   - command_safety.py: 命令安全分类（只读白名单/高风险检测/注入检测）
   - settings.py: 规则持久化（读写 .bean/settings.json）
   - manager.py: 迁移自 app/agent/permissions.py（向后兼容）
"""

from app.permissions.rules import (
    Behavior,
    MatchType,
    PermissionResult,
    PermissionRule,
    PermissionRuleEngine,
)
from app.permissions.command_safety import (
    classify_command_risk,
    has_command_injection_risk,
    is_git_dangerous,
    RiskLevel,
)
from app.permissions.settings import (
    load_rules,
    save_rules,
    merge_session_rules,
)

__all__ = [
    # 规则引擎
    "Behavior",
    "MatchType",
    "PermissionResult",
    "PermissionRule",
    "PermissionRuleEngine",
    # 命令安全
    "classify_command_risk",
    "has_command_injection_risk",
    "is_git_dangerous",
    "RiskLevel",
    # 持久化
    "load_rules",
    "save_rules",
    "merge_session_rules",
]
