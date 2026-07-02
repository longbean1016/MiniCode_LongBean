"""权限规则持久化模块：读写 .bean/settings.json 中的权限配置。
   对标 Claude Code 的 settings.json 权限规则存储格式。"""

from pathlib import Path
from typing import Any

from app.permissions.rules import Behavior, MatchType, PermissionRule


# ── 默认配置文件路径 ──
def _default_settings_path(workspace_root: str = ".") -> Path:
    """获取 .bean/settings.json 的路径（项目级配置）。"""
    return Path(workspace_root).resolve() / ".bean" / "settings.json"


# ── 持久化 ──

def load_rules(workspace_root: str = ".") -> list[PermissionRule]:
    """从 .bean/settings.json 加载权限规则。

       Args:
           workspace_root: 项目根目录路径

       Returns:
           加载的规则列表，文件不存在时返回空列表
    """
    settings_path = _default_settings_path(workspace_root)
    if not settings_path.exists():
        return []

    try:
        data = _read_json(settings_path)
    except (OSError, ValueError):
        return []

    # 读取 permissions 节点
    permissions = data.get("permissions", {})
    if not isinstance(permissions, dict):
        return []

    rules: list[PermissionRule] = []
    # permissions 结构：{ "run_command": { "allow": ["pattern1", ...], "deny": [...], "ask": [...] } }
    for tool_name, tool_perms in permissions.items():
        if not isinstance(tool_perms, dict):
            continue
        for behavior_name, patterns in tool_perms.items():
            if behavior_name not in ("allow", "deny", "ask"):
                continue
            if not isinstance(patterns, list):
                continue
            behavior: Behavior = behavior_name  # type: ignore[assignment]
            for pattern_str in patterns:
                if not isinstance(pattern_str, str) or not pattern_str.strip():
                    continue
                rule = _parse_rule_string(tool_name, behavior, pattern_str.strip())
                if rule is not None:
                    rules.append(rule)

    return rules


def save_rules(rules: list[PermissionRule], workspace_root: str = ".") -> None:
    """将权限规则保存到 .bean/settings.json。

       已存在的其他配置字段（非 permissions）保持不变。

       Args:
           rules: 要保存的规则列表
           workspace_root: 项目根目录路径
    """
    settings_path = _default_settings_path(workspace_root)

    # 读取现有配置（如果存在）
    data: dict[str, Any] = {}
    if settings_path.exists():
        try:
            data = _read_json(settings_path)
        except (OSError, ValueError):
            data = {}

    # 构建新的 permissions 结构
    permissions: dict[str, dict[str, list[str]]] = {}
    for rule in rules:
        if rule.tool not in permissions:
            permissions[rule.tool] = {"allow": [], "deny": [], "ask": []}
        pattern_str = _format_rule_string(rule)
        permissions[rule.tool][rule.behavior].append(pattern_str)

    data["permissions"] = permissions

    # 确保 .bean 目录存在
    bean_dir = settings_path.parent
    bean_dir.mkdir(parents=True, exist_ok=True)

    # 写入文件
    _write_json(settings_path, data)


def merge_session_rules(
    persistent_rules: list[PermissionRule],
    session_rules: list[PermissionRule],
) -> list[PermissionRule]:
    """合并持久化规则和会话规则。

       会话规则优先级更高：如果会话规则和持久化规则冲突，保留会话规则。
       去重逻辑：相同 tool + behavior + match_type + pattern 的规则只保留一条。

       Args:
           persistent_rules: 从文件加载的持久化规则
           session_rules: 当前会话新增的规则

       Returns:
           合并后的规则列表
    """
    # 用 (tool, behavior, match_type, pattern) 作为去重键
    seen: set[tuple[str, str, str, str]] = set()
    merged: list[PermissionRule] = []

    # 先加载会话规则（高优先级）
    for rule in session_rules:
        key = (rule.tool, rule.behavior, rule.match_type, rule.pattern)
        if key not in seen:
            seen.add(key)
            merged.append(rule)

    # 再加载持久化规则（低优先级，不覆盖会话）
    for rule in persistent_rules:
        key = (rule.tool, rule.behavior, rule.match_type, rule.pattern)
        if key not in seen:
            seen.add(key)
            merged.append(rule)

    return merged


# ── 规则序列化/反序列化 ──

def _format_rule_string(rule: PermissionRule) -> str:
    """将规则格式化为字符串。

       格式：pattern 或 pattern (source)
       例如：'git status:*' 或 'rm *'

       对标 Claude Code settings.json 中的规则格式：
       - exact:  "git status"
       - prefix: "git status:*"
       - wildcard: "git *"
    """
    if rule.match_type == "exact":
        return rule.pattern
    elif rule.match_type == "prefix":
        return f"{rule.pattern}:*"
    elif rule.match_type == "wildcard":
        return f"{rule.pattern} *"
    return rule.pattern


def _parse_rule_string(
    tool_name: str,
    behavior: Behavior,
    pattern_str: str,
) -> PermissionRule | None:
    """从字符串解析权限规则。

       格式：
       - 'git status' → exact 匹配
       - 'git status:*' → prefix 匹配
       - 'git *' → wildcard 匹配
    """
    if not pattern_str:
        return None

    # 前缀匹配：pattern:*
    if pattern_str.endswith(":*"):
        prefix = pattern_str[:-2]
        return PermissionRule(
            tool=tool_name,
            behavior=behavior,
            match_type="prefix",
            pattern=prefix,
            source="user_settings",
        )

    # 通配符匹配：包含 *
    if "*" in pattern_str:
        return PermissionRule(
            tool=tool_name,
            behavior=behavior,
            match_type="wildcard",
            pattern=pattern_str,
            source="user_settings",
        )

    # 默认精确匹配
    return PermissionRule(
        tool=tool_name,
        behavior=behavior,
        match_type="exact",
        pattern=pattern_str,
        source="user_settings",
    )


# ── JSON 读写 ──

def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {}
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """写入 JSON 文件（带缩进）。"""
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
