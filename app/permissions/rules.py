"""权限规则引擎：支持 allow/deny/ask 三种行为 × exact/prefix/wildcard 三种匹配模式。
   对标 Claude Code bashPermissions.ts 的规则匹配逻辑。"""

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Literal


# ── 权限行为类型 ──
Behavior = Literal["allow", "deny", "ask"]

# ── 匹配模式 ──
MatchType = Literal["exact", "prefix", "wildcard"]


@dataclass(slots=True)
class PermissionRule:
    """表示一条权限规则。

       对标 Claude Code 的 PermissionRule：
       - tool: 工具名，如 "run_command" / "read_file" / "write_file"
       - behavior: allow（放行）/ deny（拒绝）/ ask（询问用户）
       - match_type: exact（精确匹配）/ prefix（前缀匹配）/ wildcard（通配符匹配）
       - pattern: 匹配内容（命令文本或文件路径）
       - source: 规则来源（user_settings / session / system）
    """
    tool: str
    behavior: Behavior
    match_type: MatchType
    pattern: str
    source: str = "session"  # 规则来源：user_settings / session / system


@dataclass(slots=True)
class PermissionResult:
    """权限检查结果。

       对标 Claude Code 的 PermissionResult：
       - behavior: allow / deny / ask / passthrough（未命中任何规则）
       - message: 给用户的提示信息
       - matched_rule: 命中的规则（如果有）
       - suggestions: 规则建议列表（用户批准后可保存的规则）
    """
    behavior: Literal["allow", "deny", "ask", "passthrough"]
    message: str = ""
    matched_rule: PermissionRule | None = None
    suggestions: list[PermissionRule] = field(default_factory=list)


class PermissionRuleEngine:
    """权限规则引擎：管理规则集合并执行匹配检查。

       对标 Claude Code 的 matchingRulesForInput + bashToolCheckPermission：
       1. 先检查精确匹配（exact）
       2. 再检查前缀/通配符 deny
       3. 再检查前缀/通配符 ask
       4. 再检查前缀/通配符 allow
       5. 都不命中返回 passthrough（默认询问）
    """

    def __init__(self) -> None:
        # 所有已注册的规则列表
        self._rules: list[PermissionRule] = []

    # ── 规则管理 ──

    def add_rule(self, rule: PermissionRule) -> None:
        """添加一条权限规则。"""
        self._rules.append(rule)

    def add_rules(self, rules: list[PermissionRule]) -> None:
        """批量添加权限规则。"""
        self._rules.extend(rules)

    def remove_rule(self, rule: PermissionRule) -> bool:
        """移除一条权限规则，返回是否成功。"""
        try:
            self._rules.remove(rule)
            return True
        except ValueError:
            return False

    def get_rules_for_tool(self, tool_name: str) -> list[PermissionRule]:
        """获取指定工具的所有规则。"""
        return [r for r in self._rules if r.tool == tool_name]

    def clear_rules(self) -> None:
        """清空所有规则。"""
        self._rules.clear()

    # ── 规则匹配 ──

    def check(
        self,
        tool_name: str,
        content: str,
        *,
        skip_allow: bool = False,
    ) -> PermissionResult:
        """对指定工具和内容执行权限检查。

        Args:
            tool_name: 工具名称（如 "run_command"）
            content: 要检查的内容（如命令文本 "rm -rf /tmp"）
            skip_allow: 是否跳过 allow 规则（deny 检查时使用）

        Returns:
            PermissionResult: 检查结果
        """
        # 按优先级排序：exact > deny > ask > allow
        # 同一优先级内，deny 优先于 ask，ask 优先于 allow
        tool_rules = self.get_rules_for_tool(tool_name)

        # ── 第1步：精确匹配优先 ──
        exact_rules = [r for r in tool_rules if r.match_type == "exact"]
        exact_match = self._find_match(exact_rules, content)
        if exact_match is not None:
            if exact_match.behavior == "deny":
                return PermissionResult("deny", f"命令被拒绝：命中规则 '{exact_match.pattern}'", exact_match)
            if exact_match.behavior == "ask":
                return PermissionResult("ask", f"命令需要授权：命中规则 '{exact_match.pattern}'", exact_match)
            if exact_match.behavior == "allow":
                return PermissionResult("allow", f"命令已授权：命中规则 '{exact_match.pattern}'", exact_match)

        # ── 第2步：前缀/通配符 deny 优先 ──
        fuzzy_rules = [r for r in tool_rules if r.match_type != "exact"]
        fuzzy_deny = self._find_match(
            [r for r in fuzzy_rules if r.behavior == "deny"], content)
        if fuzzy_deny is not None:
            return PermissionResult("deny", f"命令被拒绝：命中规则 '{fuzzy_deny.pattern}'", fuzzy_deny)

        # ── 第3步：前缀/通配符 ask ──
        fuzzy_ask = self._find_match(
            [r for r in fuzzy_rules if r.behavior == "ask"], content)
        if fuzzy_ask is not None:
            return PermissionResult("ask", f"命令需要授权：命中规则 '{fuzzy_ask.pattern}'", fuzzy_ask)

        # ── 第4步：前缀/通配符 allow ──
        if not skip_allow:
            fuzzy_allow = self._find_match(
                [r for r in fuzzy_rules if r.behavior == "allow"], content)
            if fuzzy_allow is not None:
                return PermissionResult("allow", f"命令已授权：命中规则 '{fuzzy_allow.pattern}'", fuzzy_allow)

        # ── 第5步：未命中任何规则 → passthrough ──
        return PermissionResult("passthrough", "未命中任何权限规则，默认需要授权")

    def _find_match(
        self,
        rules: list[PermissionRule],
        content: str,
    ) -> PermissionRule | None:
        """在规则列表中查找第一个匹配的规则。"""
        for rule in rules:
            if self._matches(rule, content):
                return rule
        return None

    def _matches(self, rule: PermissionRule, content: str) -> bool:
        """判断一条规则是否匹配给定内容。"""
        if rule.match_type == "exact":
            # 精确匹配：内容必须完全相同
            return rule.pattern == content
        elif rule.match_type == "prefix":
            # 前缀匹配：内容以 pattern 开头，且后面是空格或结尾
            # 防止 "ls:*" 误匹配 "lsof"
            if content == rule.pattern:
                return True
            return content.startswith(rule.pattern + " ")
        elif rule.match_type == "wildcard":
            # 通配符匹配：使用 fnmatch（支持 * 和 ?）
            return fnmatch(content, rule.pattern)
        return False

    # ── 规则建议生成 ──

    def suggest_rules(
        self,
        tool_name: str,
        content: str,
        behavior: Behavior = "allow",
    ) -> list[PermissionRule]:
        """根据用户批准的命令，自动生成建议的规则。

           对标 Claude Code 的 suggestionForExactCommand / suggestionForPrefix：
           - 单层命令（如 'ls -la'）→ 建议前缀规则 'ls:*'
           - 子命令（如 'git push'）→ 建议前缀规则 'git push:*'
           - 复杂命令 → 建议精确规则

        Args:
            tool_name: 工具名称
            content: 用户刚批准的命令内容
            behavior: 建议的权限行为（默认 allow）

        Returns:
            建议的规则列表
        """
        suggestions: list[PermissionRule] = []
        tokens = content.strip().split()

        if not tokens:
            return suggestions

        # 单 token 命令（如 'ls'）→ 精确匹配
        if len(tokens) == 1:
            suggestions.append(PermissionRule(
                tool=tool_name,
                behavior=behavior,
                match_type="prefix",
                pattern=f"{tokens[0]}:*",
                source="session",
            ))
            return suggestions

        # 多 token 命令：尝试提取子命令前缀
        # 第二个 token 看起来像子命令（纯小写字母+连字符）→ 双词前缀
        main_cmd = tokens[0]
        sub_cmd = tokens[1] if len(tokens) > 1 else ""

        # 去掉常见安全前缀（环境变量赋值如 FOO=bar）
        if "=" in main_cmd:
            main_cmd = tokens[1] if len(tokens) > 1 else main_cmd
            sub_cmd = tokens[2] if len(tokens) > 2 else ""

        # 判断第二个 token 是否像子命令（纯字母+连字符，不以 - 开头）
        if sub_cmd and not sub_cmd.startswith("-") and sub_cmd.replace("-", "").isalpha():
            # 双词前缀：如 'git push:*'
            suggestions.append(PermissionRule(
                tool=tool_name,
                behavior=behavior,
                match_type="prefix",
                pattern=f"{main_cmd} {sub_cmd}:*",
                source="session",
            ))

        # 总是追加单词前缀作为备选
        suggestions.append(PermissionRule(
            tool=tool_name,
            behavior=behavior,
            match_type="prefix",
            pattern=f"{main_cmd}:*",
            source="session",
        ))

        return suggestions
