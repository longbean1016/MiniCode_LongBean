from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_KV_RE = re.compile(r"^-\s+\*\*(.+?)\*\*:\s*(.+)$")
_PATH_SCOPED_RULE_RE = re.compile(
    r"^(?:在|对|针对|对于)\s*([A-Za-z0-9_./\\-]+)(?:\s*(?:目录|文件夹|路径))?(?:\s*(?:下|中|里))?"
)
_BARE_PATH_SCOPED_RULE_RE = re.compile(
    r"^([A-Za-z0-9_./\\-]+)\s*(?:目录|文件夹|路径|下|中|里)\b"
)


@dataclass(frozen=True, slots=True)
class UserPolicyRule:
    """一条来自 USER.md 的结构化用户规则。"""

    instruction: str
    scope_type: str = "path"
    scope_value: str = ""

    def applies_to(self, task_text: str) -> bool:
        """判断规则是否命中当前任务。"""
        if self.scope_type != "path":
            return True

        normalized_scope = _normalize_scope_value(self.scope_value)
        normalized_task = _normalize_policy_match_text(task_text)
        if not normalized_scope or not normalized_task:
            return False

        candidates = {normalized_scope}
        basename = normalized_scope.split("/")[-1]
        if basename:
            candidates.add(basename)

        return any(candidate and candidate in normalized_task for candidate in candidates)


@dataclass(slots=True)
class ResolvedUserPolicy:
    """按全局偏好和作用域规则拆开的 USER.md 运行时策略。"""

    global_preferences: list[str] = field(default_factory=list)
    scoped_rules: list[UserPolicyRule] = field(default_factory=list)
    source_path: str = ""

    def active_rules_for(self, task_text: str) -> list[UserPolicyRule]:
        """筛出当前任务真正命中的规则。"""
        active_rules: list[UserPolicyRule] = []
        seen: set[tuple[str, str, str]] = set()
        for rule in self.scoped_rules:
            if not rule.applies_to(task_text):
                continue
            dedupe_key = (
                rule.scope_type,
                _normalize_scope_value(rule.scope_value),
                " ".join(rule.instruction.strip().lower().split()),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            active_rules.append(rule)
        return active_rules

    def to_prompt_section(self, active_rules: list[UserPolicyRule] | None = None) -> str:
        """渲染成每轮 system prompt 可直接注入的文本。"""
        lines: list[str] = []
        for item in _dedupe_lines(self.global_preferences)[:6]:
            lines.append(f"- 全局偏好：{item}")

        for rule in active_rules or []:
            scope_value = _normalize_scope_value(rule.scope_value) or rule.scope_value.strip()
            lines.append(f"- 当前任务命中规则（{scope_value}）：{rule.instruction.strip()}")

        if not lines:
            return ""
        return "\n".join(lines)


@dataclass(slots=True)
class UserProfileSnapshot:
    """工作区级 USER.md 快照，只保留当前上下文管理真正需要的字段。"""

    language: str = ""
    verbosity: str = ""
    response_style: str = ""
    comments: str = ""
    custom_instructions: str = ""
    source_path: str = ""
    extra_preferences: dict[str, str] = field(default_factory=dict)

    def to_preference_lines(self) -> list[str]:
        """把 USER.md 内容转成适合注入 state/compact memory 的短偏好列表。"""
        lines = self._build_structured_preference_lines()
        if self.custom_instructions.strip():
            lines.extend(_split_instruction_lines(self.custom_instructions))

        return _dedupe_lines(lines)

    def build_resolved_policy(self) -> ResolvedUserPolicy:
        """把自由文本偏好拆成全局偏好和作用域规则。"""
        global_preferences = self._build_structured_preference_lines()
        scoped_rules: list[UserPolicyRule] = []

        for line in _split_instruction_lines(self.custom_instructions):
            rule = _build_scoped_rule(line)
            if rule is None:
                global_preferences.append(line)
                continue
            scoped_rules.append(rule)

        return ResolvedUserPolicy(
            global_preferences=_dedupe_lines(global_preferences),
            scoped_rules=_dedupe_policy_rules(scoped_rules),
            source_path=self.source_path,
        )

    def _build_structured_preference_lines(self) -> list[str]:
        """只渲染可稳定映射的结构化偏好字段。"""
        lines: list[str] = []

        normalized_language = self.language.strip().lower()
        if normalized_language in {"zh-cn", "zh", "chinese"}:
            lines.append("默认使用中文回答")
        elif normalized_language:
            lines.append(f"默认使用 {self.language.strip()} 回答")

        normalized_verbosity = self.verbosity.strip().lower()
        if normalized_verbosity == "concise":
            lines.append("回答尽量简洁")
        elif normalized_verbosity == "detailed":
            lines.append("回答可以更详细")

        normalized_style = self.response_style.strip().lower()
        if normalized_style == "technical":
            lines.append("回答风格偏技术和直接")
        elif normalized_style:
            lines.append(f"回答风格偏 {self.response_style.strip()}")

        if "中文" in self.comments:
            lines.append("修改代码时加中文注释")
        elif self.comments.strip():
            lines.append(self.comments.strip())

        return lines


def load_user_profile(workspace: str) -> UserProfileSnapshot | None:
    """从工作区读取 USER.md；不存在时返回 None。"""
    path = Path(workspace).resolve() / "USER.md"
    if not path.exists() or not path.is_file():
        return None

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None

    profile = parse_user_md(content)
    profile.source_path = str(path)
    return profile


def save_user_profile(workspace: str, profile: UserProfileSnapshot) -> str:
    """把最小 USER.md 快照写回工作区根目录，并返回目标路径。"""
    path = _get_user_profile_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_user_md(profile), encoding="utf-8")
    profile.source_path = str(path)
    return str(path)


def parse_user_md(content: str) -> UserProfileSnapshot:
    """解析一个最小可用的 USER.md。"""
    profile = UserProfileSnapshot()
    sections = _split_sections(content)

    preferences = _parse_key_values(sections.get("preferences", ""))
    coding_style = _parse_key_values(sections.get("coding_style", ""))

    profile.language = preferences.get("language", "")
    profile.verbosity = preferences.get("verbosity", "")
    profile.response_style = preferences.get("response_style", "")
    profile.comments = coding_style.get("comments", "")
    profile.custom_instructions = _merge_manual_instruction_text(
        _extract_loose_manual_text(content),
        sections.get("custom_instructions", "").strip(),
    )
    profile.extra_preferences = preferences
    return profile


def serialize_user_md(profile: UserProfileSnapshot) -> str:
    """把最小 USER.md 快照序列化成 markdown。"""
    lines: list[str] = ["# User Profile", ""]

    # 只输出当前项目真正会参与上下文管理的字段，避免 USER.md 越写越散。
    preference_lines: list[str] = []
    if profile.language.strip():
        preference_lines.append(f"- **Language**: {profile.language.strip()}")
    if profile.verbosity.strip():
        preference_lines.append(f"- **Verbosity**: {profile.verbosity.strip()}")
    if profile.response_style.strip():
        preference_lines.append(f"- **Response Style**: {profile.response_style.strip()}")
    for key, value in profile.extra_preferences.items():
        normalized_key = key.strip().lower()
        if normalized_key in {"language", "verbosity", "response_style"}:
            continue
        cleaned_value = value.strip()
        if not cleaned_value:
            continue
        label = normalized_key.replace("_", " ").title()
        preference_lines.append(f"- **{label}**: {cleaned_value}")

    if preference_lines:
        lines.append("## Preferences")
        lines.extend(preference_lines)
        lines.append("")

    coding_style_lines: list[str] = []
    if profile.comments.strip():
        coding_style_lines.append(f"- **Comments**: {profile.comments.strip()}")
    if coding_style_lines:
        lines.append("## Coding Style")
        lines.extend(coding_style_lines)
        lines.append("")

    if profile.custom_instructions.strip():
        lines.append("## Custom Instructions")
        lines.append(profile.custom_instructions.strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


@dataclass(slots=True)
class UserProfileCommandResult:
    """`/user` 命令处理结果。"""

    handled: bool
    response_text: str = ""


def handle_user_profile_command(user_input: str, workspace: str) -> UserProfileCommandResult:
    """处理工作区级 `/user` 命令，提供手动维护 USER.md 的入口。"""
    raw = user_input.strip()
    if not raw.lower().startswith("/user"):
        return UserProfileCommandResult(handled=False)

    args = raw[len("/user") :].strip()
    if not args:
        return UserProfileCommandResult(
            handled=True,
            response_text=render_user_profile_summary(workspace),
        )

    normalized_args = args.lower()
    if normalized_args == "paths":
        path = _get_user_profile_path(workspace)
        status = "exists" if path.exists() else "missing"
        return UserProfileCommandResult(
            handled=True,
            response_text=f"USER.md 路径: {path} ({status})",
        )

    if normalized_args == "reset":
        path = _get_user_profile_path(workspace)
        if path.exists():
            path.unlink()
            return UserProfileCommandResult(
                handled=True,
                response_text=f"已删除 USER.md: {path}",
            )
        return UserProfileCommandResult(
            handled=True,
            response_text=f"USER.md 不存在: {path}",
        )

    if args.lower().startswith("add "):
        payload = args[4:].strip()
        return UserProfileCommandResult(
            handled=True,
            response_text=_handle_user_add(payload, workspace),
        )

    if args.lower().startswith("set "):
        payload = args[4:].strip()
        return UserProfileCommandResult(
            handled=True,
            response_text=_handle_user_set(payload, workspace),
        )

    return UserProfileCommandResult(
        handled=True,
        response_text=(
            "可用命令：/user, /user add <内容>, /user paths, /user set <key> <value>, /user reset"
        ),
    )


def render_user_profile_summary(workspace: str) -> str:
    """把当前 USER.md 渲染成中文摘要，便于终端直接查看。"""
    profile = load_user_profile(workspace)
    path = _get_user_profile_path(workspace)
    if profile is None:
        return (
            f"当前未配置 USER.md。\n"
            f"USER.md 路径: {path} (missing)\n"
            "可用命令：/user add <内容> 或 /user set <key> <value>"
        )

    lines = profile.to_preference_lines()
    if not lines:
        lines = ["当前 USER.md 没有可用于上下文注入的偏好项"]

    rendered = [f"USER.md 路径: {path} (exists)"]
    rendered.extend(f"- {line}" for line in lines)
    return "\n".join(rendered)


def _split_sections(content: str) -> dict[str, str]:
    """把 markdown 按 `##` 标题切成 section。"""
    sections: dict[str, str] = {}
    parts = _SECTION_RE.split(content)
    for index in range(1, len(parts) - 1, 2):
        heading = parts[index].strip().lower().replace(" ", "_")
        sections[heading] = parts[index + 1]
    return sections


def _extract_loose_manual_text(content: str) -> str:
    """提取没有落在 `##` section 里的自由文本，兼容用户手动直接编辑 USER.md。"""
    lines: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("## "):
            continue
        if _KV_RE.match(stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def _parse_key_values(body: str) -> dict[str, str]:
    """解析 `- **Key**: value` 这种 USER.md 键值行。"""
    result: dict[str, str] = {}
    for raw_line in body.strip().splitlines():
        match = _KV_RE.match(raw_line.strip())
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        result[key] = value
    return result


def _split_instruction_lines(text: str) -> list[str]:
    """把 custom instructions 拆成几条短偏好说明。"""
    result: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip().rstrip("。.;；")
        if not line:
            continue
        result.append(line)
    return result[:4]


def _dedupe_lines(lines: list[str]) -> list[str]:
    """去重并保留原顺序。"""
    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = " ".join(line.strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(line.strip())
    return result


def _merge_manual_instruction_text(*parts: str) -> str:
    """合并多来源手写说明，避免 USER.md 里的自由文本被结构化字段覆盖。"""
    merged_lines: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for line in part.splitlines():
            cleaned = line.strip()
            normalized = " ".join(cleaned.lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged_lines.append(cleaned)
    return "\n".join(merged_lines)


def _build_scoped_rule(line: str) -> UserPolicyRule | None:
    """尽量从自由文本里提取“某个路径/目录下的规则”。"""
    cleaned = line.strip()
    if not cleaned:
        return None

    match = _PATH_SCOPED_RULE_RE.match(cleaned) or _BARE_PATH_SCOPED_RULE_RE.match(cleaned)
    if match is None:
        return None

    scope_value = _normalize_scope_value(match.group(1))
    if not scope_value:
        return None

    return UserPolicyRule(
        instruction=cleaned,
        scope_type="path",
        scope_value=scope_value,
    )


def _dedupe_policy_rules(rules: list[UserPolicyRule]) -> list[UserPolicyRule]:
    """对结构化规则做稳定去重。"""
    result: list[UserPolicyRule] = []
    seen: set[tuple[str, str, str]] = set()
    for rule in rules:
        dedupe_key = (
            rule.scope_type,
            _normalize_scope_value(rule.scope_value),
            " ".join(rule.instruction.strip().lower().split()),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result.append(rule)
    return result


def _normalize_scope_value(value: str) -> str:
    """把路径作用域统一成稳定匹配格式。"""
    normalized = str(value).strip().strip("\"'`")
    normalized = normalized.replace("\\", "/")
    normalized = normalized.lstrip("./")
    normalized = normalized.strip("/")
    return normalized.lower()


def _normalize_policy_match_text(text: str) -> str:
    """归一化任务文本，便于路径规则匹配。"""
    return " ".join(str(text).strip().lower().replace("\\", "/").split())


def _handle_user_set(payload: str, workspace: str) -> str:
    """处理 `/user set <key> <value>` 并持久化到 USER.md。"""
    parts = payload.strip().split(maxsplit=1)
    if len(parts) < 2:
        return (
            "用法：/user set <key> <value>\n"
            "支持 key：preferences.language, preferences.verbosity, "
            "preferences.response_style, coding_style.comments, custom_instructions"
        )

    key = parts[0].strip().lower()
    value = parts[1].strip()
    if not value:
        return "value 不能为空。"

    profile = load_user_profile(workspace) or UserProfileSnapshot()
    if not _apply_user_setting(profile, key, value):
        return (
            "不支持的 key。\n"
            "支持 key：preferences.language, preferences.verbosity, "
            "preferences.response_style, coding_style.comments, custom_instructions"
        )

    path = save_user_profile(workspace, profile)
    return f"已写入 USER.md: {path}"


def _handle_user_add(payload: str, workspace: str) -> str:
    """处理 `/user add <内容>`，统一把自由偏好追加到 USER.md。"""
    content = payload.strip()
    if not content:
        return "用法：/user add <内容>"

    profile = load_user_profile(workspace) or UserProfileSnapshot()
    existing_lines = _split_instruction_lines(profile.custom_instructions)
    existing_lines.append(content)
    profile.custom_instructions = "\n".join(
        f"- {line}" for line in _dedupe_lines(existing_lines)
    )

    # 常见的“中文注释”偏好顺手同步到结构化字段，方便后续稳定注入。
    if "注释" in content and "中文" in content and not profile.comments.strip():
        profile.comments = "中文注释"

    path = save_user_profile(workspace, profile)
    return f"已追加到 USER.md: {path}"


def _apply_user_setting(profile: UserProfileSnapshot, key: str, value: str) -> bool:
    """把一条设置应用到最小 USER.md 快照。"""
    key_aliases = {
        "language": "preferences.language",
        "verbosity": "preferences.verbosity",
        "response_style": "preferences.response_style",
        "comments": "coding_style.comments",
    }
    normalized_key = key_aliases.get(key, key)
    cleaned_value = value.strip()

    if normalized_key == "preferences.language":
        profile.language = cleaned_value
        profile.extra_preferences["language"] = cleaned_value
        return True
    if normalized_key == "preferences.verbosity":
        profile.verbosity = cleaned_value
        profile.extra_preferences["verbosity"] = cleaned_value
        return True
    if normalized_key == "preferences.response_style":
        profile.response_style = cleaned_value
        profile.extra_preferences["response_style"] = cleaned_value
        return True
    if normalized_key == "coding_style.comments":
        profile.comments = cleaned_value
        return True
    if normalized_key == "custom_instructions":
        profile.custom_instructions = cleaned_value
        return True

    return False


def _get_user_profile_path(workspace: str) -> Path:
    """返回当前工作区唯一的 USER.md 路径。"""
    return Path(workspace).resolve() / "USER.md"
