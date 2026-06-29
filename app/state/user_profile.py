from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

"""用户偏好解析与规则生效模块，负责把 `.memory/USER.md` 转成运行时策略。"""


_PATH_SCOPED_RULE_RE = re.compile(
    r"^(?:在|对|针对|对于)\s*([A-Za-z0-9_./\\-]+)(?:\s*(?:目录|文件夹|路径))?(?:\s*(?:下|中|里))?"
)
_BARE_PATH_SCOPED_RULE_RE = re.compile(
    r"^([A-Za-z0-9_./\\-]+)\s*(?:目录|文件夹|路径|下|中|里)\b"
)
_LIST_ITEM_RE = re.compile(r"^\s*-\s+(.*\S)\s*$")
_INDENTED_LINE_RE = re.compile(r"^(?: {2,}|\t+)(.*)$")
_USER_TITLE = "# 用户记忆"


@dataclass(frozen=True, slots=True)
class UserPolicyRule:
    instruction: str
    scope_type: str = "path"
    scope_value: str = ""

    def applies_to(self, task_text: str) -> bool:
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
    identity_lines: list[str] = field(default_factory=list)
    global_preferences: list[str] = field(default_factory=list)
    scoped_rules: list[UserPolicyRule] = field(default_factory=list)
    source_path: str = ""

    def active_rules_for(self, task_text: str) -> list[UserPolicyRule]:
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
        lines: list[str] = []
        for item in _dedupe_lines(self.identity_lines)[:8]:
            lines.append(f"- 身份信息：{item}")
        for item in _dedupe_lines(self.global_preferences)[:12]:
            lines.append(f"- 全局偏好：{item}")
        for rule in active_rules or []:
            scope_value = _normalize_scope_value(rule.scope_value) or rule.scope_value.strip()
            lines.append(f"- 当前任务命中规则（{scope_value}）：{rule.instruction.strip()}")
        return "\n".join(lines).strip()


@dataclass(slots=True)
class UserProfileSnapshot:
    language: str = ""
    verbosity: str = ""
    response_style: str = ""
    comments: str = ""
    identity_instructions: str = ""
    preference_instructions: str = ""
    custom_instructions: str = ""
    source_path: str = ""
    extra_preferences: dict[str, str] = field(default_factory=dict)

    def to_preference_lines(self) -> list[str]:
        lines = self._build_identity_lines()
        lines.extend(self._build_structured_preference_lines())
        lines.extend(_split_instruction_lines(self.preference_instructions, limit=12))
        lines.extend(_split_instruction_lines(self.custom_instructions, limit=8))
        return _dedupe_lines(lines)

    def build_resolved_policy(self) -> ResolvedUserPolicy:
        identity_lines = self._build_identity_lines()
        global_preferences = self._build_structured_preference_lines()
        scoped_rules: list[UserPolicyRule] = []

        free_text_lines = []
        free_text_lines.extend(_split_instruction_lines(self.preference_instructions, limit=12))
        free_text_lines.extend(_split_instruction_lines(self.custom_instructions, limit=8))

        for line in free_text_lines:
            rule = _build_scoped_rule(line)
            if rule is None:
                global_preferences.append(line)
                continue
            scoped_rules.append(rule)

        return ResolvedUserPolicy(
            identity_lines=_dedupe_lines(identity_lines),
            global_preferences=_dedupe_lines(global_preferences),
            scoped_rules=_dedupe_policy_rules(scoped_rules),
            source_path=self.source_path,
        )

    def _build_identity_lines(self) -> list[str]:
        return _split_instruction_lines(self.identity_instructions, limit=8)

    def _build_structured_preference_lines(self) -> list[str]:
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


@dataclass(slots=True)
class UserProfileCommandResult:
    handled: bool
    response_text: str = ""


def load_user_profile(workspace: str) -> UserProfileSnapshot | None:
    path = _get_user_profile_path(workspace)
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
    path = _get_user_profile_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_user_md(profile), encoding="utf-8")
    profile.source_path = str(path)
    return str(path)


def parse_user_md(content: str) -> UserProfileSnapshot:
    profile = UserProfileSnapshot()
    entries = _parse_markdown_list_entries(content)

    identity_lines: list[str] = []
    preference_lines: list[str] = []
    custom_lines: list[str] = []
    extra_preferences: dict[str, str] = {}

    for entry in entries:
        normalized = entry.strip()
        lower_line = normalized.lower()

        if lower_line.startswith("身份信息：") or lower_line.startswith("身份信息:"):
            identity_lines.append(_strip_prefix_value(normalized))
            continue
        if lower_line.startswith("全局偏好：") or lower_line.startswith("全局偏好:"):
            preference_lines.append(_strip_prefix_value(normalized))
            continue
        if lower_line.startswith("自定义：") or lower_line.startswith("自定义:"):
            custom_lines.append(_strip_prefix_value(normalized))
            continue

        if "默认使用中文回答" in normalized:
            profile.language = "zh-CN"
        elif normalized.startswith("默认使用 ") and normalized.endswith(" 回答"):
            profile.language = normalized[len("默认使用 ") : -len(" 回答")].strip()
        elif normalized == "回答尽量简洁":
            profile.verbosity = "concise"
        elif normalized == "回答可以更详细":
            profile.verbosity = "detailed"
        elif normalized == "回答风格偏技术和直接":
            profile.response_style = "technical"
        elif normalized.startswith("回答风格偏 "):
            profile.response_style = normalized[len("回答风格偏 ") :].strip()
        elif normalized == "修改代码时加中文注释":
            profile.comments = "中文注释"

        if normalized:
            preference_lines.append(normalized)

    profile.identity_instructions = _merge_manual_instruction_text(*identity_lines)
    profile.preference_instructions = _merge_manual_instruction_text(*preference_lines)
    profile.custom_instructions = _merge_manual_instruction_text(*custom_lines)
    if profile.language:
        extra_preferences["language"] = profile.language
    if profile.verbosity:
        extra_preferences["verbosity"] = profile.verbosity
    if profile.response_style:
        extra_preferences["response_style"] = profile.response_style
    profile.extra_preferences = extra_preferences
    return profile


def serialize_user_md(profile: UserProfileSnapshot) -> str:
    lines: list[str] = [_USER_TITLE, ""]

    rendered_entries: list[str] = []
    for item in _dedupe_lines(profile._build_identity_lines())[:8]:
        rendered_entries.append(f"- 身份信息：{item}")

    structured_preferences = profile._build_structured_preference_lines()
    free_preferences = _split_instruction_lines(profile.preference_instructions, limit=12)
    for item in _dedupe_lines(structured_preferences + free_preferences)[:12]:
        rendered_entries.append(f"- 全局偏好：{item}")

    for item in _dedupe_lines(_split_instruction_lines(profile.custom_instructions, limit=8))[:8]:
        rendered_entries.append(f"- 自定义：{item}")

    lines.extend(rendered_entries)
    if rendered_entries:
        lines.append("")
    lines.append("---")
    return "\n".join(lines).rstrip() + "\n"


def handle_user_profile_command(user_input: str, workspace: str) -> UserProfileCommandResult:
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
            "可用命令：/user, /user add <内容>, /user add identity <内容>, "
            "/user add preferences <内容>, /user add custom <内容>, "
            "/user paths, /user set <key> <value>, /user reset"
        ),
    )


def render_user_profile_summary(workspace: str) -> str:
    profile = load_user_profile(workspace)
    path = _get_user_profile_path(workspace)
    if profile is None:
        return (
            f"当前未配置 USER.md。\n"
            f"USER.md 路径: {path} (missing)\n"
            "可用命令：/user add <内容>、/user add identity <内容> 或 /user set <key> <value>"
        )

    rendered = [f"USER.md 路径: {path} (exists)"]
    identity_lines = _dedupe_lines(profile._build_identity_lines())
    preference_lines = _dedupe_lines(
        profile._build_structured_preference_lines()
        + _split_instruction_lines(profile.preference_instructions, limit=12)
    )
    custom_lines = _dedupe_lines(_split_instruction_lines(profile.custom_instructions, limit=8))

    if identity_lines:
        rendered.append("")
        rendered.append("[Identity]")
        rendered.extend(f"- {line}" for line in identity_lines)
    if preference_lines:
        rendered.append("")
        rendered.append("[Preferences]")
        rendered.extend(f"- {line}" for line in preference_lines)
    if custom_lines:
        rendered.append("")
        rendered.append("[Custom]")
        rendered.extend(f"- {line}" for line in custom_lines)
    if len(rendered) == 1:
        rendered.append("")
        rendered.append("当前 USER.md 没有可用于上下文注入的偏好项")
    return "\n".join(rendered)


def _parse_markdown_list_entries(content: str) -> list[str]:
    entries: list[str] = []
    current_lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current_lines:
                current_lines.append("")
            continue
        if line.strip() == "---":
            continue
        if line.lstrip().startswith("#"):
            continue
        item_match = _LIST_ITEM_RE.match(line)
        if item_match:
            if current_lines:
                entries.append("\n".join(current_lines).rstrip())
            current_lines = [item_match.group(1).strip()]
            continue
        indented_match = _INDENTED_LINE_RE.match(line)
        if indented_match and current_lines:
            current_lines.append(indented_match.group(1).rstrip())
    if current_lines:
        entries.append("\n".join(current_lines).rstrip())
    return [entry for entry in entries if entry.strip()]


def _split_instruction_lines(text: str, limit: int | None = None) -> list[str]:
    result: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip().rstrip("。.;；")
        if not line:
            continue
        result.append(line)
    if limit is None or limit <= 0:
        return result
    return result[:limit]


def _dedupe_lines(lines: list[str]) -> list[str]:
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
    merged_lines: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for line in part.splitlines():
            cleaned = line.strip().lstrip("-").strip()
            normalized = " ".join(cleaned.lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged_lines.append(cleaned)
    return "\n".join(merged_lines)


def _build_scoped_rule(line: str) -> UserPolicyRule | None:
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
    normalized = str(value).strip().strip("\"'`")
    normalized = normalized.replace("\\", "/")
    normalized = normalized.lstrip("./")
    normalized = normalized.strip("/")
    return normalized.lower()


def _normalize_policy_match_text(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("\\", "/").split())


def _handle_user_set(payload: str, workspace: str) -> str:
    parts = payload.strip().split(maxsplit=1)
    if len(parts) < 2:
        return (
            "用法：/user set <key> <value>\n"
            "支持 key：preferences.language, preferences.verbosity, "
            "preferences.response_style, coding_style.comments, "
            "identity_instructions, preference_instructions, custom_instructions"
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
            "preferences.response_style, coding_style.comments, "
            "identity_instructions, preference_instructions, custom_instructions"
        )

    path = save_user_profile(workspace, profile)
    return f"已写入 USER.md: {path}"


def _handle_user_add(payload: str, workspace: str) -> str:
    content = payload.strip()
    if not content:
        return (
            "用法：/user add <内容>\n"
            "或：/user add identity <内容>\n"
            "或：/user add preferences <内容>\n"
            "或：/user add custom <内容>"
        )

    profile = load_user_profile(workspace) or UserProfileSnapshot()
    category, normalized_content = _parse_user_add_payload(content)
    if not normalized_content:
        return "追加内容不能为空。"

    if category == "identity":
        profile.identity_instructions = _append_instruction_line(
            profile.identity_instructions,
            normalized_content,
            limit=8,
        )
    elif category == "preferences":
        profile.preference_instructions = _append_instruction_line(
            profile.preference_instructions,
            normalized_content,
            limit=12,
        )
    else:
        profile.custom_instructions = _append_instruction_line(
            profile.custom_instructions,
            normalized_content,
            limit=8,
        )

    if "注释" in normalized_content and "中文" in normalized_content and not profile.comments.strip():
        profile.comments = "中文注释"

    path = save_user_profile(workspace, profile)
    return f"已追加到 USER.md: {path}"


def _parse_user_add_payload(payload: str) -> tuple[str, str]:
    stripped = payload.strip()
    if not stripped:
        return "custom", ""

    parts = stripped.split(maxsplit=1)
    category_aliases = {
        "identity": "identity",
        "identify": "identity",
        "idetify": "identity",
        "preferences": "preferences",
        "preference": "preferences",
        "custom": "custom",
    }
    normalized_head = parts[0].strip().lower()
    if normalized_head in category_aliases:
        return category_aliases[normalized_head], (parts[1].strip() if len(parts) > 1 else "")
    return "custom", stripped


def _append_instruction_line(existing_text: str, content: str, *, limit: int) -> str:
    existing_lines = _split_instruction_lines(existing_text, limit=0)
    existing_lines.append(content.strip())
    return "\n".join(f"- {line}" for line in _dedupe_lines(existing_lines)[:limit])


def _apply_user_setting(profile: UserProfileSnapshot, key: str, value: str) -> bool:
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
    if normalized_key == "identity_instructions":
        profile.identity_instructions = cleaned_value
        return True
    if normalized_key == "preference_instructions":
        profile.preference_instructions = cleaned_value
        return True
    if normalized_key == "custom_instructions":
        profile.custom_instructions = cleaned_value
        return True
    return False


def _strip_prefix_value(line: str) -> str:
    _, _, value = line.partition("：")
    if not value:
        _, _, value = line.partition(":")
    return value.strip()


def _get_user_profile_path(workspace: str) -> Path:
    return Path(workspace).resolve() / ".memory" / "USER.md"
