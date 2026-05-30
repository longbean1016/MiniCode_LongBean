from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_KV_RE = re.compile(r"^-\s+\*\*(.+?)\*\*:\s*(.+)$")


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

        if self.custom_instructions.strip():
            lines.extend(_split_instruction_lines(self.custom_instructions))

        return _dedupe_lines(lines)


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
