from __future__ import annotations

"""上下文信号解析模块，把持久记忆快照转换成可注入的压缩信号。"""

import re

from app.memory.memory_store import FrozenMemorySnapshot

_TRANSIENT_CONSTRAINT_PHRASES = (
    "本次",
    "这次",
    "本轮",
    "验收测试",
    "工具调用",
    "完成后",
    "立即停止",
    "不要继续探索",
    "写文件",
    "写入 tmp",
    "tmp/",
    "先读取",
    "最后只回复",
    "最多允许",
)
_STABLE_CONSTRAINT_MARKERS = (
    "默认",
    "统一",
    "所有",
    "长期",
    "约定",
    "必须",
    "不能",
    "不要",
    "只按",
    "优先",
)
_PROJECT_CONSTRAINT_ANCHORS = (
    ".py",
    "代码",
    "文件",
    "模块",
    "接口",
    "上下文",
    "token",
    "压缩",
    "python",
    "测试",
)
_NOISE_PREFIXES = ("│", "├", "└", "dir ", "file ")
_MARKDOWN_TABLE_RE = re.compile(r"^\|.+\|$")
_CODE_LIKE_PREFIXES = ("def ", "class ", "return ", "import ", "from ", "if ", "for ", "while ")


def resolve_user_preferences(*, workspace: str) -> list[str]:
    # USER.md 现在由 MemoryStore 以 Markdown 列表格式管理，
    # 用户偏好通过 MemoryStore.get_prompt_context() 注入到持久记忆段
    return []


def resolve_project_constraints(
    *,
    memory_snapshot: FrozenMemorySnapshot | None,
) -> list[str]:
    if memory_snapshot is None:
        return []

    lines: list[str] = []
    seen: set[str] = set()
    for item in memory_snapshot.memory_entries:
        if not _looks_like_stable_project_constraint(item):
            continue
        _append_unique_line(lines, seen, item, limit=120)
        if len(lines) >= 10:
            break
    return lines


def resolve_recent_risks(
    *,
    cached_risks: list[str] | None = None,
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for item in cached_risks or []:
        if not _looks_like_recent_risk(item):
            continue
        _append_unique_line(lines, seen, item, limit=120)
        if len(lines) >= 5:
            break
    return lines


def _append_unique_line(lines: list[str], seen: set[str], raw_text: str, *, limit: int) -> None:
    line = _shorten(raw_text, limit)
    normalized = " ".join(line.strip().lower().split())
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    lines.append(line)


def _looks_like_stable_project_constraint(text: str) -> bool:
    normalized = " ".join(str(text).strip().lower().split())
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _TRANSIENT_CONSTRAINT_PHRASES):
        return False
    if _looks_like_structured_noise(normalized):
        return False
    has_stable_marker = any(marker in normalized for marker in _STABLE_CONSTRAINT_MARKERS)
    has_project_anchor = any(anchor in normalized for anchor in _PROJECT_CONSTRAINT_ANCHORS)
    return has_stable_marker or has_project_anchor


def _looks_like_recent_risk(text: str) -> bool:
    normalized = " ".join(str(text).strip().split())
    if not normalized:
        return False
    if _looks_like_structured_noise(normalized):
        return False
    return any(
        marker in normalized.lower()
        for marker in (
            "风险",
            "失败",
            "error",
            "warning",
            "超长",
            "超限",
            "上下文",
            "token",
            "budget",
            "挤占",
            "冲突",
            "阻塞",
            "异常",
        )
    )


def _looks_like_structured_noise(text: str) -> bool:
    normalized = str(text).strip()
    if not normalized:
        return True
    lower_line = normalized.lower()
    if lower_line.startswith(_NOISE_PREFIXES):
        return True
    if _MARKDOWN_TABLE_RE.match(normalized):
        return True
    if lower_line.startswith(_CODE_LIKE_PREFIXES):
        return True
    if normalized.count("|") >= 4:
        return True
    return False


def _shorten(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text).strip().split())
    if len(normalized) <= max_chars:
        return normalized
    suffix = " ...[已截断]"
    return normalized[: max(0, max_chars - len(suffix))] + suffix
