from __future__ import annotations

"""上下文信号解析模块，把运行状态转换成可用于决策的压缩信号。"""

import re

from app.memory.pipeline import MemoryPipeline
from app.memory.store import MemoryEntry
from app.state.user_profile import load_user_profile
from app.state.working_memory import WorkingMemory

_CONSTRAINT_CATEGORIES = {"constraint", "convention"}
_CONSTRAINT_TAGS = {
    "constraint",
    "context_management",
    "token_budget",
    "architecture",
    "convention",
}
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
)
_NOISE_PREFIXES = ("│", "├", "└", "dir ", "file ")
_MARKDOWN_TABLE_RE = re.compile(r"^\|.+\|$")
_CODE_LIKE_PREFIXES = ("def ", "class ", "return ", "import ", "from ", "if ", "for ", "while ")


def resolve_user_preferences(*, workspace: str) -> list[str]:
    """从 USER.md 解析当前会话应继承的稳定用户偏好。"""
    profile = load_user_profile(workspace)
    if profile is None:
        return []
    return profile.to_preference_lines()[:6]


def resolve_project_constraints(
    *,
    memory_pipeline: MemoryPipeline | None,
    working_memory: WorkingMemory,
) -> list[str]:
    """
    从 project scope 长时记忆筛选稳定约束，并补充少量本轮稳定约束。

    这里把 project memory 当成长期真相源，
    `working_memory.project_constraint` 只作为本轮补充层，但会过滤掉
    “最多 8 次工具调用 / 完成后立即停止” 这种当前执行指令。
    """
    lines: list[str] = []
    seen: set[str] = set()

    # 项目长期记忆仍然是压缩链路里的稳定约束主源，
    # working_memory.project_constraint 只补少量本轮新增但足够稳定的约束。
    max_lines = 10

    for entry in _load_project_constraint_entries(memory_pipeline):
        if not _looks_like_stable_project_constraint(entry.content):
            continue
        _append_unique_line(lines, seen, entry.content, limit=120)
        if len(lines) >= max_lines:
            break

    for entry in working_memory.get_entries_by_type("project_constraint"):
        if not _looks_like_stable_project_constraint(entry.content):
            continue
        _append_unique_line(lines, seen, entry.content, limit=120)
        if len(lines) >= max_lines:
            break

    return lines


def resolve_recent_risks(
    *,
    working_memory: WorkingMemory,
    cached_risks: list[str] | None = None,
) -> list[str]:
    """提取当前 session 最近需要记住的风险信号。"""
    lines: list[str] = []
    seen: set[str] = set()

    for entry in working_memory.get_entries_by_type("recent_risk"):
        if not _looks_like_recent_risk(entry.content):
            continue
        _append_unique_line(lines, seen, entry.content, limit=120)
        if len(lines) >= 5:
            break

    for entry in working_memory.get_entries_by_type("error_context"):
        if not _looks_like_recent_risk(entry.content):
            continue
        _append_unique_line(lines, seen, entry.content, limit=120)
        if len(lines) >= 5:
            break

    for item in cached_risks or []:
        if not _looks_like_recent_risk(item):
            continue
        _append_unique_line(lines, seen, item, limit=120)
        if len(lines) >= 5:
            break

    return lines


def _load_project_constraint_entries(memory_pipeline: MemoryPipeline | None) -> list[MemoryEntry]:
    """从 memory pipeline 挂着的 memory store 里筛选高置信项目约束。"""
    if memory_pipeline is None:
        return []

    read_pipeline = getattr(memory_pipeline, "read_pipeline", None)
    memory_store = getattr(read_pipeline, "memory_store", None)
    if memory_store is None:
        return []

    try:
        entries = memory_store.filter_memories(scope="project")
    except Exception:
        return []

    candidates: list[MemoryEntry] = []
    for entry in entries:
        if entry.archived:
            continue
        if float(entry.confidence) < 0.80:
            continue

        normalized_category = entry.category.strip().lower()
        normalized_tags = {
            tag.strip().lower()
            for tag in entry.tags
            if tag.strip()
        }
        if normalized_category not in _CONSTRAINT_CATEGORIES and not (
            normalized_tags & _CONSTRAINT_TAGS
        ):
            continue

        candidates.append(entry)

    candidates.sort(
        key=lambda item: (
            float(item.confidence),
            int(item.usage_count),
            float(item.updated_at),
        ),
        reverse=True,
    )
    return candidates


def _append_unique_line(lines: list[str], seen: set[str], raw_text: str, *, limit: int) -> None:
    """归一化去重后追加一条短文本。"""
    line = _shorten(raw_text, limit)
    normalized = " ".join(line.strip().lower().split())
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    lines.append(line)


def _looks_like_stable_project_constraint(text: str) -> bool:
    """只保留更像长期项目约束的内容，过滤当前轮执行指令。"""
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
    """过滤不应进入 recent_risk 的目录树、表格和代码正文。"""
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
            "tool_result",
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
    """过滤目录树、Markdown 表格和代码正文噪音。"""
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
    """裁短快照条目，避免 resolved 字段本身过长。"""
    normalized = " ".join(str(text).strip().split())
    if len(normalized) <= max_chars:
        return normalized
    suffix = " ...[已截断]"
    head_limit = max(0, max_chars - len(suffix))
    return f"{normalized[:head_limit]}{suffix}"
