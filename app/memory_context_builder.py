from __future__ import annotations

from typing import Iterable

from app.memory_store import MemoryEntry, MemoryStore
from app.session import SessionData
from app.working_memory import WorkingMemory


def _shorten(text: str, max_chars: int) -> str:
    """把一段文本截短，避免注入 prompt 的内容过长。"""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _dedupe_memories(entries: Iterable[MemoryEntry]) -> list[MemoryEntry]:
    """对长期记忆做简单去重，避免重复注入。"""
    seen: set[str] = set()
    result: list[MemoryEntry] = []

    for entry in entries:
        normalized_content = " ".join(entry.content.strip().split())
        if not normalized_content:
            continue
        if normalized_content in seen:
            continue
        seen.add(normalized_content)
        result.append(entry)

    return result


def _format_long_term_memories(
    entries: list[MemoryEntry],
    max_items: int,
    max_chars_per_item: int,
) -> str:
    """把长期记忆格式化成适合注入 prompt 的文本。"""
    if not entries:
        return ""

    picked_entries = _dedupe_memories(entries)[:max_items]
    lines: list[str] = ["## 相关长期记忆"]

    for index, entry in enumerate(picked_entries, start=1):
        category = entry.category.strip() or "note"
        content = _shorten(entry.content, max_chars_per_item)

        tags_text = ""
        if entry.tags:
            tags_text = f" [tags: {', '.join(entry.tags)}]"

        lines.append(f"{index}. ({category}) {content}{tags_text}")

    return "\n".join(lines).strip()


def build_memory_context(
    *,
    user_input: str,
    session: SessionData,
    working_memory: WorkingMemory | None,
    memory_store: MemoryStore | None,
    session_summary_override: str = "",
    top_k: int = 5,
    max_summary_chars: int = 400,
    max_memory_chars_per_item: int = 180,
) -> str:
    """
    构建注入 prompt 的记忆上下文。

    这里拼的是：
    - older history summary
    - working memory
    - long-term memory

    而不是完整原始消息历史本身。
    """
    sections: list[str] = []

    # 优先使用当前这轮刚算出的旧历史摘要。
    session_summary = _shorten(session_summary_override, max_summary_chars)

    # 如果这轮没显式传入，就回退到 session.extra 里持久化的摘要缓存。
    if not session_summary:
        cached_summary = str(session.extra.get("older_history_summary", "")).strip()
        session_summary = _shorten(cached_summary, max_summary_chars)

    if session_summary:
        sections.append("## 会话摘要\n" + session_summary)

    if working_memory is not None:
        try:
            working_memory_text = working_memory.format_for_prompt().strip()
        except Exception:
            working_memory_text = ""

        if working_memory_text:
            sections.append("## 当前工作记忆\n" + working_memory_text)

    long_term_memory_text = ""
    if memory_store is not None:
        try:
            related_memories = memory_store.search_memories(
                query=user_input,
                top_k=top_k,
            )
        except Exception:
            related_memories = []

        long_term_memory_text = _format_long_term_memories(
            entries=related_memories,
            max_items=top_k,
            max_chars_per_item=max_memory_chars_per_item,
        )

    if long_term_memory_text:
        sections.append(long_term_memory_text)

    if not sections:
        return ""

    header = (
        "以下内容是从当前会话和本地记忆中整理出的辅助上下文。"
        "它用于帮助你保持任务连续性、记住约定和避免重复犯错。"
        "当这些内容与用户本轮明确要求冲突时，以用户本轮要求为准。"
    )

    return header + "\n\n" + "\n\n".join(sections)
