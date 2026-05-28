from __future__ import annotations

import math
import re
import time
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


def _normalize_text(text: str) -> str:
    """规范化文本空白和大小写，便于本地打分。"""
    return " ".join(str(text).strip().lower().split())


def _tokenize(text: str) -> set[str]:
    """
    对查询和记忆内容做轻量切词。

    当前规则：
    - 英文/数字按连续单词切
    - 中文按单字切
    """
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized))


def _dedupe_memories(entries: Iterable[MemoryEntry]) -> list[MemoryEntry]:
    """对长期记忆做简单去重，避免重复注入。"""
    seen: set[str] = set()
    result: list[MemoryEntry] = []

    for entry in entries:
        normalized_content = _normalize_text(entry.content)
        if not normalized_content or normalized_content in seen:
            continue
        seen.add(normalized_content)
        result.append(entry)

    return result


def _build_memory_query(
    *,
    user_input: str,
    session_summary: str,
    working_memory: WorkingMemory | None,
) -> str:
    """
    构造长期记忆检索查询。

    这里不只用用户原始输入，而是把当前意图、活跃任务、关键决策拼成一段查询，
    这样更容易把“当前真正相关”的 project memory 召回出来。
    """
    parts: list[str] = []

    cleaned_user_input = user_input.strip()
    if cleaned_user_input:
        parts.append(f"user_input: {cleaned_user_input}")

    if session_summary.strip():
        parts.append(f"session_summary: {_shorten(session_summary, 220)}")

    if working_memory is not None:
        active_tasks = [
            entry.content for entry in working_memory.get_entries_by_type("active_task")[-4:]
        ]
        if active_tasks:
            parts.append("active_tasks: " + " | ".join(active_tasks))

        key_decisions = [
            entry.content for entry in working_memory.get_entries_by_type("key_decision")[-4:]
        ]
        if key_decisions:
            parts.append("key_decisions: " + " | ".join(key_decisions))

        error_contexts = [
            entry.content for entry in working_memory.get_entries_by_type("error_context")[-2:]
        ]
        if error_contexts:
            parts.append("error_contexts: " + " | ".join(error_contexts))

    return "\n".join(parts).strip()


def _score_memory_for_injection(entry: MemoryEntry, query_text: str) -> float:
    """
    为候选长期记忆计算注入分数。

    这一步是检索后的轻量 rerank，目标是：
    - 优先保留与当前查询真正相关的记忆
    - 再参考 confidence / decay / recency / usage 做细排序
    """
    query_tokens = _tokenize(query_text)
    content_tokens = _tokenize(entry.content)

    overlap_score = 0.0
    if query_tokens and content_tokens:
        overlap_score = len(query_tokens & content_tokens) / max(1, len(query_tokens))

    tag_score = 0.0
    for tag in entry.tags:
        normalized_tag = _normalize_text(tag)
        if normalized_tag and normalized_tag in _normalize_text(query_text):
            tag_score += 0.08

    domain_score = 0.0
    for domain in entry.domains:
        normalized_domain = _normalize_text(domain)
        if normalized_domain and normalized_domain in _normalize_text(query_text):
            domain_score += 0.06

    confidence_score = max(0.0, min(1.0, float(entry.confidence))) * 0.25
    decay_score = max(0.0, min(1.0, float(entry.decay_score))) * 0.15

    usage_score = min(0.18, math.log1p(max(0, int(entry.usage_count))) * 0.06)

    age_seconds = max(0.0, time.time() - float(entry.updated_at))
    recency_score = max(0.0, 1.0 - age_seconds / (30 * 86400)) * 0.12

    return overlap_score * 1.6 + tag_score + domain_score + confidence_score + decay_score + usage_score + recency_score


def _rerank_related_memories(
    entries: list[MemoryEntry],
    *,
    query_text: str,
    max_items: int,
) -> list[MemoryEntry]:
    """
    对候选长期记忆做本地 rerank。

    这里不会再把 archived 记忆放回来；
    它只在“当前有效记忆集合”里重新选出最值得注入 prompt 的少量条目。
    """
    deduped_entries = _dedupe_memories(entries)
    scored_entries = [
        (_score_memory_for_injection(entry, query_text), entry)
        for entry in deduped_entries
    ]
    scored_entries.sort(
        key=lambda item: (item[0], item[1].updated_at),
        reverse=True,
    )
    return [entry for _, entry in scored_entries[:max_items]]


def _format_long_term_memories(
    entries: list[MemoryEntry],
    max_chars_per_item: int,
) -> str:
    """把长期记忆格式化成适合注入 prompt 的文本。"""
    if not entries:
        return ""

    lines: list[str] = ["## 相关长期记忆"]

    for index, entry in enumerate(entries, start=1):
        category = entry.category.strip() or "note"
        content = _shorten(entry.content, max_chars_per_item)

        tag_text = ""
        if entry.tags:
            tag_text = f" [tags: {', '.join(entry.tags[:4])}]"

        lines.append(f"{index}. ({category}) {content}{tag_text}")

    return "\n".join(lines).strip()


def build_memory_context(
    *,
    user_input: str,
    session: SessionData,
    working_memory: WorkingMemory | None,
    memory_store: MemoryStore | None,
    session_summary_override: str = "",
    top_k: int = 4,
    retrieval_top_k: int = 8,
    max_summary_chars: int = 400,
    max_memory_chars_per_item: int = 180,
) -> str:
    """
    构建注入 prompt 的记忆上下文。

    这里拼的是：
    - older history summary
    - working memory
    - long-term memory

    设计目标：
    - 只注入少量高相关、当前有效的长期记忆
    - 默认排除 archived / superseded 旧版本
    - 先召回，再本地 rerank，最后再注入
    """
    sections: list[str] = []

    # 优先使用当前这轮刚算出来的旧历史摘要。
    session_summary = _shorten(session_summary_override, max_summary_chars)

    # 如果这轮没显式传入，就回退到 session.extra 里的持久化摘要缓存。
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
        query_text = _build_memory_query(
            user_input=user_input,
            session_summary=session_summary,
            working_memory=working_memory,
        )

        try:
            # 当前自动主链路只写 project scope，因此这里默认也只检索 project。
            # 先多召回一点候选，再做本地 rerank，减少“只靠 user_input 一次命中”的粗糙感。
            related_memories = memory_store.search_memories(
                query=query_text,
                top_k=max(top_k, retrieval_top_k),
                scope="project",
                include_archived=False,
                mark_access=False,
            )
        except Exception:
            related_memories = []

        picked_memories = _rerank_related_memories(
            related_memories,
            query_text=query_text,
            max_items=top_k,
        )

        # 只有真正被注入 prompt 的记忆，才标记为“被访问”。
        if picked_memories:
            try:
                memory_store.mark_memories_accessed([entry.id for entry in picked_memories])
            except Exception:
                pass

        long_term_memory_text = _format_long_term_memories(
            entries=picked_memories,
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
