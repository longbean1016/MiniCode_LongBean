from __future__ import annotations

import math
import re
import time
from typing import Iterable

from app.explicit_memory import is_pinned_memory_entry
from app.logger import log_event
from app.memory_models import MemoryContextResult
from app.memory_store import MemoryEntry, MemoryStore
from app.session import SessionData
from app.working_memory import WorkingMemory

# 读取链路只负责挑选和格式化本轮要注入的上下文，
# 不负责把新内容写回长期记忆。


def _shorten(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _tokenize(text: str) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized))


def _scope_priority(scope: str) -> int:
    normalized_scope = _normalize_text(scope)
    if normalized_scope == "project":
        return 2
    if normalized_scope == "user":
        return 1
    return 0


def _dedupe_memories(entries: Iterable[MemoryEntry]) -> list[MemoryEntry]:
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    result: list[MemoryEntry] = []

    for entry in entries:
        if entry.id and entry.id in seen_ids:
            continue

        normalized_content = _normalize_text(entry.content)
        if not normalized_content or normalized_content in seen_content:
            continue

        # 这里的去重优先信任稳定 id；
        # 没有 id 时再退回到归一化后的正文，避免同一条记忆被不同召回通路重复注入。
        if entry.id:
            seen_ids.add(entry.id)
        seen_content.add(normalized_content)
        result.append(entry)

    return result


def _build_memory_query(
    *,
    user_input: str,
    session_summary: str,
    working_memory: WorkingMemory | None,
) -> str:
    # 检索查询不只来自当前一句用户输入，
    # 还会吸收会话摘要和 working memory 里的活跃线索。
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
    # 这里评估的是“是否值得直接塞进 prompt”的综合价值，
    # 不是严格意义上的语义相似度。
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
    scope_bonus = 0.08 if _normalize_text(entry.scope) == "project" else 0.04

    return (
        overlap_score * 1.6
        + tag_score
        + domain_score
        + confidence_score
        + decay_score
        + usage_score
        + recency_score
        + scope_bonus
    )


def _rerank_related_memories(
    entries: list[MemoryEntry],
    *,
    query_text: str,
    max_items: int,
) -> list[MemoryEntry]:
    # 底层检索只给候选，这里再按注入价值做二次排序。
    scored_entries = [
        (_score_memory_for_injection(entry, query_text), entry)
        for entry in _dedupe_memories(entries)
    ]
    scored_entries.sort(
        key=lambda item: (item[0], _scope_priority(item[1].scope), item[1].updated_at),
        reverse=True,
    )
    return [entry for _, entry in scored_entries[:max_items]]


def _build_memory_debug_lines(
    *,
    query_text: str,
    related_memories: list[MemoryEntry],
    pinned_memories: list[MemoryEntry],
    picked_memories: list[MemoryEntry],
) -> list[str]:
    # debug_lines 不是给模型看的，而是给检索调试看的：
    # 方便复盘“为什么这轮注入了这些记忆，而不是另外一些”。
    lines: list[str] = []
    lines.append(f"[memory-retrieval] query={_shorten(query_text, 400)}")

    if related_memories:
        candidate_parts: list[str] = []
        for entry in related_memories[:8]:
            score = _score_memory_for_injection(entry, query_text)
            candidate_parts.append(
                f"{entry.id}:{entry.scope}:{entry.category}:score={score:.3f}:archived={entry.archived}"
            )
        lines.append("[memory-retrieval] candidates=" + " | ".join(candidate_parts))
    else:
        lines.append("[memory-retrieval] candidates=(none)")

    if pinned_memories:
        pinned_parts = [f"{entry.id}:{entry.scope}:{entry.category}" for entry in pinned_memories]
        lines.append("[memory-retrieval] pinned=" + " | ".join(pinned_parts))
    else:
        lines.append("[memory-retrieval] pinned=(none)")

    if picked_memories:
        injected_parts: list[str] = []
        for entry in picked_memories:
            score = _score_memory_for_injection(entry, query_text)
            injected_parts.append(
                f"{entry.id}:{entry.scope}:{entry.category}:score={score:.3f}"
            )
        lines.append("[memory-retrieval] injected=" + " | ".join(injected_parts))
    else:
        lines.append("[memory-retrieval] injected=(none)")

    return lines


def _format_pinned_memories(entries: list[MemoryEntry], *, max_chars_per_item: int) -> str:
    if not entries:
        return ""

    lines = ["## 固定记忆"]
    for index, entry in enumerate(entries, start=1):
        content = _shorten(entry.content, max_chars_per_item)
        lines.append(f"{index}. ({entry.scope}/{entry.category}) {content}")
    return "\n".join(lines).strip()


def _format_long_term_memories(entries: list[MemoryEntry], max_chars_per_item: int) -> str:
    if not entries:
        return ""

    lines = ["## 相关长期记忆"]
    for index, entry in enumerate(entries, start=1):
        content = _shorten(entry.content, max_chars_per_item)
        lines.append(f"{index}. ({entry.scope}/{entry.category}) {content}")
    return "\n".join(lines).strip()


def _build_working_memory_brief(
    working_memory: WorkingMemory | None,
    *,
    max_chars_per_item: int,
) -> str:
    """
    只保留少量高价值 working memory。
    新链路里它只是辅助信号，不再承担旧版“大段会话基线”的职责。
    """
    if working_memory is None:
        return ""

    slot_specs = (
        ("当前任务", "active_task", 2),
        ("关键决策", "key_decision", 2),
        ("最近风险", "recent_risk", 2),
        ("错误上下文", "error_context", 1),
    )
    lines: list[str] = []
    for title, entry_type, limit in slot_specs:
        entries = working_memory.get_entries_by_type(entry_type)[-limit:]
        if not entries:
            continue
        lines.append(f"## {title}")
        for entry in entries:
            content = _shorten(entry.content, max_chars_per_item)
            if content:
                lines.append(f"- {content}")
    return "\n".join(lines).strip()


class MemoryReadPipeline:
    def __init__(
        self,
        memory_store: MemoryStore | None,
        *,
        retrieval_scopes: tuple[str, ...] = ("project", "user"),
        pinned_limit: int = 6,
    ) -> None:
        self.memory_store = memory_store
        self.retrieval_scopes = retrieval_scopes
        self.pinned_limit = max(0, pinned_limit)

    def build_context(
        self,
        *,
        user_input: str,
        session: SessionData,
        working_memory: WorkingMemory | None,
        session_summary_override: str = "",
        top_k: int = 4,
        retrieval_top_k: int = 8,
        max_summary_chars: int = 400,
        max_memory_chars_per_item: int = 180,
    ) -> MemoryContextResult:
        sections: list[str] = []
        session_summary = _shorten(session_summary_override, max_summary_chars)
        if not session_summary:
            cached_active_summary = str(session.extra.get("active_context_summary", "")).strip()
            session_summary = _shorten(cached_active_summary, max_summary_chars)
        if not session_summary:
            cached_summary = str(session.extra.get("older_history_summary", "")).strip()
            session_summary = _shorten(cached_summary, max_summary_chars)

        if session_summary:
            sections.append("## 会话摘要\n" + session_summary)

        if working_memory is not None:
            try:
                working_memory_text = _build_working_memory_brief(
                    working_memory,
                    max_chars_per_item=min(max_memory_chars_per_item, 120),
                ).strip()
            except Exception:
                working_memory_text = ""

            if working_memory_text:
                sections.append("## 当前工作记忆\n" + working_memory_text)

        if self.memory_store is None:
            return MemoryContextResult(prompt_context=self._build_prompt(sections))

        query_text = _build_memory_query(
            user_input=user_input,
            session_summary=session_summary,
            working_memory=working_memory,
        )
        pinned_entries = self._pick_pinned_entries()
        pinned_ids = {entry.id for entry in pinned_entries if entry.id}
        # 固定记忆先占预算，剩余名额再给动态检索结果。
        retrieval_budget = max(0, top_k - len(pinned_entries))
        related_entries = self._search_related_entries(
            query_text=query_text,
            top_k=max(top_k, retrieval_top_k),
        )
        picked_entries = _rerank_related_memories(
            [entry for entry in related_entries if entry.id not in pinned_ids],
            query_text=query_text,
            max_items=retrieval_budget,
        )

        debug_lines = _build_memory_debug_lines(
            query_text=query_text,
            related_memories=related_entries,
            pinned_memories=pinned_entries,
            picked_memories=picked_entries,
        )
        for line in debug_lines:
            log_event(line, echo=False)

        injected_entries = _dedupe_memories([*pinned_entries, *picked_entries])
        if injected_entries:
            # 标记“本轮被读过”，但真正的好坏反馈要等回合结束才能知道。
            self.memory_store.mark_memories_accessed(
                [entry.id for entry in injected_entries if entry.id]
            )

        pinned_text = _format_pinned_memories(
            pinned_entries,
            max_chars_per_item=max_memory_chars_per_item,
        )
        related_text = _format_long_term_memories(
            picked_entries,
            max_chars_per_item=max_memory_chars_per_item,
        )

        if pinned_text:
            sections.append(pinned_text)
        if related_text:
            sections.append(related_text)

        return MemoryContextResult(
            prompt_context=self._build_prompt(sections),
            query_text=query_text,
            injected_entries=injected_entries,
            pinned_entries=pinned_entries,
            retrieved_entries=picked_entries,
            debug_lines=debug_lines,
        )

    def _pick_pinned_entries(self) -> list[MemoryEntry]:
        if self.memory_store is None:
            return []

        try:
            all_entries = self.memory_store.load_memories()
        except Exception:
            return []

        pinned_entries = [
            entry
            for entry in all_entries
            if is_pinned_memory_entry(entry)
            and _normalize_text(entry.scope) in self.retrieval_scopes
        ]
        pinned_entries.sort(
            # pinned 更像长期硬约束，因此优先看 scope，再看新旧和置信度。
            key=lambda entry: (
                _scope_priority(entry.scope),
                float(entry.updated_at),
                float(entry.confidence),
            ),
            reverse=True,
        )
        return pinned_entries[: self.pinned_limit]

    def _search_related_entries(self, *, query_text: str, top_k: int) -> list[MemoryEntry]:
        if self.memory_store is None or not query_text.strip():
            return []

        result: list[MemoryEntry] = []
        for scope in self.retrieval_scopes:
            try:
                # 分 scope 搜索可以减少 user / project 之间的互相挤占。
                scope_entries = self.memory_store.search_memories(
                    query=query_text,
                    top_k=top_k,
                    scope=scope,
                    include_archived=False,
                    mark_access=False,
                )
            except Exception:
                scope_entries = []
            result.extend(scope_entries)

        return _dedupe_memories(result)

    def _build_prompt(self, sections: list[str]) -> str:
        filtered_sections = [section.strip() for section in sections if section.strip()]
        if not filtered_sections:
            return ""

        # 明确告诉模型：这些只是辅助上下文，和本轮显式要求冲突时以后者为准。
        header = (
            "以下内容是从当前会话和本地记忆中整理出的辅助上下文。"
            "它用于帮助你保持任务连续性、遵守长期约定，并避免重复犯错。"
            "如果与用户本轮的明确要求冲突，以用户本轮要求为准。"
        )
        return header + "\n\n" + "\n\n".join(filtered_sections)
