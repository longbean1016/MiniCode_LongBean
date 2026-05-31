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
    if normalized_scope == "user":
        return 2
    if normalized_scope == "project":
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

        # 去重时优先相信稳定 id。
        # 没有 id 时再回退到归一化正文，避免同一条记忆被不同召回路径重复注入。
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
    # 还会吸收会话压缩基线和 working memory 里的活跃线索。
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

    normalized_query = _normalize_text(query_text)

    tag_score = 0.0
    for tag in entry.tags:
        normalized_tag = _normalize_text(tag)
        if normalized_tag and normalized_tag in normalized_query:
            tag_score += 0.08

    domain_score = 0.0
    for domain in entry.domains:
        normalized_domain = _normalize_text(domain)
        if normalized_domain and normalized_domain in normalized_query:
            domain_score += 0.06

    confidence_score = max(0.0, min(1.0, float(entry.confidence))) * 0.25
    decay_score = max(0.0, min(1.0, float(entry.decay_score))) * 0.15
    usage_score = min(0.18, math.log1p(max(0, int(entry.usage_count))) * 0.06)

    age_seconds = max(0.0, time.time() - float(entry.updated_at))
    recency_score = max(0.0, 1.0 - age_seconds / (30 * 86400)) * 0.12

    normalized_scope = _normalize_text(entry.scope)
    if normalized_scope == "user":
        scope_bonus = 0.08
    elif normalized_scope == "project":
        scope_bonus = 0.05
    else:
        scope_bonus = 0.02

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
            injected_parts.append(f"{entry.id}:{entry.scope}:{entry.category}:score={score:.3f}")
        lines.append("[memory-retrieval] injected=" + " | ".join(injected_parts))
    else:
        lines.append("[memory-retrieval] injected=(none)")

    return lines


def _format_memory_section(
    title: str,
    entries: list[MemoryEntry],
    *,
    max_chars_per_item: int,
) -> str:
    if not entries:
        return ""

    lines = [f"## {title}"]
    for index, entry in enumerate(entries, start=1):
        content = _shorten(entry.content, max_chars_per_item)
        lines.append(f"{index}. ({entry.scope}/{entry.category}) {content}")
    return "\n".join(lines).strip()


def _split_memories_by_scope(
    entries: list[MemoryEntry],
) -> tuple[list[MemoryEntry], list[MemoryEntry], list[MemoryEntry]]:
    user_entries: list[MemoryEntry] = []
    project_entries: list[MemoryEntry] = []
    other_entries: list[MemoryEntry] = []

    for entry in entries:
        normalized_scope = _normalize_text(entry.scope)
        if normalized_scope == "user":
            user_entries.append(entry)
        elif normalized_scope == "project":
            project_entries.append(entry)
        else:
            other_entries.append(entry)

    return user_entries, project_entries, other_entries


def _split_user_memory_tiers(
    entries: list[MemoryEntry],
) -> tuple[list[MemoryEntry], list[MemoryEntry]]:
    pinned_user_entries: list[MemoryEntry] = []
    retrieved_user_entries: list[MemoryEntry] = []

    for entry in entries:
        if is_pinned_memory_entry(entry):
            pinned_user_entries.append(entry)
        else:
            retrieved_user_entries.append(entry)

    return pinned_user_entries, retrieved_user_entries


def _select_injected_entries(
    *,
    pinned_entries: list[MemoryEntry],
    picked_entries: list[MemoryEntry],
    covered_text: str,
    top_k: int,
) -> list[MemoryEntry]:
    # 这里显式做一次“注入层预算分配”：
    # 1. 固定记忆永远优先。
    # 2. 用户长期记忆至少保一条检索结果，避免 tight budget 时只剩项目记忆。
    # 3. 剩余预算再按原有排序结果补齐。
    filtered_pinned_entries = [
        entry for entry in _dedupe_memories(pinned_entries)
        if not _is_memory_already_covered(entry, covered_text)
    ]
    filtered_picked_entries = [
        entry for entry in _dedupe_memories(picked_entries)
        if not _is_memory_already_covered(entry, covered_text)
    ]
    if top_k <= 0:
        return filtered_pinned_entries + filtered_picked_entries

    remaining_budget = max(0, top_k - len(filtered_pinned_entries))
    if remaining_budget <= 0:
        return filtered_pinned_entries

    user_candidates = [
        entry for entry in filtered_picked_entries
        if _normalize_text(entry.scope) == "user"
    ]
    project_candidates = [
        entry for entry in filtered_picked_entries
        if _normalize_text(entry.scope) == "project"
    ]
    other_candidates = [
        entry for entry in filtered_picked_entries
        if _normalize_text(entry.scope) not in {"user", "project"}
    ]

    selected_entries: list[MemoryEntry] = list(filtered_pinned_entries)

    # 用户长期记忆是用户稳定偏好/习惯的主来源，优先保留至少一条。
    if user_candidates and remaining_budget > 0:
        selected_entries.append(user_candidates.pop(0))
        remaining_budget -= 1

    for bucket in (user_candidates, project_candidates, other_candidates):
        while bucket and remaining_budget > 0:
            selected_entries.append(bucket.pop(0))
            remaining_budget -= 1

    return _dedupe_memories(selected_entries)


def _build_context_dedupe_text(
    *,
    session_summary: str,
    working_memory_text: str,
) -> str:
    # active_context_summary 是当前压缩体系里的会话主基线，
    # working memory 只保护少量高价值槽位。
    # 这里把两者拼成“已覆盖文本基线”，避免重复注入长期记忆。
    parts = [part.strip() for part in (session_summary, working_memory_text) if part.strip()]
    return _normalize_text("\n".join(parts))


def _is_memory_already_covered(entry: MemoryEntry, covered_text: str) -> bool:
    if not covered_text:
        return False

    normalized_content = _normalize_text(entry.content)
    if not normalized_content:
        return True

    return normalized_content in covered_text


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
            # active_context_summary 是当前压缩体系里的会话主基线，
            # 长期记忆和 working memory 都只作为它的补充，而不是平级竞争者。
            sections.append("## 当前会话压缩基线\n" + session_summary)

        working_memory_text = ""
        if working_memory is not None:
            try:
                working_memory_text = _build_working_memory_brief(
                    working_memory,
                    max_chars_per_item=min(max_memory_chars_per_item, 120),
                ).strip()
            except Exception:
                working_memory_text = ""

        if self.memory_store is None:
            if working_memory_text:
                sections.append("## 当前工作记忆\n" + working_memory_text)
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
            # 先尽量保留更完整的候选集合，真正的注入预算分配放到后面统一处理，
            # 这样 tight budget 时仍有机会优先保住一条相关用户长期记忆。
            max_items=max(top_k, retrieval_top_k),
        )

        debug_lines = _build_memory_debug_lines(
            query_text=query_text,
            related_memories=related_entries,
            pinned_memories=pinned_entries,
            picked_memories=picked_entries,
        )
        for line in debug_lines:
            log_event(line, echo=False)

        covered_text = _build_context_dedupe_text(
            session_summary=session_summary,
            working_memory_text=working_memory_text,
        )
        injected_entries = _select_injected_entries(
            pinned_entries=pinned_entries,
            picked_entries=picked_entries,
            covered_text=covered_text,
            top_k=top_k,
        )
        if injected_entries:
            # 标记“本轮被读过”，但真正的好坏反馈要等回合结束才能知道。
            self.memory_store.mark_memories_accessed(
                [entry.id for entry in injected_entries if entry.id]
            )

        user_entries, project_entries, other_entries = _split_memories_by_scope(injected_entries)
        pinned_user_entries, retrieved_user_entries = _split_user_memory_tiers(user_entries)
        for title, entries in (
            ("固定用户长期记忆", pinned_user_entries),
            ("相关用户长期记忆", retrieved_user_entries),
            ("项目长期记忆", project_entries),
            ("其他长期记忆", other_entries),
        ):
            section_text = _format_memory_section(
                title,
                entries,
                max_chars_per_item=max_memory_chars_per_item,
            )
            if section_text:
                sections.append(section_text)

        if working_memory_text:
            sections.append("## 当前工作记忆\n" + working_memory_text)

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
