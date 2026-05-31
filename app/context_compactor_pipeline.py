from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.context_compact_memory import (
    CompactMemorySnapshot,
    parse_compact_memory_context,
    render_compact_memory_context,
)
from app.context_auto_compact import (
    AUTO_COMPACT_TARGET_RATIO,
    AutoCompactResult,
    run_auto_compact,
)
from app.context_compactor import CompactionResult, compact_recent_messages
from app.context_manager import estimate_messages_tokens
from app.context_message_safety import normalize_tool_call_pairs
from app.types import ChatMessage


@dataclass(slots=True)
class ContextPipelineResult:
    """统一返回每轮请求前的上下文治理结果。"""

    messages: list[ChatMessage]
    compaction_result: CompactionResult
    auto_compact_result: AutoCompactResult
    steps_taken: list[str] = field(default_factory=list)
    compaction_history_entry: dict[str, Any] = field(default_factory=dict)
    resolved_compact_memory_snapshot: CompactMemorySnapshot = field(default_factory=dict)
    resolved_compact_memory_context: str = ""


class ContextCompactorPipeline:
    """
    每轮请求前的上下文处理入口。

    当前顺序尽量贴近 minicode：
    1. Tool Result Budget
    2. Read Dedup
    3. Microcompact 风格的旧 tool_result 清理
    4. Auto Compact Dispatcher
       - Session Memory Compact
       - Full Compact
    """

    def process_request(
        self,
        *,
        messages: list[ChatMessage],
        summary_source_messages: list[ChatMessage] | None = None,
        max_recent_tool_results: int,
        truncate_tool_result_chars: int,
        workspace: str,
        usable_budget: int,
        fixed_overhead_tokens: int,
        auto_compact_summary: str,
        auto_compact_snapshot: CompactMemorySnapshot | None = None,
        force_auto_compact: bool = False,
        pinned_tool_names: set[str] | None = None,
    ) -> ContextPipelineResult:
        # 先做轻量 recent-window 压缩，把最肥的 tool_result 处理掉。
        compaction_result = compact_recent_messages(
            messages,
            max_recent_tool_results=max_recent_tool_results,
            truncate_tool_result_chars=truncate_tool_result_chars,
            workspace=workspace,
            pinned_tool_names=pinned_tool_names,
            target_tokens=max(1, int(usable_budget * AUTO_COMPACT_TARGET_RATIO) - fixed_overhead_tokens),
            protected_recent_messages=6,
        )
        steps_taken = [
            "tool_budget",
            "read_dedup",
            "recent_tool_cleanup",
        ]

        # recent window 压完后，再根据真实预算压力决定是否进入 Auto Compact。
        recent_tokens_after_compaction = estimate_messages_tokens(compaction_result.messages)
        auto_compact_result = run_auto_compact(
            messages=compaction_result.messages,
            usable_budget=usable_budget,
            summary_base=auto_compact_summary,
            summary_snapshot=auto_compact_snapshot,
            summary_source_messages=summary_source_messages or messages,
            fixed_overhead_tokens=fixed_overhead_tokens,
            force_full=force_auto_compact,
        )
        if auto_compact_result.applied:
            steps_taken.append(f"auto_compact:{auto_compact_result.strategy}")

        history_entry = {
            "truncated_tool_results": compaction_result.truncated_tool_results,
            "cleared_old_tool_results": compaction_result.cleared_old_tool_results,
            "deduped_read_results": compaction_result.deduped_read_results,
            "semantic_compacted_pairs": compaction_result.semantic_compacted_pairs,
            "dropped_progress_messages": compaction_result.dropped_progress_messages,
            "priority_dropped_messages": compaction_result.priority_dropped_messages,
            "recent_tokens_after_compaction": recent_tokens_after_compaction,
            "tokens_freed_estimate": (
                compaction_result.tokens_freed_estimate + auto_compact_result.tokens_freed_estimate
            ),
            "auto_compact_applied": auto_compact_result.applied,
            "auto_compact_strategy": auto_compact_result.strategy,
            "auto_compact_tokens_before": auto_compact_result.tokens_before,
            "auto_compact_tokens_after": auto_compact_result.tokens_after,
        }
        output_messages = auto_compact_result.messages if auto_compact_result.applied else compaction_result.messages
        normalized_messages = normalize_tool_call_pairs(output_messages)
        resolved_snapshot = auto_compact_snapshot or parse_compact_memory_context(auto_compact_summary)
        resolved_context = auto_compact_summary.strip()
        if auto_compact_result.summary_snapshot:
            resolved_snapshot = auto_compact_result.summary_snapshot
            resolved_context = (
                auto_compact_result.summary_text.strip()
                or render_compact_memory_context(resolved_snapshot)
            )
        elif resolved_snapshot and not resolved_context:
            resolved_context = render_compact_memory_context(resolved_snapshot)
        return ContextPipelineResult(
            messages=normalized_messages,
            compaction_result=compaction_result,
            auto_compact_result=auto_compact_result,
            steps_taken=steps_taken,
            compaction_history_entry=history_entry,
            resolved_compact_memory_snapshot=resolved_snapshot,
            resolved_compact_memory_context=resolved_context,
        )
