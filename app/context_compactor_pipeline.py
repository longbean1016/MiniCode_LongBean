from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.context_auto_compact import (
    AUTO_COMPACT_TARGET_RATIO,
    AutoCompactDispatcher,
    AutoCompactDispatcherConfig,
    AutoCompactResult,
)
from app.context_compact_memory import (
    CompactMemorySnapshot,
    parse_active_context_summary,
    render_active_context_summary,
)
from app.context_compactor import (
    CompactionResult,
    compact_recent_messages,
    microcompact_old_tool_results,
)
from app.context_manager import estimate_messages_tokens
from app.context_message_safety import normalize_tool_call_pairs
from app.history_summarizer import OlderHistorySummarizer
from app.types import ChatMessage

_MICROCOMPACT_KEEP_RECENT_TOOL_RESULTS = 5
_MICROCOMPACT_INTERVAL_SECONDS = 60 * 60


@dataclass(slots=True)
class ContextPipelineResult:
    """统一返回每轮请求前的上下文治理结果。"""

    messages: list[ChatMessage]
    compaction_result: CompactionResult
    auto_compact_result: AutoCompactResult
    steps_taken: list[str] = field(default_factory=list)
    compaction_history_entry: dict[str, Any] = field(default_factory=dict)
    resolved_active_context_snapshot: CompactMemorySnapshot = field(default_factory=dict)
    resolved_active_context_summary: str = ""
    last_microcompact_at: float = 0.0
    auto_compact_failure_count: int = 0
    auto_compact_suppressed_until: float = 0.0
    protected_recent_messages: int = 6


@dataclass(slots=True)
class LightweightCompactionConfig:
    """
    轻量治理阶段的配置。

    这层对应新版 MiniCode 里的请求前小步优化：
    - 先控制 tool_result 体积
    - 先去掉重复扫描噪声
    - 先压掉最近窗口里最容易膨胀的部分
    """

    max_recent_tool_results: int
    truncate_tool_result_chars: int
    protected_recent_messages: int = 6


@dataclass(slots=True)
class MicrocompactState:
    """
    microcompact 的跨轮次状态。

    当前项目把它落在 context_state 里，而不是常驻内存对象里，
    这样恢复会话时仍能延续上一轮的节流窗口。
    """

    last_time_based_compact: float = 0.0
    time_based_interval: float = _MICROCOMPACT_INTERVAL_SECONDS
    keep_recent_tool_results: int = 0
    total_tokens_cleared: int = 0


@dataclass(slots=True)
class MicrocompactResult:
    """microcompact 阶段的结果。"""

    messages: list[ChatMessage]
    applied: bool = False
    cleared_count: int = 0
    tokens_freed_estimate: int = 0
    last_compact_at: float = 0.0
    carried_tool_findings: list[str] = field(default_factory=list)
    carried_open_issues: list[str] = field(default_factory=list)
    carried_key_decisions: list[str] = field(default_factory=list)
    # 记录 microcompact 决策细节，便于日志和 context_state 直接定位原因。
    reason: str = "not_evaluated"
    tool_result_count: int = 0
    keep_recent_tool_results: int = 0
    cooldown_remaining_seconds: float = 0.0


class LightweightContextPhase:
    """对应新版 MiniCode 的 tool budget / read dedup / recent cleanup 阶段。"""

    def run(
        self,
        *,
        messages: list[ChatMessage],
        config: LightweightCompactionConfig,
        workspace: str,
        usable_budget: int,
        fixed_overhead_tokens: int,
        pinned_tool_names: set[str] | None,
    ) -> tuple[CompactionResult, list[str]]:
        target_tokens = max(
            1,
            int(usable_budget * AUTO_COMPACT_TARGET_RATIO) - fixed_overhead_tokens,
        )
        result = compact_recent_messages(
            messages,
            max_recent_tool_results=config.max_recent_tool_results,
            truncate_tool_result_chars=config.truncate_tool_result_chars,
            workspace=workspace,
            pinned_tool_names=pinned_tool_names,
            target_tokens=target_tokens,
            protected_recent_messages=config.protected_recent_messages,
        )
        return result, ["tool_budget", "read_dedup", "recent_tool_cleanup"]


class MicrocompactEngine:
    """
    轻量 microcompact 引擎。

    行为上尽量贴近新版 MiniCode：
    - 不生成 assistant 摘要
    - 不破坏 tool_call / tool_result 协议
    - 只把较旧的 tool_result 正文替换成占位说明
    - 使用时间窗口做节流，避免每轮重复清理同一批历史结果
    """

    def __init__(self, state: MicrocompactState | None = None):
        self._state = state or MicrocompactState()

    @property
    def state(self) -> MicrocompactState:
        return self._state

    def run(
        self,
        *,
        messages: list[ChatMessage],
        pinned_tool_names: set[str] | None,
        usable_budget: int,
        semantic_summarizer: OlderHistorySummarizer | None = None,
    ) -> MicrocompactResult:
        now = time.time()
        decision = self._evaluate(
            messages=messages,
            usable_budget=usable_budget,
            now=now,
        )
        if not bool(decision["should_apply"]):
            return MicrocompactResult(
                messages=list(messages),
                last_compact_at=self._state.last_time_based_compact,
                reason=str(decision["reason"]),
                tool_result_count=int(decision["tool_result_count"]),
                keep_recent_tool_results=int(decision["keep_recent_tool_results"]),
                cooldown_remaining_seconds=float(decision["cooldown_remaining_seconds"]),
            )

        compaction_result = microcompact_old_tool_results(
            messages,
            keep_recent_tool_results=self._state.keep_recent_tool_results,
            pinned_tool_names=pinned_tool_names,
            semantic_summarizer=semantic_summarizer,
        )
        if compaction_result.cleared_old_tool_results <= 0:
            return MicrocompactResult(
                messages=list(compaction_result.messages),
                last_compact_at=self._state.last_time_based_compact,
                carried_tool_findings=list(compaction_result.carried_tool_findings),
                carried_open_issues=list(compaction_result.carried_open_issues),
                carried_key_decisions=list(compaction_result.carried_key_decisions),
                reason="no_old_tool_results",
                tool_result_count=int(decision["tool_result_count"]),
                keep_recent_tool_results=int(decision["keep_recent_tool_results"]),
            )

        self._state.last_time_based_compact = now
        self._state.total_tokens_cleared += compaction_result.tokens_freed_estimate
        return MicrocompactResult(
            messages=list(compaction_result.messages),
            applied=True,
            cleared_count=compaction_result.cleared_old_tool_results,
            tokens_freed_estimate=compaction_result.tokens_freed_estimate,
            last_compact_at=now,
            carried_tool_findings=list(compaction_result.carried_tool_findings),
            carried_open_issues=list(compaction_result.carried_open_issues),
            carried_key_decisions=list(compaction_result.carried_key_decisions),
            reason="applied",
            tool_result_count=int(decision["tool_result_count"]),
            keep_recent_tool_results=int(decision["keep_recent_tool_results"]),
        )

    def _should_apply(
        self,
        *,
        messages: list[ChatMessage],
        usable_budget: int,
        now: float,
    ) -> bool:
        # 先做时间节流。
        # 这里按 MiniCode 新版的思路：microcompact 不是“只要 tool_result 多就每轮都压”，
        # 而是压过一次后先冷却 1 小时，避免连续多轮把同一批旧工具结果反复清空。
        if (
            self._state.last_time_based_compact > 0
            and (now - self._state.last_time_based_compact) < self._state.time_based_interval
        ):
            return False

        tool_results = [
            message for message in messages if message.get("role") == "tool_result"
        ]
        # 只要旧 tool_result 数量超过“保留最近 N 条”的阈值，就允许做一次轻量清理。
        # 这里不看总 token 高水位；microcompact 的职责只是提前削掉旧工具正文噪音。
        return len(tool_results) > max(0, self._state.keep_recent_tool_results)


    def _evaluate(
        self,
        *,
        messages: list[ChatMessage],
        usable_budget: int,
        now: float,
    ) -> dict[str, float | int | str | bool]:
        tool_results = [
            message for message in messages if message.get("role") == "tool_result"
        ]
        tool_result_count = len(tool_results)
        keep_recent = max(0, self._state.keep_recent_tool_results)

        # 先做时间节流。microcompact 一旦刚执行过，就冷却 1 小时，避免连续多轮重复清理同一批旧结果。
        if (
            self._state.last_time_based_compact > 0
            and (now - self._state.last_time_based_compact) < self._state.time_based_interval
        ):
            return {
                "should_apply": False,
                "reason": "cooldown",
                "tool_result_count": tool_result_count,
                "keep_recent_tool_results": keep_recent,
                "cooldown_remaining_seconds": max(
                    0.0,
                    self._state.time_based_interval
                    - (now - self._state.last_time_based_compact),
                ),
            }

        # 只有旧 tool_result 数量超过“保留最近 N 条”的阈值时，才允许做 microcompact。
        if tool_result_count <= keep_recent:
            return {
                "should_apply": False,
                "reason": "below_threshold",
                "tool_result_count": tool_result_count,
                "keep_recent_tool_results": keep_recent,
                "cooldown_remaining_seconds": 0.0,
            }

        return {
            "should_apply": True,
            "reason": "ready",
            "tool_result_count": tool_result_count,
            "keep_recent_tool_results": keep_recent,
            "cooldown_remaining_seconds": 0.0,
        }


class ContextCompactor:
    """
    统一的请求前上下文编排器。

    这个类是对齐新版 MiniCode 结构的核心：
    1. 轻量治理阶段
    2. microcompact
    3. auto compact 调度

    但对外仍然产出当前项目已有的 ContextPipelineResult，
    这样 runtime、context_state、日志、测试都不需要重写。
    """

    def __init__(self):
        self._lightweight_phase = LightweightContextPhase()

    def process_request(
        self,
        *,
        messages: list[ChatMessage],
        summary_source_messages: list[ChatMessage] | None,
        lightweight_config: LightweightCompactionConfig,
        workspace: str,
        usable_budget: int,
        fixed_overhead_tokens: int,
        auto_compact_summary: str,
        auto_compact_snapshot: CompactMemorySnapshot | None,
        force_auto_compact: bool,
        pinned_tool_names: set[str] | None,
        microcompact_state: MicrocompactState,
        auto_compact_failure_count: int = 0,
        auto_compact_suppressed_until: float = 0.0,
        semantic_summarizer: OlderHistorySummarizer | None = None,
    ) -> ContextPipelineResult:
        auto_compact = AutoCompactDispatcher(
            config=AutoCompactDispatcherConfig(),
            failure_count=auto_compact_failure_count,
            suppressed_until=auto_compact_suppressed_until,
        )
        compaction_result, steps_taken = self._lightweight_phase.run(
            messages=messages,
            config=lightweight_config,
            workspace=workspace,
            usable_budget=usable_budget,
            fixed_overhead_tokens=fixed_overhead_tokens,
            pinned_tool_names=pinned_tool_names,
        )

        microcompact_engine = MicrocompactEngine(microcompact_state)
        microcompact_result = microcompact_engine.run(
            messages=compaction_result.messages,
            pinned_tool_names=pinned_tool_names,
            usable_budget=usable_budget,
            semantic_summarizer=semantic_summarizer,
        )
        if microcompact_result.applied:
            overlay = CompactionResult(
                messages=list(microcompact_result.messages),
                cleared_old_tool_results=microcompact_result.cleared_count,
                tokens_freed_estimate=microcompact_result.tokens_freed_estimate,
                carried_tool_findings=list(microcompact_result.carried_tool_findings),
                carried_open_issues=list(microcompact_result.carried_open_issues),
                carried_key_decisions=list(microcompact_result.carried_key_decisions),
            )
            _merge_compaction_result(base=compaction_result, overlay=overlay)
            steps_taken.append("microcompact")

        recent_tokens_after_compaction = estimate_messages_tokens(compaction_result.messages)
        auto_compact_result = auto_compact.dispatch(
            messages=compaction_result.messages,
            usable_budget=usable_budget,
            summary_base=auto_compact_summary,
            summary_snapshot=auto_compact_snapshot,
            summary_source_messages=summary_source_messages or messages,
            fixed_overhead_tokens=fixed_overhead_tokens,
            force_full=force_auto_compact,
            semantic_summarizer=semantic_summarizer,
        )
        if auto_compact_result.applied:
            steps_taken.append(f"auto_compact:{auto_compact_result.strategy}")

        output_messages = (
            auto_compact_result.messages
            if auto_compact_result.applied
            else compaction_result.messages
        )
        normalized_messages = normalize_tool_call_pairs(output_messages)
        resolved_snapshot, resolved_context = _resolve_active_context_outputs(
            auto_compact_summary=auto_compact_summary,
            auto_compact_snapshot=auto_compact_snapshot,
            auto_compact_result=auto_compact_result,
        )

        history_entry = {
            "truncated_tool_results": compaction_result.truncated_tool_results,
            "cleared_old_tool_results": compaction_result.cleared_old_tool_results,
            "deduped_read_results": compaction_result.deduped_read_results,
            "semantic_compacted_pairs": compaction_result.semantic_compacted_pairs,
            "dropped_progress_messages": compaction_result.dropped_progress_messages,
            "priority_dropped_messages": compaction_result.priority_dropped_messages,
            "recent_tokens_after_compaction": recent_tokens_after_compaction,
            "tokens_freed_estimate": (
                compaction_result.tokens_freed_estimate
                + auto_compact_result.tokens_freed_estimate
            ),
            "microcompact_applied": microcompact_result.applied,
            "microcompact_reason": microcompact_result.reason,
            "microcompact_tool_results": microcompact_result.tool_result_count,
            "microcompact_keep_recent": microcompact_result.keep_recent_tool_results,
            "microcompact_cooldown_remaining_seconds": (
                microcompact_result.cooldown_remaining_seconds
            ),
            "last_microcompact_at": microcompact_result.last_compact_at,
            "auto_compact_applied": auto_compact_result.applied,
            "auto_compact_strategy": auto_compact_result.strategy,
            "auto_compact_tokens_before": auto_compact_result.tokens_before,
            "auto_compact_tokens_after": auto_compact_result.tokens_after,
            "steps_taken": list(steps_taken),
        }
        return ContextPipelineResult(
            messages=normalized_messages,
            compaction_result=compaction_result,
            auto_compact_result=auto_compact_result,
            steps_taken=steps_taken,
            compaction_history_entry=history_entry,
            resolved_active_context_snapshot=resolved_snapshot,
            resolved_active_context_summary=resolved_context,
            last_microcompact_at=microcompact_result.last_compact_at,
            auto_compact_failure_count=auto_compact_result.failure_count,
            auto_compact_suppressed_until=auto_compact_result.suppressed_until,
        )


class ContextCompactorPipeline:
    """
    当前项目的兼容入口。

    对外接口保持不变，但内部已经切成更接近新版 MiniCode 的结构：
    `LightweightContextPhase -> MicrocompactEngine -> AutoCompactDispatcher -> ContextCompactor`
    """

    def __init__(self):
        self._compactor = ContextCompactor()

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
        last_microcompact_at: float = 0.0,
        auto_compact_failure_count: int = 0,
        auto_compact_suppressed_until: float = 0.0,
        semantic_summarizer: OlderHistorySummarizer | None = None,
    ) -> ContextPipelineResult:
        return self._compactor.process_request(
            messages=messages,
            summary_source_messages=summary_source_messages,
            lightweight_config=LightweightCompactionConfig(
                max_recent_tool_results=max_recent_tool_results,
                truncate_tool_result_chars=truncate_tool_result_chars,
            ),
            workspace=workspace,
            usable_budget=usable_budget,
            fixed_overhead_tokens=fixed_overhead_tokens,
            auto_compact_summary=auto_compact_summary,
            auto_compact_snapshot=auto_compact_snapshot,
            force_auto_compact=force_auto_compact,
            pinned_tool_names=pinned_tool_names,
            microcompact_state=MicrocompactState(
                last_time_based_compact=max(0.0, float(last_microcompact_at or 0.0)),
                keep_recent_tool_results=_MICROCOMPACT_KEEP_RECENT_TOOL_RESULTS,
            ),
            auto_compact_failure_count=auto_compact_failure_count,
            auto_compact_suppressed_until=auto_compact_suppressed_until,
            semantic_summarizer=semantic_summarizer,
        )


def _resolve_active_context_outputs(
    *,
    auto_compact_summary: str,
    auto_compact_snapshot: CompactMemorySnapshot | None,
    auto_compact_result: AutoCompactResult,
) -> tuple[CompactMemorySnapshot, str]:
    """统一解析 active context 的最终输出，避免散落在不同阶段重复拼接。"""
    resolved_snapshot = auto_compact_snapshot or parse_active_context_summary(
        auto_compact_summary
    )
    resolved_context = auto_compact_summary.strip()
    if auto_compact_result.summary_snapshot:
        resolved_snapshot = auto_compact_result.summary_snapshot
        resolved_context = (
            auto_compact_result.summary_text.strip()
            or render_active_context_summary(resolved_snapshot)
        )
    elif resolved_snapshot and not resolved_context:
        resolved_context = render_active_context_summary(resolved_snapshot)
    return resolved_snapshot, resolved_context


def _merge_compaction_result(*, base: CompactionResult, overlay: CompactionResult) -> None:
    """
    把额外阶段的统计并回主结果。

    这里继续沿用当前项目已有的 CompactionResult，
    避免 runtime、日志、测试、state 结构一起被迫改动。
    """

    base.messages = list(overlay.messages)
    base.truncated_tool_results += overlay.truncated_tool_results
    base.cleared_old_tool_results += overlay.cleared_old_tool_results
    base.deduped_read_results += overlay.deduped_read_results
    base.semantic_compacted_pairs += overlay.semantic_compacted_pairs
    base.dropped_progress_messages += overlay.dropped_progress_messages
    base.priority_dropped_messages += overlay.priority_dropped_messages
    base.tokens_freed_estimate += overlay.tokens_freed_estimate
    base.carried_tool_findings.extend(overlay.carried_tool_findings)
    base.carried_open_issues.extend(overlay.carried_open_issues)
    base.carried_key_decisions.extend(overlay.carried_key_decisions)
