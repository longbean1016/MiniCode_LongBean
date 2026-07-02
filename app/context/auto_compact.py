from __future__ import annotations

import re
import time
from dataclasses import dataclass

"""自动压缩策略模块，负责在上下文逼近阈值时触发历史压缩。"""

from app.context.compact_memory import (
    CompactMemorySnapshot,
    build_active_context_event_snapshot,
    merge_active_context_snapshots,
    parse_active_context_summary,
    prioritize_snapshot_for_current_focus,
    render_active_context_summary,
    render_full_active_context_summary,
)
from app.context.message_safety import is_internal_compaction_marker
from app.context.manager import estimate_messages_tokens, estimate_tokens
from app.context.history_summarizer import OlderHistorySummarizer
from app.types import ChatMessage

AUTO_COMPACT_TRIGGER_RATIO = 0.92
AUTO_COMPACT_TARGET_RATIO = 0.78
AUTO_COMPACT_SESSION_TARGET_RATIO = 0.58
AUTO_COMPACT_FULL_TARGET_RATIO = 0.35
AUTO_COMPACT_TOOL_RESULT_TRIGGER_RATIO = 0.45
AUTO_COMPACT_REPEATED_SCAN_TRIGGER_COUNT = 3
AUTO_COMPACT_MIN_KEEP_MESSAGES = 3
AUTO_COMPACT_MIN_FULL_KEEP_MESSAGES = 1
AUTO_COMPACT_SESSION_SUMMARY_MAX_CHARS = 640
AUTO_COMPACT_FULL_SUMMARY_MAX_TOKENS = 2000
AUTO_COMPACT_MIN_SUMMARY_CHARS = 60
AUTO_COMPACT_MIN_SUMMARY_TOKENS = 40
AUTO_COMPACT_SUMMARY_SHRINK_RATIO = 0.78


@dataclass(slots=True)
class AutoCompactResult:
    """保存一次 Auto Compact 的结果。"""

    messages: list[ChatMessage]
    applied: bool = False
    strategy: str = "none"
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_freed_estimate: int = 0
    summary_text: str = ""
    summary_snapshot: CompactMemorySnapshot | None = None
    failure_count: int = 0
    suppressed_until: float = 0.0


@dataclass(slots=True)
class AutoCompactDispatcherConfig:
    """
    Auto Compact 调度配置。

    当前项目仍使用既有阈值常量，只是把触发参数收口成配置对象，
    这样更接近新版 MiniCode 的 dispatcher 结构。
    """

    trigger_ratio: float = AUTO_COMPACT_TRIGGER_RATIO
    target_ratio: float = AUTO_COMPACT_TARGET_RATIO
    tool_result_trigger_ratio: float = AUTO_COMPACT_TOOL_RESULT_TRIGGER_RATIO
    repeated_scan_trigger_count: int = AUTO_COMPACT_REPEATED_SCAN_TRIGGER_COUNT
    circuit_breaker_limit: int = 3
    warning_suppress_seconds: float = 30.0


class AutoCompactDispatcher:
    """
    对应新版 MiniCode 的 Auto Compact 调度器。

    这个对象专门负责“是否触发”和“按何种顺序调度策略”，
    具体的 session/full compact 细节仍复用当前项目已有实现。
    """

    def __init__(
        self,
        config: AutoCompactDispatcherConfig | None = None,
        *,
        failure_count: int = 0,
        suppressed_until: float = 0.0,
    ):
        self._config = config or AutoCompactDispatcherConfig()
        self._failure_count = max(0, int(failure_count))
        self._suppressed_until = max(0.0, float(suppressed_until or 0.0))

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def suppressed_until(self) -> float:
        return self._suppressed_until

    @property
    def is_tripped(self) -> bool:
        return self._failure_count >= self._config.circuit_breaker_limit

    def should_trigger(
        self,
        *,
        total_tokens: int,
        usable_budget: int,
        tool_result_tokens: int = 0,
        repeated_scan_count: int = 0,
    ) -> bool:
        # 与 MiniCode 新版一致：一旦 circuit breaker 触发，
        # 后续自动压缩先短路，避免每轮都重复做高成本尝试。
        if self.is_tripped:
            return False
        if self._suppressed_until > time.time():
            return False
        return should_trigger_auto_compact(
            total_tokens=total_tokens,
            usable_budget=usable_budget,
            tool_result_tokens=tool_result_tokens,
            repeated_scan_count=repeated_scan_count,
        )

    def dispatch(
        self,
        *,
        messages: list[ChatMessage],
        usable_budget: int,
        summary_base: str,
        summary_snapshot: CompactMemorySnapshot | None = None,
        summary_source_messages: list[ChatMessage] | None = None,
        fixed_overhead_tokens: int = 0,
        force_full: bool = False,
        semantic_summarizer: OlderHistorySummarizer | None = None,
    ) -> AutoCompactResult:
        # 在调度器入口统一计算 token 压力与扫描噪声。
        # 这样旧入口和新入口最终都会走同一套触发条件。
        current_messages = [dict(message) for message in messages]
        tokens_before = fixed_overhead_tokens + estimate_messages_tokens(current_messages)
        tool_result_tokens = _estimate_tool_result_tokens(current_messages)
        repeated_scan_count = _count_repeated_scan_results(current_messages)

        if not force_full and self.is_tripped:
            return AutoCompactResult(
                messages=current_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                failure_count=self._failure_count,
                suppressed_until=self._suppressed_until,
            )

        if not force_full and not self.should_trigger(
            total_tokens=tokens_before,
            usable_budget=usable_budget,
            tool_result_tokens=tool_result_tokens,
            repeated_scan_count=repeated_scan_count,
        ):
            return AutoCompactResult(
                messages=current_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                failure_count=self._failure_count,
                suppressed_until=self._suppressed_until,
            )

        if not force_full:
            # 顺序上先尝试 session memory compact。
            # 只有轻量摘要压不下去时，才退化到更重的 full compact。
            session_result = _run_session_memory_compact(
                messages=current_messages,
                usable_budget=usable_budget,
                summary_base=summary_base,
                summary_snapshot=summary_snapshot,
                fixed_overhead_tokens=fixed_overhead_tokens,
                semantic_summarizer=semantic_summarizer,
            )
            if session_result.applied:
                self._mark_success()
                session_result.failure_count = self._failure_count
                session_result.suppressed_until = self._suppressed_until
                return session_result

        result = _run_full_compact(
            messages=current_messages,
            usable_budget=usable_budget,
            summary_base=summary_base,
            summary_snapshot=summary_snapshot,
            summary_source_messages=summary_source_messages,
            fixed_overhead_tokens=fixed_overhead_tokens,
            semantic_summarizer=semantic_summarizer,
        )
        if result.applied:
            self._mark_success()
        else:
            self._mark_failure()
        result.failure_count = self._failure_count
        result.suppressed_until = self._suppressed_until
        return result

    def _mark_success(self) -> None:
        # 成功压缩后清空失败计数，并在短时间内抑制重复告警/重复压缩。
        self._failure_count = 0
        self._suppressed_until = time.time() + self._config.warning_suppress_seconds

    def _mark_failure(self) -> None:
        self._failure_count += 1


def should_trigger_auto_compact(
    *,
    total_tokens: int,
    usable_budget: int,
    tool_result_tokens: int = 0,
    repeated_scan_count: int = 0,
) -> bool:
    """Trigger auto compact only on overall context high-water pressure."""
    _ = tool_result_tokens
    _ = repeated_scan_count
    if usable_budget <= 0:
        return False
    return total_tokens >= int(usable_budget * AUTO_COMPACT_TRIGGER_RATIO)


def run_auto_compact(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
    summary_base: str,
    summary_snapshot: CompactMemorySnapshot | None = None,
    summary_source_messages: list[ChatMessage] | None = None,
    fixed_overhead_tokens: int = 0,
    force_full: bool = False,
    semantic_summarizer: OlderHistorySummarizer | None = None,
) -> AutoCompactResult:
    """
    运行类似 minicode 的 Auto Compact 调度。

    策略顺序：
    1. Session Memory Compact
    2. Full Compact
    """
    return AutoCompactDispatcher().dispatch(
        messages=messages,
        usable_budget=usable_budget,
        summary_base=summary_base,
        summary_snapshot=summary_snapshot,
        summary_source_messages=summary_source_messages,
        fixed_overhead_tokens=fixed_overhead_tokens,
        force_full=force_full,
        semantic_summarizer=semantic_summarizer,
    )


def _run_session_memory_compact(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
    summary_base: str,
    summary_snapshot: CompactMemorySnapshot | None,
    fixed_overhead_tokens: int,
    semantic_summarizer: OlderHistorySummarizer | None = None,
) -> AutoCompactResult:
    """优先用已有摘要和工作记忆做一次轻量压缩。"""
    # session compact 优先复用已有结构化摘要，不重新发明摘要语义。
    # 这一层的目标是“把较早历史折进摘要，尽量保住最近窗口”，而不是重新做重型摘要。
    baseline_snapshot = summary_snapshot or parse_active_context_summary(summary_base)
    snapshot_text = (
        render_active_context_summary(baseline_snapshot).strip()
        if baseline_snapshot
        else ""
    )
    normalized_summary = _limit_summary_text(
        (snapshot_text or summary_base.strip()),
        AUTO_COMPACT_SESSION_SUMMARY_MAX_CHARS,
    )
    if not normalized_summary:
        return AutoCompactResult(
            messages=messages,
            tokens_before=fixed_overhead_tokens + estimate_messages_tokens(messages),
            tokens_after=fixed_overhead_tokens + estimate_messages_tokens(messages),
        )

    system_messages, other_messages = _split_system_messages(messages)
    if len(other_messages) <= AUTO_COMPACT_MIN_KEEP_MESSAGES:
        return AutoCompactResult(
            messages=messages,
            tokens_before=fixed_overhead_tokens + estimate_messages_tokens(messages),
            tokens_after=fixed_overhead_tokens + estimate_messages_tokens(messages),
        )

    # 先确定最近 tail，再把更早内容折进 marker。
    # 这是为了让模型看到的仍是一段自然连续的最新对话，而不是被先裁碎 recent window。
    tail_messages = _select_recent_tail(
        messages=other_messages,
        usable_budget=usable_budget,
        target_ratio=AUTO_COMPACT_SESSION_TARGET_RATIO,
        min_keep_messages=AUTO_COMPACT_MIN_KEEP_MESSAGES,
    )
    removed_count = max(0, len(other_messages) - len(tail_messages))
    if removed_count <= 0:
        return AutoCompactResult(
            messages=messages,
            tokens_before=fixed_overhead_tokens + estimate_messages_tokens(messages),
            tokens_after=fixed_overhead_tokens + estimate_messages_tokens(messages),
        )

    removed_messages = other_messages[:removed_count]
    if semantic_summarizer is not None:
        # session compact 优先让模型把“被折进去的较早对话”整理成结构化摘要，
        # 但模型失败时仍然直接退回当前这份规则基线，不影响压缩主流程。
        normalized_summary, normalized_snapshot = semantic_summarizer.summarize_active_context_compaction(
            strategy="session",
            removed_messages=removed_messages,
            fallback_summary=normalized_summary,
            fallback_snapshot=baseline_snapshot,
            focus_lines=[],
        )
        normalized_summary = _limit_summary_text(
            normalized_summary,
            AUTO_COMPACT_SESSION_SUMMARY_MAX_CHARS,
        )
        baseline_snapshot = normalized_snapshot or baseline_snapshot

    compacted, summary_text, tokens_after = _fit_compacted_messages(
        system_messages=system_messages,
        tail_messages=tail_messages,
        summary_text=normalized_summary,
        marker_title="会话记忆压缩",
        summary_title="压缩摘要",
        removed_count=removed_count,
        usable_budget=usable_budget,
        fixed_overhead_tokens=fixed_overhead_tokens,
        min_keep_messages=AUTO_COMPACT_MIN_KEEP_MESSAGES,
    )

    tokens_before = fixed_overhead_tokens + estimate_messages_tokens(messages)
    freed = max(0, tokens_before - tokens_after)
    return AutoCompactResult(
        messages=compacted,
        applied=freed > 0 and tokens_after <= int(usable_budget * AUTO_COMPACT_TRIGGER_RATIO),
        strategy="session_memory" if freed > 0 and tokens_after <= int(usable_budget * AUTO_COMPACT_TRIGGER_RATIO) else "none",
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_freed_estimate=freed,
        summary_text=summary_text,
        summary_snapshot=baseline_snapshot or None,
    )


def _run_full_compact(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
    summary_base: str,
    summary_snapshot: CompactMemorySnapshot | None,
    summary_source_messages: list[ChatMessage] | None,
    fixed_overhead_tokens: int,
    semantic_summarizer: OlderHistorySummarizer | None = None,
) -> AutoCompactResult:
    """当轻量摘要压不下去时，退化到更强的全量压缩。"""
    system_messages, other_messages = _split_system_messages(messages)
    if len(other_messages) <= AUTO_COMPACT_MIN_FULL_KEEP_MESSAGES:
        return AutoCompactResult(
            messages=messages,
            tokens_before=fixed_overhead_tokens + estimate_messages_tokens(messages),
            tokens_after=fixed_overhead_tokens + estimate_messages_tokens(messages),
        )

    # full compact 会更激进地缩小 tail，因为此时说明 session compact 已经压不下去了。
    tail_messages = _select_recent_tail(
        messages=other_messages,
        usable_budget=usable_budget,
        target_ratio=AUTO_COMPACT_FULL_TARGET_RATIO,
        min_keep_messages=AUTO_COMPACT_MIN_FULL_KEEP_MESSAGES,
    )
    tail_messages = _ensure_full_compact_removes_tool_context(
        original_messages=other_messages,
        tail_messages=tail_messages,
        min_keep_messages=AUTO_COMPACT_MIN_FULL_KEEP_MESSAGES,
    )
    removed_count = max(0, len(other_messages) - len(tail_messages))
    removed_messages = other_messages[:removed_count]
    # summary_source_messages 允许用“压缩前更完整的历史”来生成摘要，
    # 避免当前 messages 已经做过 recent 微压缩后，把原始工具上下文丢得太多。
    _, summary_other_messages = _split_system_messages(summary_source_messages or messages)
    if len(summary_other_messages) >= removed_count:
        removed_messages = summary_other_messages[:removed_count]
        if removed_count > 0:
            trailing_source = summary_other_messages[-removed_count:]
            if trailing_source != removed_messages:
                removed_messages = [*removed_messages, *trailing_source]
    merged_snapshot, rendered_summary = _build_structured_summary_package(
        removed_messages=removed_messages,
        focus_source_messages=summary_other_messages or other_messages,
        summary_base=summary_base,
        summary_snapshot=summary_snapshot,
        semantic_summarizer=semantic_summarizer,
    )
    summary_text = _limit_summary_text(
        rendered_summary,
        AUTO_COMPACT_FULL_SUMMARY_MAX_TOKENS,
        by_tokens=True,
    )

    compacted, summary_text, tokens_after = _fit_compacted_messages(
        system_messages=system_messages,
        tail_messages=tail_messages,
        summary_text=summary_text,
        marker_title="全量压缩",
        summary_title="对话摘要",
        removed_count=removed_count,
        usable_budget=usable_budget,
        fixed_overhead_tokens=fixed_overhead_tokens,
        min_keep_messages=AUTO_COMPACT_MIN_FULL_KEEP_MESSAGES,
    )

    tokens_before = fixed_overhead_tokens + estimate_messages_tokens(messages)
    freed = max(0, tokens_before - tokens_after)
    applied = freed > 0 and tokens_after <= int(usable_budget * AUTO_COMPACT_TRIGGER_RATIO)
    return AutoCompactResult(
        messages=compacted,
        applied=applied,
        strategy="full" if applied else "none",
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_freed_estimate=freed,
        summary_text=summary_text,
        summary_snapshot=merged_snapshot or None,
    )


def _fit_compacted_messages(
    *,
    system_messages: list[ChatMessage],
    tail_messages: list[ChatMessage],
    summary_text: str,
    marker_title: str,
    summary_title: str,
    removed_count: int,
    usable_budget: int,
    fixed_overhead_tokens: int,
    min_keep_messages: int,
) -> tuple[list[ChatMessage], str, int]:
    """持续收紧摘要和尾部，直到压到目标阈值以下。"""
    tail_current = [dict(message) for message in tail_messages]
    summary_current = summary_text.strip()
    threshold_tokens = max(1, int(usable_budget * AUTO_COMPACT_TARGET_RATIO))

    while True:
        marker = _build_marker(
            marker_title=marker_title,
            summary_title=summary_title,
            summary_text=summary_current,
            removed_count=removed_count,
        )
        compacted = list(system_messages)
        compacted.append(marker)
        compacted.extend(tail_current)

        tokens_after = fixed_overhead_tokens + estimate_messages_tokens(compacted)
        if tokens_after <= threshold_tokens:
            return compacted, summary_current, tokens_after

        # 先缩 tail，再缩 summary。
        # 原因是 tail 里保存的是“最近可继续执行的真实对话结构”，
        # 只要还能删掉更旧的 tail 边缘，就不该先把摘要压到失真。
        if len(tail_current) > min_keep_messages:
            tail_current = _drop_oldest_tail_segment(tail_current, min_keep_messages)
            continue

        summary_tokens = estimate_tokens(summary_current)
        if summary_tokens > AUTO_COMPACT_MIN_SUMMARY_TOKENS:
            next_limit = max(
                AUTO_COMPACT_MIN_SUMMARY_TOKENS,
                int(summary_tokens * AUTO_COMPACT_SUMMARY_SHRINK_RATIO),
            )
            next_summary = _limit_summary_text(
                summary_current,
                next_limit,
                by_tokens=True,
            )
            if next_summary == summary_current:
                return compacted, summary_current, tokens_after
            summary_current = next_summary
            continue

        return compacted, summary_current, tokens_after


def _build_marker(
    *,
    marker_title: str,
    summary_title: str,
    summary_text: str,
    removed_count: int,
) -> ChatMessage:
    """构造写入 context_state 和模型上下文的中文压缩标记。"""
    return {
        "role": "system",
        "content": (
            f"[{marker_title}]\n"
            f"已折叠较早消息数：{removed_count}\n\n"
            f"## {summary_title}\n"
            f"{summary_text}\n\n"
            "--- 最近对话继续如下 ---"
        ),
    }


def _split_system_messages(messages: list[ChatMessage]) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """把系统消息和普通对话拆开，避免压缩时误删系统提示。"""
    system_messages: list[ChatMessage] = []
    other_messages: list[ChatMessage] = []
    for message in messages:
        if message.get("role") == "system":
            if is_internal_compaction_marker(message):
                continue
            system_messages.append(dict(message))
        else:
            other_messages.append(dict(message))
    return system_messages, other_messages


def _select_recent_tail(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
    target_ratio: float,
    min_keep_messages: int,
) -> list[ChatMessage]:
    """从尾部反向保留最近消息，尽量保留一个完整的最近窗口。"""
    if not messages:
        return []

    target_tokens = max(1, int(usable_budget * target_ratio))
    kept_reversed: list[ChatMessage] = []
    tail_tokens = 0

    # 从后往前收集，天然更偏向保留最近轮次。
    # 到达目标预算后立即停，避免把“较早但不关键”的历史继续拖进 tail。
    for message in reversed(messages):
        kept_reversed.append(dict(message))
        tail_tokens += estimate_messages_tokens([message])
        if len(kept_reversed) >= min_keep_messages and tail_tokens >= target_tokens:
            break

    tail_messages = list(reversed(kept_reversed))
    # 截完后再修正边界，保证不会把 tool_call/tool_result 从中间切断。
    return _adjust_tail_for_tool_pairs(messages, tail_messages)


def _adjust_tail_for_tool_pairs(
    original_messages: list[ChatMessage],
    tail_messages: list[ChatMessage],
) -> list[ChatMessage]:
    """避免把 tool_call 和 tool_result 从中间切断。"""
    if not tail_messages:
        return tail_messages

    tail_start = len(original_messages) - len(tail_messages)
    # 如果 tail 正好从 tool_result 开始，要把对应 tool_call 一并拉进来。
    # 否则模型会看到一个“没有调用来源的工具结果”，这类孤立证据很容易让后续推理跑偏。
    while tail_start > 0:
        first_message = original_messages[tail_start]
        if first_message.get("role") != "tool_result":
            break

        previous_message = original_messages[tail_start - 1]
        if previous_message.get("role") != "assistant_tool_call":
            break

        tail_start -= 1

    return [dict(message) for message in original_messages[tail_start:]]


def _drop_oldest_tail_segment(
    tail_messages: list[ChatMessage],
    min_keep_messages: int,
) -> list[ChatMessage]:
    """从尾部窗口前端裁掉最旧的一段，同时避免留下孤立的 tool_result。"""
    if len(tail_messages) <= min_keep_messages:
        return tail_messages

    next_tail = list(tail_messages[1:])
    # 如果删完第一条后，新的开头正好是 tool_result，
    # 说明 assistant_tool_call 被单独删掉了，需要把配对结果也一起挪掉。
    if (
        next_tail
        and next_tail[0].get("role") == "tool_result"
        and len(next_tail) > min_keep_messages
    ):
        next_tail = next_tail[1:]
    return next_tail


def _ensure_full_compact_removes_tool_context(
    *,
    original_messages: list[ChatMessage],
    tail_messages: list[ChatMessage],
    min_keep_messages: int,
) -> list[ChatMessage]:
    """
    full compact 触发后，尽量把一段完整的旧工具轮次折叠进 summary。

    否则会出现：只折叠掉最早 user 消息，真正高价值的旧 tool_result 仍留在 tail，
    下一轮再被 recent-window 截断，语义无法沉淀到 compact summary。
    """
    if not tail_messages:
        return tail_messages

    tail_current = [dict(message) for message in tail_messages]
    while len(tail_current) > min_keep_messages and _count_tool_results(tail_current) > 1:
        next_tail = _drop_leading_tool_round(tail_current, min_keep_messages)
        if len(next_tail) == len(tail_current):
            break
        tail_current = next_tail

    while (
        len(tail_current) > min_keep_messages
        and tail_current
        and tail_current[0].get("role") == "tool_result"
    ):
        tail_current = tail_current[1:]
    return tail_current


def _count_tool_results(messages: list[ChatMessage]) -> int:
    """估算 tail 中还包含多少段工具结果。"""
    return sum(1 for message in messages if message.get("role") == "tool_result")


def _drop_leading_tool_round(
    tail_messages: list[ChatMessage],
    min_keep_messages: int,
) -> list[ChatMessage]:
    """优先移除 tail 头部较旧的一段工具交互，保留最后一轮工具上下文。"""
    if len(tail_messages) <= min_keep_messages:
        return tail_messages
    if not tail_messages:
        return tail_messages

    first_role = tail_messages[0].get("role")
    if first_role == "assistant_tool_call":
        next_index = 1
        if next_index < len(tail_messages) and len(tail_messages) - 2 >= min_keep_messages:
            if tail_messages[next_index].get("role") == "tool_result":
                return list(tail_messages[2:])
        return list(tail_messages[1:])
    if first_role == "tool_result":
        return list(tail_messages[1:])
    return _drop_oldest_tail_segment(tail_messages, min_keep_messages)


def _build_structured_summary(
    *,
    removed_messages: list[ChatMessage],
    summary_base: str,
    summary_snapshot: CompactMemorySnapshot | None,
) -> str:
    """构造结构化快照摘要，保留语义槽位而不是原始措辞。"""
    _, rendered = _build_structured_summary_package(
        removed_messages=removed_messages,
        summary_base=summary_base,
        summary_snapshot=summary_snapshot,
    )
    return rendered


def _build_structured_summary_package(
    *,
    removed_messages: list[ChatMessage],
    focus_source_messages: list[ChatMessage] | None,
    summary_base: str,
    summary_snapshot: CompactMemorySnapshot | None,
    semantic_summarizer: OlderHistorySummarizer | None = None,
) -> tuple[CompactMemorySnapshot, str]:
    """同时返回 full compact 的结构化快照和渲染文本。"""
    # full compact 的关键不是生成一段漂亮 prose，
    # 而是把旧历史尽量归并到结构化槽位里，便于下一轮继续被稳定引用。
    base_snapshot = summary_snapshot or parse_active_context_summary(summary_base)
    if not base_snapshot and summary_base.strip():
        base_snapshot = {
            "history_summary": [
                _limit_summary_text(summary_base.strip(), 140),
            ]
        }
    event_snapshot = build_active_context_event_snapshot(
        removed_messages=removed_messages,
    )
    merged_snapshot = merge_active_context_snapshots(
        base_snapshot=base_snapshot,
        overlay_snapshot=event_snapshot,
    )
    focus_snapshot = _extract_full_compact_focus_snapshot(
        focus_source_messages or removed_messages
    )
    if focus_snapshot:
        # 明确识别到“当前主问题 / 当前目标 / 当前风险”时，
        # 要把这些锚点顶到 full compact 摘要前面，避免旧主题继续主导后续推理。
        merged_snapshot = merge_active_context_snapshots(
            base_snapshot=merged_snapshot,
            overlay_snapshot=focus_snapshot,
        )
        for section_key in ("active_tasks", "decisions", "open_issues"):
            pinned_lines = list(focus_snapshot.get(section_key, []))
            if pinned_lines:
                merged_snapshot[section_key] = pinned_lines[:1]
    # full compact 里最容易出的问题不是“没有工具结论”，而是“旧主题工具结论把当前主线带偏”。
    # 这里先保留 base + event 的合并结果；只有当我们明确识别到当前主问题/目标/风险时，
    # 才进一步按当前焦点去过滤不对齐的旧 tool_findings。
    focus_lines = [
        *focus_snapshot.get("active_tasks", []),
        *focus_snapshot.get("decisions", []),
        *focus_snapshot.get("open_issues", []),
    ]
    if focus_lines:
        # 只有 full compact 且存在明确焦点时，才对 tool_findings 做对齐过滤。
        # 普通 snapshot / cached state 恢复不做这一步，避免把有价值的旧承接误删。
        merged_snapshot = prioritize_snapshot_for_current_focus(
            merged_snapshot,
            focus_lines=focus_lines,
            drop_unaligned_tool_findings=True,
        )
        # 如果当前已经明确切到新主题，而这次被折叠的历史里没有新的工具结论支撑，
        # 那么保留旧主题 tool_findings 只会继续把后续回答带偏，这里直接清掉。
        if not event_snapshot.get("tool_findings") and "tool_findings" in merged_snapshot:
            merged_snapshot.pop("tool_findings", None)
    elif _has_current_focus_signal(focus_source_messages or removed_messages):
        # 有“当前任务/当前主问题”等信号，但没抽到标准 focus_snapshot 时，
        # 也说明用户主线已经切换。此时若没有新的事件型工具结论支撑，
        # 继续保留旧主题 tool_findings 同样会把摘要带偏。
        if not event_snapshot.get("tool_findings") and "tool_findings" in merged_snapshot:
            merged_snapshot.pop("tool_findings", None)
    rendered = render_full_active_context_summary(merged_snapshot).strip()
    if semantic_summarizer is not None:
        # full compact 更容易因为历史跨度太大而丢掉主线，
        # 所以这里先尝试模型补一份结构化摘要，再和规则结果做融合。
        try:
            rendered, merged_snapshot = semantic_summarizer.summarize_active_context_compaction(
                strategy="full",
                removed_messages=removed_messages,
                fallback_summary=rendered,
                fallback_snapshot=merged_snapshot,
                focus_lines=focus_lines,
            )
        except Exception:
            pass
    if rendered:
        return merged_snapshot, rendered
    normalized_base = summary_base.strip()
    if normalized_base:
        return base_snapshot, _limit_summary_text(normalized_base, 120)
    return (
        merged_snapshot,
        "结构化压缩已保留较早对话中的关键任务、决策、风险与工具发现。",
    )


def _limit_summary_text(
    text: str,
    max_size: int,
    *,
    by_tokens: bool = False,
) -> str:
    """限制压缩摘要长度，避免摘要本身反过来撑爆上下文。"""
    normalized = text.strip()
    if max_size <= 0:
        return ""
    if by_tokens:
        return _limit_summary_tokens(normalized, max_size)
    if len(normalized) <= max_size:
        return normalized
    suffix = " ...[摘要已截断]"
    if max_size <= len(suffix):
        return normalized[:max_size]
    head_limit = max_size - len(suffix)
    return f"{normalized[:head_limit]}{suffix}"


def _limit_summary_tokens(text: str, max_tokens: int) -> str:
    """按 token 预算截断摘要，更接近 minicode 的 summary budget。"""
    normalized = text.strip()
    if not normalized:
        return ""
    if estimate_tokens(normalized) <= max_tokens:
        return normalized

    suffix = " ...[摘要已截断]"
    if max_tokens <= estimate_tokens(suffix):
        return _truncate_text_to_token_budget(normalized, max_tokens, "")

    head_budget = max_tokens - estimate_tokens(suffix)
    head = _truncate_text_to_token_budget(normalized, head_budget, "")
    if not head:
        return _truncate_text_to_token_budget(normalized, max_tokens, "")
    return f"{head}{suffix}"


def _truncate_text_to_token_budget(text: str, max_tokens: int, suffix: str) -> str:
    """用二分法找到满足 token 预算的最长前缀。"""
    normalized = text.strip()
    if not normalized or max_tokens <= 0:
        return ""

    low = 0
    high = len(normalized)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = normalized[:middle].rstrip()
        if suffix:
            candidate = f"{candidate}{suffix}" if candidate else suffix
        if estimate_tokens(candidate) <= max_tokens:
            best = candidate
            low = middle + 1
            continue
        high = middle - 1
    return best


def _extract_full_compact_focus_snapshot(
    messages: list[ChatMessage],
) -> CompactMemorySnapshot:
    # full compact 最怕“旧主题还很多，新主题刚刚出现”，
    # 这时如果只按全局频率抽摘要，旧主题很容易把当前主线盖过去。
    # 这里专门从最近 user / assistant 文本里抓“当前主问题 / 当前目标 / 当前风险”。
    captured: dict[str, str] = {}
    markers = ("当前主问题", "当前目标", "当前风险")

    for message in reversed(messages):
        role = str(message.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        for marker in markers:
            if marker in captured:
                continue
            candidate = _extract_focus_marker_clause(content, marker, markers)
            if candidate:
                captured[marker] = candidate
        if len(captured) == len(markers):
            break

    snapshot: CompactMemorySnapshot = {}
    if "当前主问题" in captured:
        snapshot["active_tasks"] = [captured["当前主问题"]]
    if "当前目标" in captured:
        snapshot["decisions"] = [captured["当前目标"]]
    if "当前风险" in captured:
        snapshot["open_issues"] = [captured["当前风险"]]
    return snapshot


def _has_current_focus_signal(messages: list[ChatMessage]) -> bool:
    """兜底识别“当前任务/当前主问题”等字样，辅助 full compact 清理旧主题工具发现。"""
    markers = ("当前任务", "当前主问题", "当前目标", "当前风险")
    for message in reversed(messages):
        role = str(message.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = " ".join(str(message.get("content", "")).split())
        if any(marker in content for marker in markers):
            return True
    return False


def _extract_focus_marker_clause(
    content: str,
    marker: str,
    markers: tuple[str, ...],
) -> str:
    # 优先提取“标题: 内容”这种明确声明，拿不到再退回到 marker 起始后的截断片段。
    # 这样既能保住结构化表达，也能兼容用户自然语言里混着写“当前主问题”的情况。
    pattern = re.compile(
        rf"({re.escape(marker)}\s*[：:]\s*.*?(?:[。！？!?]|$))"
    )
    match = pattern.search(content)
    if match:
        candidate = match.group(1).strip(" -")
    else:
        start = content.find(marker)
        if start < 0:
            return ""
        candidate = content[start:].strip(" -")
        candidate = candidate[:120]

    if not candidate:
        return ""

    if any(other != marker and other in candidate for other in markers):
        return ""
    # 同一句里混进多个 marker 时，通常说明截取边界已经串段，
    # 直接丢掉比带着混杂语义进入 full compact 更安全。
    return _limit_summary_text(candidate, 120)


def _estimate_tool_result_tokens(messages: list[ChatMessage]) -> int:
    """估算当前消息列表中 tool_result 的 token 占用。"""
    return estimate_messages_tokens(
        [message for message in messages if message.get("role") == "tool_result"]
    )


def _count_repeated_scan_results(messages: list[ChatMessage]) -> int:
    """统计 recent window 中重复扫描类结果的数量。"""
    seen: set[tuple[str, str]] = set()
    repeated = 0

    for message in messages:
        if message.get("role") != "tool_result":
            continue
        tool_name = str(message.get("tool_name", "")).strip()
        if tool_name not in {"glob_files", "grep_files", "read_file"}:
            continue
        content = str(message.get("content", "")).strip().lower()
        signature = (tool_name, _limit_summary_text(content, 120))
        if signature in seen:
            repeated += 1
            continue
        seen.add(signature)

    return repeated


def _dedupe_points(points: list[str]) -> list[str]:
    """对摘要 bullet 做轻量去重。"""
    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        normalized = " ".join(point.strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(point)
    return deduped


# ── 上下文溢出兜底（原 reactive_compact.py） ──

_OVERFLOW_ERROR_PATTERNS = (
    "prompt too long",
    "context length",
    "maximum context length",
    "too many tokens",
    "token limit",
    "exceeds the context",
    "上下文",
    "超长",
)


@dataclass(slots=True)
class ReactiveCompactResult:
    """保存一次恢复性压缩的结果。"""

    messages: list[ChatMessage]
    recovered: bool = False
    strategy: str = "none"
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_freed_estimate: int = 0


def is_context_overflow_error(error: BaseException | str) -> bool:
    """判断异常是否属于上下文过长一类的可恢复错误。"""
    error_text = str(error).lower()
    return any(pattern in error_text for pattern in _OVERFLOW_ERROR_PATTERNS)


def recover_from_context_overflow(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
) -> ReactiveCompactResult:
    """
    模型报上下文过长后，先强制走一次 full compact。
    如果仍然不够小，再退化到更激进的最近尾部保留。
    """
    current_messages = [dict(message) for message in messages]
    tokens_before = estimate_messages_tokens(current_messages)

    full_result = run_auto_compact(
        messages=current_messages,
        usable_budget=usable_budget,
        summary_base="",
        fixed_overhead_tokens=0,
        force_full=True,
    )
    if full_result.tokens_after < tokens_before:
        recovered_messages = _prepend_reactive_marker(full_result.messages)
        tokens_after = estimate_messages_tokens(recovered_messages)
        return ReactiveCompactResult(
            messages=recovered_messages,
            recovered=True,
            strategy=f"reactive_{full_result.strategy}",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_freed_estimate=max(0, tokens_before - tokens_after),
        )

    fallback_messages = _aggressive_tail_recover(
        messages=current_messages,
        usable_budget=usable_budget,
    )
    tokens_after = estimate_messages_tokens(fallback_messages)
    return ReactiveCompactResult(
        messages=fallback_messages,
        recovered=tokens_after < tokens_before,
        strategy="reactive_tail",
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_freed_estimate=max(0, tokens_before - tokens_after),
    )


def _prepend_reactive_marker(messages: list[ChatMessage]) -> list[ChatMessage]:
    """补一条恢复标记，便于日志和调试观察。"""
    marker: ChatMessage = {
        "role": "system",
        "content": "[恢复压缩]\n上一轮请求触发了上下文超长，当前已重建上下文后再次重试。",
    }
    result = [dict(message) for message in messages]
    result.insert(0, marker)
    return result


def _aggressive_tail_recover(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
) -> list[ChatMessage]:
    """兜底方案：只保留最近极小的一段上下文。"""
    system_messages = _select_reactive_system_messages(messages)
    other_messages: list[ChatMessage] = []
    for message in messages:
        if message.get("role") != "system":
            other_messages.append(dict(message))

    keep_limit = min(len(other_messages), 1)
    target_tokens = max(1, int(usable_budget * 0.12))
    kept_reversed: list[ChatMessage] = []

    for message in reversed(other_messages):
        kept_reversed.append(dict(message))
        if len(kept_reversed) >= keep_limit:
            break

    if not kept_reversed and other_messages:
        kept_reversed.append(dict(other_messages[-1]))

    while estimate_messages_tokens(list(reversed(kept_reversed))) > target_tokens and len(kept_reversed) > 1:
        kept_reversed.pop()
        if not kept_reversed and other_messages:
            kept_reversed.append(dict(other_messages[-1]))
            break

    kept_tail = list(reversed(kept_reversed))
    result = list(system_messages)
    result.append(
        {
            "role": "system",
            "content": "[恢复压缩]\n为了避开模型上下文上限，较早对话已被强制裁剪，只保留最近必要片段。",
        }
    )
    result.extend(kept_tail)
    return result


def _select_reactive_system_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """
    Reactive recover 只保留基础 system prompt。

    前一轮如果已经插入过 compact / recover marker，再把它们原样带进来会导致
    fallback 体积不降反升，进而让重试分支失效。
    """
    system_messages = [dict(message) for message in messages if message.get("role") == "system"]
    if not system_messages:
        return []

    for message in system_messages:
        if not _is_internal_compaction_marker(message):
            return [message]
    return [system_messages[0]]


def _is_internal_compaction_marker(message: ChatMessage) -> bool:
    """兼容旧调用点，实际委托给共享判定逻辑。"""
    return is_internal_compaction_marker(message)
