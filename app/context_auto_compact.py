from __future__ import annotations

from dataclasses import dataclass

from app.context_compact_memory import (
    CompactMemorySnapshot,
    build_event_compact_memory_snapshot,
    merge_compact_memory_snapshots,
    parse_compact_memory_context,
    render_compact_memory_context,
    render_full_compact_memory_context,
)
from app.context_message_safety import is_internal_compaction_marker
from app.context_manager import estimate_messages_tokens, estimate_tokens
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


def should_trigger_auto_compact(
    *,
    total_tokens: int,
    usable_budget: int,
    tool_result_tokens: int = 0,
    repeated_scan_count: int = 0,
) -> bool:
    """达到高水位时触发自动压缩。"""
    if usable_budget <= 0:
        return False
    if total_tokens >= int(usable_budget * AUTO_COMPACT_TRIGGER_RATIO):
        return True
    if tool_result_tokens >= int(usable_budget * AUTO_COMPACT_TOOL_RESULT_TRIGGER_RATIO):
        return True
    return repeated_scan_count >= AUTO_COMPACT_REPEATED_SCAN_TRIGGER_COUNT


def run_auto_compact(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
    summary_base: str,
    summary_snapshot: CompactMemorySnapshot | None = None,
    summary_source_messages: list[ChatMessage] | None = None,
    fixed_overhead_tokens: int = 0,
    force_full: bool = False,
) -> AutoCompactResult:
    """
    运行类似 minicode 的 Auto Compact 调度。

    策略顺序：
    1. Session Memory Compact
    2. Full Compact
    """
    current_messages = [dict(message) for message in messages]
    tokens_before = fixed_overhead_tokens + estimate_messages_tokens(current_messages)
    tool_result_tokens = _estimate_tool_result_tokens(current_messages)
    repeated_scan_count = _count_repeated_scan_results(current_messages)

    if not force_full and not should_trigger_auto_compact(
        total_tokens=tokens_before,
        usable_budget=usable_budget,
        tool_result_tokens=tool_result_tokens,
        repeated_scan_count=repeated_scan_count,
    ):
        return AutoCompactResult(
            messages=current_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
        )

    if not force_full:
        session_result = _run_session_memory_compact(
            messages=current_messages,
            usable_budget=usable_budget,
            summary_base=summary_base,
            summary_snapshot=summary_snapshot,
            fixed_overhead_tokens=fixed_overhead_tokens,
        )
        if session_result.applied:
            return session_result

    return _run_full_compact(
        messages=current_messages,
        usable_budget=usable_budget,
        summary_base=summary_base,
        summary_snapshot=summary_snapshot,
        summary_source_messages=summary_source_messages,
        fixed_overhead_tokens=fixed_overhead_tokens,
    )


def _run_session_memory_compact(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
    summary_base: str,
    summary_snapshot: CompactMemorySnapshot | None,
    fixed_overhead_tokens: int,
) -> AutoCompactResult:
    """优先用已有摘要和工作记忆做一次轻量压缩。"""
    baseline_snapshot = summary_snapshot or parse_compact_memory_context(summary_base)
    snapshot_text = (
        render_compact_memory_context(baseline_snapshot).strip()
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
) -> AutoCompactResult:
    """当轻量摘要压不下去时，退化到更强的全量压缩。"""
    system_messages, other_messages = _split_system_messages(messages)
    if len(other_messages) <= AUTO_COMPACT_MIN_FULL_KEEP_MESSAGES:
        return AutoCompactResult(
            messages=messages,
            tokens_before=fixed_overhead_tokens + estimate_messages_tokens(messages),
            tokens_after=fixed_overhead_tokens + estimate_messages_tokens(messages),
        )

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
    _, summary_other_messages = _split_system_messages(summary_source_messages or messages)
    if len(summary_other_messages) >= removed_count:
        removed_messages = summary_other_messages[:removed_count]
    merged_snapshot, rendered_summary = _build_structured_summary_package(
        removed_messages=removed_messages,
        summary_base=summary_base,
        summary_snapshot=summary_snapshot,
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

    for message in reversed(messages):
        kept_reversed.append(dict(message))
        tail_tokens += estimate_messages_tokens([message])
        if len(kept_reversed) >= min_keep_messages and tail_tokens >= target_tokens:
            break

    tail_messages = list(reversed(kept_reversed))
    return _adjust_tail_for_tool_pairs(messages, tail_messages)


def _adjust_tail_for_tool_pairs(
    original_messages: list[ChatMessage],
    tail_messages: list[ChatMessage],
) -> list[ChatMessage]:
    """避免把 tool_call 和 tool_result 从中间切断。"""
    if not tail_messages:
        return tail_messages

    tail_start = len(original_messages) - len(tail_messages)
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
    summary_base: str,
    summary_snapshot: CompactMemorySnapshot | None,
) -> tuple[CompactMemorySnapshot, str]:
    """同时返回 full compact 的结构化快照和渲染文本。"""
    base_snapshot = summary_snapshot or parse_compact_memory_context(summary_base)
    if not base_snapshot and summary_base.strip():
        base_snapshot = {
            "history_summary": [
                _limit_summary_text(summary_base.strip(), 140),
            ]
        }
    event_snapshot = build_event_compact_memory_snapshot(
        removed_messages=removed_messages,
    )
    merged_snapshot = merge_compact_memory_snapshots(
        base_snapshot=base_snapshot,
        overlay_snapshot=event_snapshot,
    )
    rendered = render_full_compact_memory_context(merged_snapshot).strip()
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
        if tool_name not in {"list_files", "grep_files", "read_file"}:
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
