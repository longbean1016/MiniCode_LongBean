from __future__ import annotations

from dataclasses import dataclass

from app.context_manager import estimate_messages_tokens
from app.types import ChatMessage

AUTO_COMPACT_TRIGGER_RATIO = 0.85
AUTO_COMPACT_TARGET_RATIO = 0.78
AUTO_COMPACT_SESSION_TARGET_RATIO = 0.58
AUTO_COMPACT_FULL_TARGET_RATIO = 0.42
AUTO_COMPACT_MIN_KEEP_MESSAGES = 3
AUTO_COMPACT_MIN_FULL_KEEP_MESSAGES = 2
AUTO_COMPACT_SESSION_SUMMARY_MAX_CHARS = 220
AUTO_COMPACT_FULL_SUMMARY_MAX_CHARS = 320
AUTO_COMPACT_MIN_SUMMARY_CHARS = 80


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


def should_trigger_auto_compact(*, total_tokens: int, usable_budget: int) -> bool:
    """达到高水位时触发自动压缩。"""
    if usable_budget <= 0:
        return False
    return total_tokens >= int(usable_budget * AUTO_COMPACT_TRIGGER_RATIO)


def run_auto_compact(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
    summary_base: str,
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

    if not force_full and not should_trigger_auto_compact(
        total_tokens=tokens_before,
        usable_budget=usable_budget,
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
            fixed_overhead_tokens=fixed_overhead_tokens,
        )
        if session_result.applied:
            return session_result

    return _run_full_compact(
        messages=current_messages,
        usable_budget=usable_budget,
        summary_base=summary_base,
        fixed_overhead_tokens=fixed_overhead_tokens,
    )


def _run_session_memory_compact(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
    summary_base: str,
    fixed_overhead_tokens: int,
) -> AutoCompactResult:
    """优先用已有摘要和工作记忆做一次轻量压缩。"""
    normalized_summary = _limit_summary_text(
        summary_base.strip(),
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
    )


def _run_full_compact(
    *,
    messages: list[ChatMessage],
    usable_budget: int,
    summary_base: str,
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
    removed_messages = other_messages[: max(0, len(other_messages) - len(tail_messages))]
    removed_count = max(0, len(other_messages) - len(tail_messages))
    summary_text = _limit_summary_text(
        _build_structured_summary(
            removed_messages=removed_messages,
            summary_base=summary_base,
        ),
        AUTO_COMPACT_FULL_SUMMARY_MAX_CHARS,
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

        if len(summary_current) > AUTO_COMPACT_MIN_SUMMARY_CHARS:
            next_limit = max(
                AUTO_COMPACT_MIN_SUMMARY_CHARS,
                int(len(summary_current) * 0.7),
            )
            next_summary = _limit_summary_text(summary_current, next_limit)
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


def _build_structured_summary(
    *,
    removed_messages: list[ChatMessage],
    summary_base: str,
) -> str:
    """构造一个不依赖模型调用的中文紧凑摘要。"""
    parts: list[str] = []
    normalized_base = summary_base.strip()
    if normalized_base:
        parts.append("已有摘要：")
        parts.append(normalized_base[:160])

    user_points: list[str] = []
    assistant_points: list[str] = []
    tool_points: list[str] = []
    error_points: list[str] = []

    for message in removed_messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip().replace("\n", " ")
        if not content:
            continue

        if role == "user":
            user_points.append(content[:80])
            continue
        if role == "assistant":
            assistant_points.append(content[:80])
            continue
        if role == "assistant_tool_call":
            tool_name = str(message.get("tool_name", "")) or "unknown"
            tool_points.append(f"调用工具：{tool_name}")
            continue
        if role == "tool_result":
            tool_name = str(message.get("tool_name", "")) or "unknown"
            preview = content[:60]
            if message.get("is_error"):
                error_points.append(f"{tool_name}：{preview}")
            else:
                tool_points.append(f"{tool_name}：{preview}")

    if user_points:
        parts.append("用户关注点：")
        parts.extend(f"- {point}" for point in user_points[:3])
    if assistant_points:
        parts.append("已形成的中间结论：")
        parts.extend(f"- {point}" for point in assistant_points[:3])
    if tool_points:
        parts.append("关键工具轨迹：")
        parts.extend(f"- {point}" for point in tool_points[:4])
    if error_points:
        parts.append("需要记住的错误：")
        parts.extend(f"- {point}" for point in error_points[:3])

    summary = "\n".join(parts).strip()
    if summary:
        return summary
    return "较早的对话已经压缩，只保留后续继续推理所需的关键信息。"


def _limit_summary_text(text: str, max_chars: int) -> str:
    """限制压缩摘要长度，避免摘要本身反过来撑爆上下文。"""
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    suffix = " ...[摘要已截断]"
    if max_chars <= len(suffix):
        return normalized[:max_chars]
    head_limit = max_chars - len(suffix)
    return f"{normalized[:head_limit]}{suffix}"
