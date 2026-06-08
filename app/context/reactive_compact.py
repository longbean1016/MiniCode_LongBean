from __future__ import annotations

from dataclasses import dataclass

"""上下文溢出兜底模块，在请求失败后尝试做应急压缩恢复。"""

from app.context.auto_compact import run_auto_compact
from app.context.manager import estimate_messages_tokens
from app.context.message_safety import is_internal_compaction_marker
from app.types import ChatMessage

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
