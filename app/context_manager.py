from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.types import ChatMessage


# 这里沿用 minicode 的简化估算思路：
# - 中文按 1.5 个字符约等于 1 个 token
# - 英文按 4 个字符约等于 1 个 token
_CJK_PATTERN = re.compile(r"[\u4E00-\u9FFF]")

_ROLE_OVERHEAD = {
    "system": 3,
    "user": 4,
    "assistant": 3,
    "assistant_tool_call": 7,
    "tool_result": 6,
    "assistant_progress": 3,
}

# 默认可用上下文预算改成 128k，更接近 minicode 当前对大多数模型的保守配置。
DEFAULT_USABLE_CONTEXT_BUDGET = 128_000


@dataclass(slots=True)
class ContextStats:
    """保存一次模型请求前的上下文统计信息。"""

    usable_budget: int
    total_tokens: int
    usage_ratio: float
    system_tokens: int
    recent_tokens: int
    memory_tokens: int
    tool_result_tokens: int
    message_count: int
    tool_result_count: int


@dataclass(slots=True)
class ContextPolicy:
    """根据上下文压力得到的裁剪和注入策略。"""

    level: int
    keep_rounds: int
    memory_top_k: int
    retrieval_top_k: int
    memory_item_chars: int
    max_recent_tool_results: int
    truncate_tool_result_chars: int


def estimate_tokens(text: str) -> int:
    """按中英文字符数粗略估算 token。"""
    if not text:
        return 0

    cjk_count = len(_CJK_PATTERN.findall(text))
    ascii_count = len(text) - cjk_count
    return max(1, int(cjk_count / 1.5 + ascii_count / 4.0))


def estimate_message_tokens(message: ChatMessage) -> int:
    """估算单条消息的 token，包括 role 固定开销。"""
    role = str(message.get("role", ""))
    tokens = _ROLE_OVERHEAD.get(role, 3)

    content = message.get("content", "")
    if isinstance(content, str):
        tokens += estimate_tokens(content)

    tool_name = message.get("tool_name", "")
    if isinstance(tool_name, str) and tool_name:
        tokens += estimate_tokens(tool_name)

    if "input" in message:
        raw_input = message["input"]
        if isinstance(raw_input, dict):
            input_text = json.dumps(raw_input, ensure_ascii=False)
        else:
            input_text = str(raw_input)
        tokens += estimate_tokens(input_text)

    return tokens


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    """估算一组消息的总 token。"""
    return sum(estimate_message_tokens(message) for message in messages)


def collect_context_stats(
    *,
    system_prompt: str,
    recent_messages: list[ChatMessage],
    memory_context: str,
    usable_budget: int,
) -> ContextStats:
    """
    统计一次模型请求前的上下文占用。

    这里把 system prompt 和记忆上下文拆开统计，
    方便后面观察“到底是 recent history 胖，还是 memory 注入太多”。
    """
    system_tokens = estimate_message_tokens({"role": "system", "content": system_prompt})
    recent_tokens = estimate_messages_tokens(recent_messages)
    memory_tokens = estimate_tokens(memory_context)

    tool_result_tokens = 0
    tool_result_count = 0
    for message in recent_messages:
        if message.get("role") != "tool_result":
            continue

        tool_result_count += 1
        tool_result_tokens += estimate_message_tokens(message)

    total_tokens = system_tokens + recent_tokens + memory_tokens
    usage_ratio = 0.0
    if usable_budget > 0:
        usage_ratio = total_tokens / usable_budget

    return ContextStats(
        usable_budget=usable_budget,
        total_tokens=total_tokens,
        usage_ratio=usage_ratio,
        system_tokens=system_tokens,
        recent_tokens=recent_tokens,
        memory_tokens=memory_tokens,
        tool_result_tokens=tool_result_tokens,
        message_count=1 + len(recent_messages),
        tool_result_count=tool_result_count,
    )


def decide_context_policy(stats: ContextStats) -> ContextPolicy:
    """根据上下文占用比例决定压缩等级和记忆注入预算。"""
    ratio = stats.usage_ratio

    if ratio >= 0.80:
        return ContextPolicy(
            level=3,
            keep_rounds=3,
            memory_top_k=1,
            retrieval_top_k=2,
            memory_item_chars=80,
            max_recent_tool_results=2,
            truncate_tool_result_chars=800,
        )
    if ratio >= 0.65:
        return ContextPolicy(
            level=2,
            keep_rounds=4,
            memory_top_k=2,
            retrieval_top_k=4,
            memory_item_chars=110,
            max_recent_tool_results=3,
            truncate_tool_result_chars=1600,
        )
    if ratio >= 0.45:
        return ContextPolicy(
            level=1,
            keep_rounds=5,
            memory_top_k=3,
            retrieval_top_k=6,
            memory_item_chars=140,
            max_recent_tool_results=4,
            truncate_tool_result_chars=2500,
        )
    return ContextPolicy(
        level=0,
        keep_rounds=6,
        memory_top_k=4,
        retrieval_top_k=8,
        memory_item_chars=180,
        max_recent_tool_results=5,
        truncate_tool_result_chars=4000,
    )
