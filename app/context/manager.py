from __future__ import annotations

import json
import re
from dataclasses import dataclass

"""上下文管理器，维护消息列表和基础 token 使用统计。"""

from app.state.session import SessionData
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
    """保存一次模型请求前的上下文统计信息（仅保留实际使用的字段）。"""

    usable_budget: int
    total_tokens: int
    usage_ratio: float


@dataclass(slots=True)
class ContextPolicy:
    """根据上下文压力得到的裁剪和注入策略。"""

    level: int
    keep_rounds: int
    memory_top_k: int
    retrieval_top_k: int
    memory_item_chars: int
    max_recent_tool_results: int


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
    """估算一组消息的总 token。

       对标 Claude Code tokenCountWithEstimation()：
       从后往前找最近一次 API 返回的真实 token 数作为基线，
       仅对基线之后的新增消息做粗略估算（字符数 ÷ 4）。
    """
    # ── 从后往前找有真实 token 数据的 assistant 消息 ──
    baseline_tokens = 0
    baseline_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant":
            stored = msg.get("_api_total_tokens")
            if isinstance(stored, (int, float)) and stored > 0:
                baseline_tokens = int(stored)
                baseline_idx = i
                break

    # ── 没有基线 → 全部估算 ──
    if baseline_idx < 0:
        return sum(estimate_message_tokens(m) for m in messages)

    # ── 基线 + 新增消息估算 ──
    new_messages = messages[baseline_idx + 1:]
    return baseline_tokens + sum(estimate_message_tokens(m) for m in new_messages)


def collect_context_stats(
    *,
    system_prompt: str,
    recent_messages: list[ChatMessage],
    memory_context: str,
    usable_budget: int,
) -> ContextStats:
    """统计一次模型请求前的上下文 token 占用和压力比。"""
    total_tokens = (
        estimate_message_tokens({"role": "system", "content": system_prompt}) +
        estimate_messages_tokens(recent_messages) +
        estimate_tokens(memory_context)
    )
    usage_ratio = total_tokens / usable_budget if usable_budget > 0 else 0.0

    return ContextStats(
        usable_budget=usable_budget,
        total_tokens=total_tokens,
        usage_ratio=usage_ratio,
    )


def decide_context_policy(
    stats: ContextStats | None = None,
    *,
    analysis_mode: bool = False,
) -> ContextPolicy:
    """返回固定的上下文策略。256K 上下文下无需动态收紧。"""
    if analysis_mode:
        return ContextPolicy(
            level=0,
            keep_rounds=8,
            memory_top_k=4,
            retrieval_top_k=8,
            memory_item_chars=180,
            max_recent_tool_results=8,
        )
    return ContextPolicy(
        level=0,
        keep_rounds=6,
        memory_top_k=4,
        retrieval_top_k=8,
        memory_item_chars=180,
        max_recent_tool_results=5,
    )


@dataclass(slots=True)
class CompactionPolicy:
    keep_rounds: int
    min_round_delta_for_resummarize: int
    level: int


def build_compaction_policy(session: SessionData | None = None) -> CompactionPolicy:
    """返回固定的压缩策略。"""
    return CompactionPolicy(keep_rounds=6, min_round_delta_for_resummarize=2, level=0)
