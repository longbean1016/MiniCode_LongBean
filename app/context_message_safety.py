from __future__ import annotations

from app.types import ChatMessage


def is_internal_compaction_marker(message: ChatMessage) -> bool:
    """识别上下文治理内部生成的 system marker。"""
    if message.get("role") != "system":
        return False
    content = str(message.get("content", "")).strip()
    return (
        content.startswith("[会话记忆压缩]")
        or content.startswith("[全量压缩]")
        or content.startswith("[恢复压缩]")
    )


def normalize_tool_call_pairs(messages: list[ChatMessage]) -> list[ChatMessage]:
    """
    清理被压缩过程打断的工具调用对。

    OpenAI 协议要求 tool_result 必须紧跟在其对应 assistant tool_calls 之后。
    一旦 compact 折掉了前置 assistant_tool_call，后续请求就会直接 400。
    这里统一丢弃孤立的 assistant_tool_call / tool_result，保证出站消息可发送。
    """
    assistant_ids = {
        str(message.get("tool_use_id", "")).strip()
        for message in messages
        if message.get("role") == "assistant_tool_call"
        and str(message.get("tool_use_id", "")).strip()
    }
    tool_result_ids = {
        str(message.get("tool_use_id", "")).strip()
        for message in messages
        if message.get("role") == "tool_result"
        and str(message.get("tool_use_id", "")).strip()
    }
    valid_ids = assistant_ids & tool_result_ids

    normalized: list[ChatMessage] = []
    for message in messages:
        role = message.get("role")
        if role not in {"assistant_tool_call", "tool_result"}:
            normalized.append(dict(message))
            continue

        tool_use_id = str(message.get("tool_use_id", "")).strip()
        if tool_use_id and tool_use_id in valid_ids:
            normalized.append(dict(message))

    return normalized
