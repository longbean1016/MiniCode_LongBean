from __future__ import annotations

from dataclasses import dataclass, field

"""消息构造器，负责按协议顺序追加 user、assistant 和 tool 消息。
   每条消息自动附带 created_at 时间戳，用于 microcompact 空闲时间触发判断。"""

import time

from app.types import ChatMessage


@dataclass(slots=True)
class MessageBuilder:
    """统一构造和追加对话消息。"""

    history: list[ChatMessage] = field(default_factory=list)

    def _now(self) -> float:
        """返回当前时间戳，所有消息统一使用此方法记录创建时间。"""
        return time.time()

    def add_user(self, content: str) -> None:
        """追加用户消息。"""
        self.history.append({
            "role": "user",
            "content": content,
            "created_at": self._now(),
        })

    def add_assistant(self, content: str) -> None:
        """追加模型普通回答。"""
        self.history.append({
            "role": "assistant",
            "content": content,
            "created_at": self._now(),
        })

    def add_progress(self, content: str) -> None:
        """追加模型中间进度。"""
        self.history.append({
            "role": "assistant_progress",
            "content": content,
            "created_at": self._now(),
        })

    def add_tool_call(self, tool_use_id: str, tool_name: str, input_data: object) -> None:
        """追加模型发起的工具调用。"""
        self.history.append({
            "role": "assistant_tool_call",
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "input": input_data,
            "created_at": self._now(),
        })

    def add_tool_result(
        self,
        tool_use_id: str,
        tool_name: str,
        content: str,
        is_error: bool = False,
        meta: dict[str, object] | None = None,
    ) -> None:
        """追加工具执行结果，并保留可选元信息。"""
        self.history.append({
            "role": "tool_result",
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "content": content,
            "is_error": is_error,
            "meta": dict(meta or {}),
            "created_at": self._now(),
        })

    def build(self) -> list[ChatMessage]:
        """返回当前完整历史。"""
        return list(self.history)

    def extend(self, messages: list[ChatMessage]) -> None:
        """批量追加已有消息。"""
        self.history.extend(messages)
