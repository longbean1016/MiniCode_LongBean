from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal

"""模型适配层，负责把统一消息协议转成具体模型调用。"""

from openai import OpenAI

from app.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.logger import log_event
from app.message_bridge import (
    build_openai_messages,
    build_openai_tools,
    parse_openai_response_message,
)
from app.retry import RetryPolicy, run_with_retry, should_retry_model_error
from app.tooling import ToolRegistry
from app.types import AgentStep, ChatMessage


@dataclass(slots=True)
class StreamChunk:
    """流式响应的单个 chunk。

    OpenAI streaming API 把 tool_calls 拆成多次 delta 推送，
    所以这里使用增量模式而非一次性返回完整结构。

    type 含义：
    - "text":            模型正在输出文本片段
    - "tool_call_name":  工具调用的 id 和函数名（第一块）
    - "tool_call_args":  工具调用的 arguments JSON 增量（后续块）
    """

    type: Literal["text", "tool_call_name", "tool_call_args"]
    text: str = ""
    tool_id: str = ""
    tool_index: int = 0


class OpenAIModelAdapter:
    """
    模型适配器。

    职责：
    1. 把内部消息格式转换成 OpenAI-compatible 接口需要的格式
    2. 发起聊天模型请求
    3. 把模型返回整理成统一的 AgentStep

    当前额外补了一层轻量可靠性保护：
    - 临时网络故障时有限次重试
    - 连续失败过多时短时间熔断，避免每轮都反复卡住
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        tool_registry: ToolRegistry,
        *,
        retry_max_attempts: int = 3,
        retry_base_delay_seconds: float = 0.8,
        retry_backoff_multiplier: float = 2.0,
        retry_max_delay_seconds: float = 4.0,
        circuit_failure_threshold: int = 3,
        circuit_recovery_timeout_seconds: float = 30.0,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.tool_registry = tool_registry

        # 主模型是关键链路，但仍然需要有限次重试来扛住临时抖动。
        self.retry_policy = RetryPolicy(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay_seconds,
            backoff_multiplier=retry_backoff_multiplier,
            max_delay_seconds=retry_max_delay_seconds,
        )

        # 如果主模型连续失败太多次，就短时间熔断。
        # 这样可以避免用户每问一次，都要重新经历一轮长时间超时。
        self.circuit_breaker = CircuitBreaker(
            name="chat_model",
            failure_threshold=circuit_failure_threshold,
            recovery_timeout_seconds=circuit_recovery_timeout_seconds,
        )

    def next(
        self,
        messages: list[ChatMessage],
        on_stream_chunk=None,
        store=None,
    ) -> AgentStep:  # type: ignore
        """
        调用聊天模型并返回统一的 AgentStep。
        """

        # 这里先统一做协议翻译，再进入具体模型调用。
        # agent_loop 不需要知道 OpenAI 要求什么 message/tool 结构，
        # 这样后面替换模型提供商时，主循环可以尽量不动。
        openai_messages = build_openai_messages(messages)
        openai_tools = build_openai_tools(self.tool_registry.list_tools())

        # 熔断检查放在真正发请求前，避免每轮都再白白经历一次网络超时。
        if not self.circuit_breaker.allow_request():
            raise CircuitOpenError(self.circuit_breaker.reject_reason())

        def _request_model() -> Any:
            # tools 始终一并传入，让模型自己决定“回答”还是“发起工具调用”。
            # 这样主循环只处理统一 AgentStep，不需要先分岔出工具模式与非工具模式。
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,  # type: ignore
                tools=openai_tools,  # type: ignore
                extra_body={"thinking": {"type": "disabled"}},
            )

        try:
            response = run_with_retry(
                _request_model,
                policy=self.retry_policy,
                should_retry=should_retry_model_error,
                on_retry=lambda attempt, error, delay: log_event(
                    (
                        f"聊天模型调用失败，准备第 {attempt + 1} 次尝试："
                        f"{type(error).__name__}: {error}，"
                        f"等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.circuit_breaker.record_failure(error)
            raise

        # 只有真正拿到成功响应后才恢复熔断状态，
        # 否则短暂的半失败/解析失败也会把线路错误地标记成健康。
        self.circuit_breaker.record_success()
        return parse_openai_response_message(response.choices[0].message)

    def stream_chat(
        self,
        messages: list[ChatMessage],
    ) -> Iterator[StreamChunk]:
        """流式调用聊天模型，逐 chunk yield 文本或工具调用增量。

        与 next() 共用同一套重试 / 熔断 / 协议翻译逻辑，
        只是把 OpenAI SDK stream=True 模式的结果拆成 StreamChunk 流。
        由于 Textual worker 在独立线程中运行，使用同步迭代器即可。
        """
        # 协议翻译与 next() 共用同一套 bridge 函数
        openai_messages = build_openai_messages(messages)
        openai_tools = build_openai_tools(self.tool_registry.list_tools())

        # 熔断检查，防止连续失败时反复发起必然超时的请求
        if not self.circuit_breaker.allow_request():
            raise CircuitOpenError(self.circuit_breaker.reject_reason())

        def _request_stream() -> Any:
            """发起流式请求，stream=True 启用逐 chunk 返回。"""
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,  # type: ignore[arg-type]
                tools=openai_tools,  # type: ignore[arg-type]
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"thinking": {"type": "disabled"}},
            )

        try:
            # 流式请求也走重试保护，临时网络抖动时自动恢复
            response_stream = run_with_retry(
                _request_stream,
                policy=self.retry_policy,
                should_retry=should_retry_model_error,
                on_retry=lambda attempt, error, delay: log_event(
                    (
                        f"聊天模型流式调用失败，准备第 {attempt + 1} 次尝试："
                        f"{type(error).__name__}: {error}，等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.circuit_breaker.record_failure(error)
            raise

        # 成功拿到流式响应后才恢复熔断
        self.circuit_breaker.record_success()

        # 遍历流式响应，把每一块 delta 转成 StreamChunk
        for chunk in response_stream:
            # 用法统计（Usage）出现在最后一个 chunk 中，没有 choices 字段
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 模型返回了文本内容片段
            if delta.content:
                yield StreamChunk(type="text", text=delta.content)

            # 模型返回了工具调用增量（可能跨多个 chunk 逐步推送）
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    # 第一块通常包含 tool_call id 和函数名
                    if tc_delta.id:
                        yield StreamChunk(
                            type="tool_call_name",
                            text=tc_delta.function.name or "",
                            tool_id=tc_delta.id,
                            tool_index=tc_delta.index or 0,
                        )
                    # 后续块追加 arguments JSON 片段
                    if tc_delta.function and tc_delta.function.arguments:
                        yield StreamChunk(
                            type="tool_call_args",
                            text=tc_delta.function.arguments,
                            tool_index=tc_delta.index or 0,
                        )
