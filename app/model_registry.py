from __future__ import annotations

from typing import Any

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

        openai_messages = build_openai_messages(messages)
        openai_tools = build_openai_tools(self.tool_registry.list_tools())

        if not self.circuit_breaker.allow_request():
            raise CircuitOpenError(self.circuit_breaker.reject_reason())

        def _request_model() -> Any:
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

        self.circuit_breaker.record_success()
        return parse_openai_response_message(response.choices[0].message)
