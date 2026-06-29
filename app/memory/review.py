from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from app.agent.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.agent.retry import RetryPolicy, run_with_retry, should_retry_model_error
from app.logger import log_event
from app.memory.memory_store import MemoryStore
from app.memory.memory_tool import get_memory_store
from app.types import ChatMessage


REVIEW_PROMPT = """
你是代码 Agent 的后台反思器。
请审阅完整对话历史，只提炼真正值得跨会话保留的持久记忆。

输出规则：
1. 用户身份 / 偏好 / 工作方式 -> 写到 user
2. 项目规范 / 环境约束 / 执行教训 -> 写到 memory
3. 没有新内容时只输出：Nothing to save.

如果要保存，请只输出 JSON 数组，每个元素形如：
{"action":"add","target":"user|memory","content":"- 具体记忆条目"}

约束：
- content 必须是单条 Markdown 列表项，且以 "- " 开头
- 只保留长期稳定信息，不要写本轮临时过程
- 不要输出额外解释
""".strip()


@dataclass(slots=True)
class BackgroundReviewRunner:
    api_key: str
    base_url: str
    model_name: str
    memory_store: MemoryStore
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 0.8
    retry_backoff_multiplier: float = 2.0
    retry_max_delay_seconds: float = 4.0
    circuit_failure_threshold: int = 3
    circuit_recovery_timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.retry_policy = RetryPolicy(
            max_attempts=self.retry_max_attempts,
            base_delay_seconds=self.retry_base_delay_seconds,
            backoff_multiplier=self.retry_backoff_multiplier,
            max_delay_seconds=self.retry_max_delay_seconds,
        )
        self.circuit_breaker = CircuitBreaker(
            name="background_review",
            failure_threshold=self.circuit_failure_threshold,
            recovery_timeout_seconds=self.circuit_recovery_timeout_seconds,
        )

    def spawn_review(self, history: list[ChatMessage], *, session_id: str = "") -> None:
        snapshot = [dict(message) for message in history]
        thread = threading.Thread(
            target=self._run_review_thread,
            args=(snapshot, session_id),
            name=f"memory-review-{session_id or 'session'}",
            daemon=True,
        )
        thread.start()

    def _run_review_thread(self, history: list[ChatMessage], session_id: str) -> None:
        try:
            payload = self._build_history_payload(history)
            response_text = self._call_review_model(payload)
            if response_text.strip() == "Nothing to save.":
                return
            operations = self._parse_operations(response_text)
            if not operations:
                return
            for operation in operations:
                if operation.get("action") != "add":
                    continue
                target = str(operation.get("target", "")).strip().lower()
                content = str(operation.get("content", "")).rstrip()
                if target not in {"memory", "user"} or not content:
                    continue
                result = self.memory_store.add(
                    target=target,
                    content=content,
                    bypass_approval=True,
                )
                if not bool(result.get("success", False)):
                    log_event(
                        f"[session={session_id or '-'}] 后台反思写入失败: {result.get('error', '')}",
                        echo=False,
                    )
        except Exception as error:
            log_event(
                f"[session={session_id or '-'}] 后台反思线程失败: {error}",
                echo=False,
            )

    def _call_review_model(self, payload: str) -> str:
        if not self.circuit_breaker.allow_request():
            raise CircuitOpenError(self.circuit_breaker.reject_reason())

        def _request_model() -> object:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": REVIEW_PROMPT},
                    {"role": "user", "content": payload},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )

        try:
            response = run_with_retry(
                _request_model,
                policy=self.retry_policy,
                should_retry=should_retry_model_error,
                on_retry=lambda attempt, error, delay: log_event(
                    (
                        f"后台反思调用失败，准备第 {attempt + 1} 次尝试："
                        f"{type(error).__name__}: {error}，等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.circuit_breaker.record_failure(error)
            raise

        self.circuit_breaker.record_success()
        return str(response.choices[0].message.content or "")

    def _build_history_payload(self, history: list[ChatMessage]) -> str:
        lines: list[str] = []
        for message in history:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant_tool_call":
                tool_name = str(message.get("tool_name", "")).strip() or "unknown_tool"
                lines.append(f"[assistant_tool_call:{tool_name}] {content[:240]}")
            elif role == "tool_result":
                tool_name = str(message.get("tool_name", "")).strip() or "unknown_tool"
                prefix = "tool_error" if bool(message.get("is_error", False)) else "tool_result"
                lines.append(f"[{prefix}:{tool_name}] {content[:320]}")
            else:
                lines.append(f"[{role}] {content[:480]}")
        return "\n".join(lines).strip()

    def _parse_operations(self, text: str) -> list[dict[str, Any]]:
        normalized = text.strip()
        if not normalized:
            return []
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        operations: list[dict[str, Any]] = []
        for item in parsed:
            if isinstance(item, dict):
                operations.append(dict(item))
        return operations


def build_review_runner(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    memory_store: MemoryStore | None = None,
    retry_max_attempts: int = 3,
    retry_base_delay_seconds: float = 0.8,
    retry_backoff_multiplier: float = 2.0,
    retry_max_delay_seconds: float = 4.0,
    circuit_failure_threshold: int = 3,
    circuit_recovery_timeout_seconds: float = 45.0,
) -> BackgroundReviewRunner:
    return BackgroundReviewRunner(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        memory_store=memory_store or get_memory_store(),
        retry_max_attempts=retry_max_attempts,
        retry_base_delay_seconds=retry_base_delay_seconds,
        retry_backoff_multiplier=retry_backoff_multiplier,
        retry_max_delay_seconds=retry_max_delay_seconds,
        circuit_failure_threshold=circuit_failure_threshold,
        circuit_recovery_timeout_seconds=circuit_recovery_timeout_seconds,
    )
