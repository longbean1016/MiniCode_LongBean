from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class RetryPolicy:
    """
    描述一次重试策略。
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.8
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 5.0


def should_retry_model_error(error: Exception) -> bool:
    """
    判断聊天模型错误是否值得重试。
    """

    text = f"{type(error).__name__}: {error}".lower()

    non_retry_markers = [
        "401",
        "403",
        "authentication",
        "api key",
        "invalid api key",
        "invalid_request_error",
        "badrequest",
        "bad request",
    ]
    if any(marker in text for marker in non_retry_markers):
        return False

    retry_markers = [
        "timeout",
        "timed out",
        "connect",
        "connection",
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "service unavailable",
        "temporarily unavailable",
    ]
    return any(marker in text for marker in retry_markers)


def should_retry_vector_error(error: Exception) -> bool:
    """
    判断 embedding / Qdrant 错误是否值得重试。
    """

    text = f"{type(error).__name__}: {error}".lower()

    non_retry_markers = [
        "401",
        "403",
        "authentication",
        "api key",
        "400",
        "bad request",
        "not a valid point id",
        "vector dimension",
        "维度",
    ]
    if any(marker in text for marker in non_retry_markers):
        return False

    retry_markers = [
        "timeout",
        "timed out",
        "connect",
        "connection",
        "10061",
        "connection refused",
        "502",
        "503",
        "504",
        "temporarily unavailable",
        "service unavailable",
        "responsehandlingexception",
    ]
    return any(marker in text for marker in retry_markers)


def run_with_retry(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    should_retry: Callable[[Exception], bool],
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """
    统一执行一次带重试的操作。
    """

    attempt = 1
    delay_seconds = max(0.0, float(policy.base_delay_seconds))

    while True:
        try:
            return operation()
        except Exception as error:
            can_retry = attempt < max(1, policy.max_attempts) and should_retry(error)
            if not can_retry:
                raise

            if on_retry is not None:
                on_retry(attempt, error, delay_seconds)

            time.sleep(delay_seconds)
            attempt += 1
            delay_seconds = min(
                max(0.0, float(policy.max_delay_seconds)),
                delay_seconds * max(1.0, float(policy.backoff_multiplier)),
            )
