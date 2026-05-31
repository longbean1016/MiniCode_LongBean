from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

"""通用重试工具，封装带回调的重试执行逻辑。"""


T = TypeVar("T")


@dataclass(slots=True)
class RetryPolicy:
    """
    描述一次重试策略。

    这里只描述“怎么重试”，不描述“什么错误值得重试”。
    是否可重试由 should_retry 回调决定，这样同一套重试器可以复用到模型、向量索引等不同场景。
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.8
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 5.0


def should_retry_model_error(error: Exception) -> bool:
    """
    判断聊天模型错误是否值得重试。
    """

    # 当前用的是基于错误文本的轻量分类，而不是绑定某一家 SDK 的异常类型层次。
    # 这样兼容性更高，切模型网关或 SDK 时通常不需要重写整套重试判断。
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
    # 鉴权/请求格式错误属于“再试也不会好”的确定性失败，要尽快暴露给上层。
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

    # 向量链路的可重试条件和模型链路不同：
    # 例如维度错误通常是配置问题，不应重试；连接拒绝则往往值得等一等再试。
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
            # 是否继续重试同时受两层约束：
            # 1. 尝试次数还没用完
            # 2. 错误类型被判定为可重试
            can_retry = attempt < max(1, policy.max_attempts) and should_retry(error)
            if not can_retry:
                raise

            if on_retry is not None:
                on_retry(attempt, error, delay_seconds)

            # 退避等待放在 attempt 递增之前，确保回调里拿到的是“本次失败后将等待多久”。
            time.sleep(delay_seconds)
            attempt += 1
            delay_seconds = min(
                max(0.0, float(policy.max_delay_seconds)),
                delay_seconds * max(1.0, float(policy.backoff_multiplier)),
            )
