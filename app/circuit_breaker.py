from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


class CircuitOpenError(RuntimeError):
    """
    熔断器处于打开状态时抛出的异常。
    """


@dataclass(slots=True)
class CircuitBreaker:
    """
    一个非常轻量的熔断器。

    目标不是完美复刻分布式中间件里的熔断状态机，
    而是给 CLI agent 增加一个“连续失败后短暂别再打同一条坏链路”的保护层。
    """

    name: str
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 45.0
    failure_count: int = 0
    opened_at: float | None = None
    last_error: str = ""
    last_failure_at: float = 0.0
    half_open_in_flight: bool = False
    _clock: Callable[[], float] = field(default=time.time, repr=False)

    def allow_request(self) -> bool:
        """
        判断当前是否允许继续发请求。
        """

        if self.opened_at is None:
            return True

        now = self._clock()
        cooldown_elapsed = (now - self.opened_at) >= max(0.0, self.recovery_timeout_seconds)
        if not cooldown_elapsed:
            return False

        # 冷却期结束后进入一次 half-open 探测。
        # 这里只允许一个探测请求在飞，避免刚恢复时并发打出一串请求把下游再次压垮。
        if self.half_open_in_flight:
            return False

        self.half_open_in_flight = True
        return True

    def record_success(self) -> None:
        """
        一次请求成功后，关闭熔断状态并清空失败计数。
        """

        # 任何一次成功都说明链路至少暂时恢复了，直接清空失败状态。
        self.failure_count = 0
        self.opened_at = None
        self.last_error = ""
        self.last_failure_at = 0.0
        self.half_open_in_flight = False

    def record_failure(self, error: Exception) -> None:
        """
        记录一次失败；超过阈值后进入打开状态。
        """

        # 失败时无论当前是否处于 half-open，都要把 in-flight 标记清掉，
        # 否则后续恢复窗口到了也可能一直卡在“已有探测请求”状态。
        self.failure_count += 1
        self.last_error = str(error)
        self.last_failure_at = self._clock()
        self.half_open_in_flight = False

        # 达到阈值后才真正打开熔断，避免一次偶发超时就把整个能力关掉。
        if self.failure_count >= max(1, self.failure_threshold):
            self.opened_at = self.last_failure_at

    def reject_reason(self) -> str:
        """
        返回当前拒绝请求时的说明文案。
        """

        if self.opened_at is None:
            return ""

        return (
            f"{self.name} 熔断中：最近连续失败 {self.failure_count} 次，"
            f"冷却 {self.recovery_timeout_seconds:.0f} 秒后再尝试。"
        )
