from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from app.logger import log_event


_Task = tuple[str, Callable[[], None]]

_TASKS: queue.Queue[_Task] = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()


def _worker_loop() -> None:
    while True:
        name, fn = _TASKS.get()
        try:
            fn()
        except Exception as error:
            log_event(f"[background:{name}] 任务执行失败: {error}", echo=False)
        finally:
            _TASKS.task_done()


def _ensure_worker_started() -> None:
    global _WORKER_STARTED

    if _WORKER_STARTED:
        return

    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return

        worker = threading.Thread(
            target=_worker_loop,
            name="minicode-background-worker",
            daemon=True,
        )
        worker.start()
        _WORKER_STARTED = True


def submit_background(fn: Callable[[], None], *, name: str = "background") -> None:
    """提交一个后台任务，由单线程顺序执行。"""
    _ensure_worker_started()
    _TASKS.put((name, fn))


def wait_for_background_tasks() -> None:
    """等待已提交的后台任务完成，主要用于退出收尾和测试。"""
    _TASKS.join()
