from __future__ import annotations

import time
from pathlib import Path


# 默认日志文件名，统一写到项目根目录
DEFAULT_LOG_FILE = "debug.log"


def _format_now() -> str:
    """返回当前时间的可读字符串。"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log_event(message: str, log_file: str = DEFAULT_LOG_FILE) -> None:
    """把一条日志同时输出到控制台和文件。"""
    # 拼出最终日志行，方便统一检索
    line = f"[{_format_now()}] {message}"

    # 打印到控制台，便于实时观察
    print(line)

    # 追加写入日志文件，保留排查现场
    path = Path(log_file)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
