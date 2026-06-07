from __future__ import annotations

import time
from pathlib import Path


# 默认日志文件名，统一写到项目根目录。
DEFAULT_LOG_FILE = "debug.log"


def _format_now() -> str:
    """返回当前时间的可读字符串。"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log_event(
    message: str,
    log_file: str = DEFAULT_LOG_FILE,
    *,
    echo: bool = True,
) -> None:
    """
    记录一条日志。

    参数说明：
    - `message`: 要写入的日志内容
    - `log_file`: 日志文件路径，默认写入项目根目录下的 `debug.log`
    - `echo`: 是否同时回显到控制台。默认开启；某些低频运维日志可以只写文件不打扰主链路输出
    """
    line = f"[{_format_now()}] {message}"

    if echo:
        print(line)

    path = Path(log_file)
    try:
        with path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except OSError:
        # 日志是旁路诊断信息，不能因为 debug.log 权限或锁定问题阻断主流程。
        return
