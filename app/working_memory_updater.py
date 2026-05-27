from __future__ import annotations

import re
from typing import Any

from app.types import ToolResult


# 用来从 run_command 命令文本里提取看起来像路径的片段。
_PATH_TOKEN_RE = re.compile(
    r'([A-Za-z]:\\[^\s"\']+|\.{0,2}[\\/][^\s"\']+|[A-Za-z0-9_.-]+[\\/][A-Za-z0-9_./\\-]+)'
)


def _shorten(text: str, max_chars: int = 120) -> str:
    """
    把文本裁短成适合放进 working memory 的长度。
    """
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def extract_active_paths(tool_name: str, tool_input: Any) -> list[str]:
    """
    从工具输入里提取活跃路径。

    第一版尽量覆盖常见工具输入格式：
    - path
    - file_path
    - directory
    - paths
    - run_command 里的命令文本
    """
    paths: list[str] = []

    # 只有字典输入才好做结构化提取。
    if isinstance(tool_input, dict):
        # 这些字段通常直接表示路径。
        for key in ("path", "file_path", "directory", "dir", "root"):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())

        # 一些工具会传字符串列表。
        for key in ("paths", "files"):
            value = tool_input.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        paths.append(item.strip())

        # run_command 比较特殊，只能从命令文本里粗提一些看起来像路径的片段。
        if tool_name == "run_command":
            command = tool_input.get("command")
            if isinstance(command, str) and command.strip():
                for match in _PATH_TOKEN_RE.findall(command):
                    token = match.strip().strip("\"'")
                    if token:
                        paths.append(token)

    # 去重并保留原顺序。
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)

    return result


def summarize_failure(tool_name: str, result: ToolResult) -> str:
    """
    把一次失败结果压成适合写入 working memory 的短摘要。
    """
    # 优先使用更稳定的 error 字段。
    base = result.error or result.output or "未知失败"

    # 只取第一行，避免把大段 stderr 全塞进短期记忆。
    first_line = str(base).strip().splitlines()[0] if str(base).strip() else "未知失败"

    return f"{tool_name}: {_shorten(first_line, 140)}"


def extract_decision_from_assistant(content: str) -> str | None:
    """
    从 assistant 最终回复里抽一条“最近关键决策”。

    第一版先用启发式规则，不调用模型。
    目标不是完美，而是尽量提取“可复用的执行方向或约束”。
    """
    text = " ".join(content.strip().split())
    if not text:
        return None

    # 太短通常更像普通确认，不一定值得记为关键决策。
    if len(text) < 12:
        return None

    # 按句号、问号、感叹号做简单切句，优先看前几句。
    sentences = re.split(r"(?<=[。！？.!?])\s+", text)

    # 这些关键词更像“已经确认的方向/约束/承诺”。
    decision_keywords = (
        "以后",
        "统一",
        "约定",
        "将会",
        "我会",
        "接下来",
        "默认",
        "使用",
        "采用",
        "改为",
        "from now on",
        "i will",
        "use ",
        "switch to",
        "default to",
    )

    for sentence in sentences[:3]:
        candidate = sentence.strip()
        lowered = candidate.lower()
        if any(keyword in lowered or keyword in candidate for keyword in decision_keywords):
            return _shorten(candidate, 160)

    # 如果没命中关键词，但整段本身就很短且像结论，也可以退而求其次保留首句。
    first_sentence = sentences[0].strip() if sentences else ""
    if 12 <= len(first_sentence) <= 120:
        return _shorten(first_sentence, 160)

    return None
