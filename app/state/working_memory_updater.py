from __future__ import annotations

import re
from typing import Any

from app.types import ToolResult


# 用来从 run_command 命令文本里提取看起来像路径的片段。
_PATH_TOKEN_RE = re.compile(
    r'([A-Za-z]:\\[^\s"\']+|\.{0,2}[\\/][^\s"\']+|[A-Za-z0-9_.-]+[\\/][A-Za-z0-9_./\\-]+)'
)
_SENTENCE_SPLIT_RE = re.compile(r"(?:\r?\n)+|(?<=[。！？!?；;，,])\s*")

_PREFERENCE_KEYWORDS = (
    "默认",
    "优先",
    "偏好",
    "习惯",
    "中文",
    "英文",
    "注释",
    "风格",
    "简洁",
    "详细",
    "按照",
    "参考",
    "命名",
    "格式",
)
_CONSTRAINT_KEYWORDS = (
    "不要",
    "不能",
    "不可以",
    "只",
    "必须",
    "需要",
    "尽量不要",
    "不要把",
    "不要改",
    "不要动",
    "保留",
    "限制",
)
_RISK_KEYWORDS = (
    "风险",
    "失败",
    "报错",
    "错误",
    "超长",
    "超限",
    "overflow",
    "exceeds limit",
    "too long",
    "超时",
    "丢",
    "漏",
    "堆积",
    "污染",
    "冲刷",
    "恢复",
    "重试",
    "压缩",
)


def _shorten(text: str, max_chars: int = 120) -> str:
    """
    把文本裁短成适合放进 working memory 的长度。
    """
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _split_sentences(text: str) -> list[str]:
    """按中文/英文标点和换行做轻量切句。"""
    normalized = str(text).strip()
    if not normalized:
        return []

    parts = _SENTENCE_SPLIT_RE.split(normalized)
    result: list[str] = []
    for part in parts:
        cleaned = " ".join(part.strip().split())
        if cleaned:
            result.append(cleaned)
    return result


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in text or keyword in lowered for keyword in keywords)


def _dedupe_texts(items: list[str]) -> list[str]:
    """按归一化文本去重，避免同一句被多次写入 working memory。"""
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = re.sub(r"\s+", " ", item.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    return result


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


def extract_user_preferences(text: str) -> list[str]:
    """
    从用户输入或 assistant 承诺里提取“回答/实现偏好”。

    这里参考 minicode 的 user profile 思路：
    - 优先抓稳定偏好，不抓一次性任务目标
    - 偏好应尽量短、可复用、可跨轮次继承
    """
    preferences: list[str] = []
    for sentence in _split_sentences(text):
        if not _contains_any(sentence, _PREFERENCE_KEYWORDS):
            continue
        if _contains_any(sentence, _RISK_KEYWORDS):
            continue
        if _contains_any(sentence, ("main.py", "agent_loop.py", ".py", "文件")) and _contains_any(sentence, _CONSTRAINT_KEYWORDS):
            continue
        preferences.append(_shorten(sentence, 140))
    return _dedupe_texts(preferences)[:3]


def extract_project_constraints(text: str) -> list[str]:
    """
    从用户输入或 assistant 结论里提取“实现约束”。

    重点抓：
    - 不要改哪些文件
    - 必须保留哪些结构
    - 只能如何计算/实现
    """
    constraints: list[str] = []
    for sentence in _split_sentences(text):
        if not _contains_any(sentence, _CONSTRAINT_KEYWORDS):
            continue
        if _contains_any(sentence, _RISK_KEYWORDS) and not _contains_any(sentence, ("必须", "需要", "不要", "不能")):
            continue
        constraints.append(_shorten(sentence, 160))
    return _dedupe_texts(constraints)[:4]


def extract_recent_risks(text: str) -> list[str]:
    """
    从 assistant 分析或失败信息里提取“近期风险/不稳定点”。

    风险强调最近需要警惕的问题，而不是一般性目标描述。
    """
    risks: list[str] = []
    for sentence in _split_sentences(text):
        if not _contains_any(sentence, _RISK_KEYWORDS):
            continue
        risks.append(_shorten(sentence, 160))
    return _dedupe_texts(risks)[:4]


def summarize_failure(tool_name: str, result: ToolResult) -> str:
    """
    把一次失败结果压成适合写入 working memory 的短摘要。
    """
    # 优先使用更稳定的 error 字段。
    base = result.error or result.output or "未知失败"

    # 只取第一行，避免把大段 stderr 全塞进短期记忆。
    first_line = str(base).strip().splitlines()[0] if str(base).strip() else "未知失败"

    return f"{tool_name}: {_shorten(first_line, 140)}"


def extract_decisions_from_assistant(content: str) -> list[str]:
    """
    从 assistant 最终回复里抽取多条“最近关键决策/结论”。

    优先保留 bullet 和短结论句，避免把整段说明作为单条决策写入。
    """
    text = str(content).strip()
    if not text:
        return []

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
        "负责",
        "顺序",
        "优先",
        "治理",
        "拆成",
        "保留",
        "from now on",
        "i will",
        "use ",
        "switch to",
        "default to",
        "working memory",
        "context",
        "compact",
        "pipeline",
        "tool_result",
    )

    candidates: list[str] = []
    for raw_line in text.splitlines():
        normalized = re.sub(r"^\s*(?:[-*•]+|\d+\.)\s*", "", raw_line).strip()
        normalized = " ".join(normalized.split())
        if not normalized or len(normalized) < 12:
            continue
        if normalized.endswith(("：", ":")):
            continue
        candidates.append(normalized)

    if not candidates:
        candidates = _split_sentences(text)

    accepted: list[str] = []
    for candidate in candidates:
        lowered = candidate.lower()
        if not any(keyword in candidate or keyword in lowered for keyword in decision_keywords):
            continue
        accepted.append(_shorten(candidate, 160))

    if not accepted:
        first_sentence = _split_sentences(text)
        if first_sentence:
            head = first_sentence[0]
            if 12 <= len(head) <= 120:
                accepted.append(_shorten(head, 160))

    return _dedupe_texts(accepted)[:4]


def extract_decision_from_assistant(content: str) -> str | None:
    """
    从 assistant 最终回复里抽一条“最近关键决策”。

    第一版先用启发式规则，不调用模型。
    目标不是完美，而是尽量提取“可复用的执行方向或约束”。
    """
    decisions = extract_decisions_from_assistant(content)
    return decisions[0] if decisions else None
