from __future__ import annotations

from dataclasses import dataclass

"""历史窗口模型，区分近期消息、较早消息和压缩边界。"""

from app.session_summary import build_session_summary
from app.types import ChatMessage


@dataclass(slots=True)
class HistoryWindow:
    """
    表示历史裁剪后的窗口结果。

    older_messages:
        比较早的历史消息，不再原样发给模型，而是交给摘要层处理。
    recent_messages:
        最近几轮完整消息，保留原始结构继续发给模型。
    older_round_count:
        被裁到窗口外的旧轮次数量。
    recent_round_count:
        当前直接保留的最近轮次数量。
    """

    older_messages: list[ChatMessage]
    recent_messages: list[ChatMessage]
    older_round_count: int = 0
    recent_round_count: int = 0


def split_history_rounds(history: list[ChatMessage]) -> list[list[ChatMessage]]:
    """
    按 user 消息把完整历史切成多轮。

    规则：
    1. 每遇到一条 user 消息，就视为新一轮开始。
    2. 在下一条 user 出现之前的所有消息，都归属当前这一轮。
    3. 如果历史开头在第一条 user 之前还有零散消息，就挂到第一轮前面。
    """
    rounds: list[list[ChatMessage]] = []
    current_round: list[ChatMessage] = []
    leading_messages: list[ChatMessage] = []

    for message in history:
        role = message.get("role")

        if role == "user":
            if current_round:
                rounds.append(current_round)

            current_round = list(leading_messages)
            current_round.append(message)
            leading_messages = []
            continue

        if not current_round:
            leading_messages.append(message)
            continue

        current_round.append(message)

    if current_round:
        rounds.append(current_round)

    if not rounds and leading_messages:
        rounds.append(list(leading_messages))

    return rounds


def select_history_window(
    history: list[ChatMessage],
    keep_rounds: int = 6,
) -> HistoryWindow:
    """
    从完整历史里切出“旧历史”和“最近几轮”。

    keep_rounds 表示保留最近多少轮完整消息。
    更早的消息不再原样透传，而是进入 older_messages 供摘要使用。
    """
    normalized_keep_rounds = max(1, keep_rounds)

    rounds = split_history_rounds(history)
    if not rounds:
        return HistoryWindow(
            older_messages=[],
            recent_messages=[],
            older_round_count=0,
            recent_round_count=0,
        )

    if len(rounds) <= normalized_keep_rounds:
        recent_messages = [msg for round_messages in rounds for msg in round_messages]
        return HistoryWindow(
            older_messages=[],
            recent_messages=recent_messages,
            older_round_count=0,
            recent_round_count=len(rounds),
        )

    older_rounds = rounds[:-normalized_keep_rounds]
    recent_rounds = rounds[-normalized_keep_rounds:]

    older_messages = [msg for round_messages in older_rounds for msg in round_messages]
    recent_messages = [msg for round_messages in recent_rounds for msg in round_messages]

    return HistoryWindow(
        older_messages=older_messages,
        recent_messages=recent_messages,
        older_round_count=len(older_rounds),
        recent_round_count=len(recent_rounds),
    )


def build_older_history_summary(older_messages: list[ChatMessage]) -> str:
    """
    仅基于被裁掉的旧历史生成规则摘要。

    这是模型摘要不可用时的稳定兜底。
    """
    if not older_messages:
        return ""

    return build_session_summary(older_messages)
