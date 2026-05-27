from __future__ import annotations

from dataclasses import dataclass

from app.session_summary import build_session_summary
from app.types import ChatMessage


@dataclass(slots=True)
class HistoryWindow:
    """
    表示一次历史裁剪后的窗口结果。

    older_messages:
        较早的历史消息，不再原样发给模型，而是交给摘要层压缩。

    recent_messages:
        最近几轮完整消息，保留原始结构继续发给模型。
    """

    older_messages: list[ChatMessage]
    recent_messages: list[ChatMessage]


def split_history_rounds(history: list[ChatMessage]) -> list[list[ChatMessage]]:
    """
    按 user 消息把完整历史切成多轮。

    规则：
    1. 每遇到一条 user 消息，就视为新一轮开始。
    2. 在下一条 user 出现之前的所有消息，都归属当前这一轮。
    3. 如果历史开头在第一条 user 之前还有零散消息，就挂到第一轮前面；
       如果整段历史里都没有 user，则把整段历史视为一轮。
    """
    rounds: list[list[ChatMessage]] = []
    current_round: list[ChatMessage] = []
    leading_messages: list[ChatMessage] = []

    for message in history:
        role = message.get("role")

        # user 是新一轮的边界。
        if role == "user":
            # 已经有上一轮内容时，先把上一轮收进去。
            if current_round:
                rounds.append(current_round)

            # 新一轮开始时，把第一条 user 之前积累的零散消息也一并挂进去，
            # 避免出现消息脱节。
            current_round = list(leading_messages)
            current_round.append(message)
            leading_messages = []
            continue

        # 还没遇到第一条 user 时，先把消息暂存到前导区。
        if not current_round:
            leading_messages.append(message)
            continue

        # 已经进入某一轮后，其余消息都归入当前轮。
        current_round.append(message)

    # 循环结束后，把最后一轮收进去。
    if current_round:
        rounds.append(current_round)

    # 如果整个历史都没有 user，就把全部消息当成单独一轮处理。
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
    更早的历史消息不再原样透传，而是进入 older_messages 供摘要使用。
    """
    # 非法值兜底为 1，避免 keep_rounds=0 时把全部历史都裁掉。
    normalized_keep_rounds = max(1, keep_rounds)

    rounds = split_history_rounds(history)
    if not rounds:
        return HistoryWindow(older_messages=[], recent_messages=[])

    # 历史轮数不多时，全部保留为 recent，不需要 older。
    if len(rounds) <= normalized_keep_rounds:
        recent_messages = [msg for round_messages in rounds for msg in round_messages]
        return HistoryWindow(older_messages=[], recent_messages=recent_messages)

    older_rounds = rounds[:-normalized_keep_rounds]
    recent_rounds = rounds[-normalized_keep_rounds:]

    older_messages = [msg for round_messages in older_rounds for msg in round_messages]
    recent_messages = [msg for round_messages in recent_rounds for msg in round_messages]

    return HistoryWindow(
        older_messages=older_messages,
        recent_messages=recent_messages,
    )


def build_older_history_summary(older_messages: list[ChatMessage]) -> str:
    """
    仅基于被裁掉的旧历史生成摘要。

    这样可以避免 session summary 和 recent tail 大量重复，
    让摘要更偏向“窗口外的历史主线”。
    """
    if not older_messages:
        return ""

    return build_session_summary(older_messages)
