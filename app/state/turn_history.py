from __future__ import annotations

from app.context.history_window import split_history_rounds
from app.types import ChatMessage


def get_last_round_messages(history: list[ChatMessage]) -> list[ChatMessage]:
    """
    从完整 history 中取出最后一轮消息。

    这里的“一轮”定义为：
    - 从一条 user 消息开始
    - 后面跟随 assistant_tool_call / tool_result / assistant 等消息
    - 直到下一条 user 出现为止
    """
    # 先按 user 边界把历史切成多轮。
    rounds = split_history_rounds(history)

    # 没有任何历史时返回空列表。
    if not rounds:
        return []

    # 最后一轮就是当前最新一次用户问题对应的完整消息链。
    return list(rounds[-1])
