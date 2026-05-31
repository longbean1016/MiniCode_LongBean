"""会话摘要辅助模块，用于生成轻量级的会话概览文本。"""

from app.types import ChatMessage


def _shorten(text: str,max_chars: int)->str:
    """把文本裁剪到固定长度，避免摘要过长。"""

    text=text.strip()
    if len(text) <=max_chars:
        return text
    return  f"{text[:max_chars]}..."


def build_session_summary(messages: list[ChatMessage])->str:
    """基于会话消息生成规则摘要"""

    user_messages: list[str]=[]
    assistant_messages: list[str]=[]
    tool_names: list[str]=[]

    # 提取用户消息、助手消息和工具调用名称
    for msg in messages:
        role=msg.get("role")

        if role=="user":
            content=str(msg.get("content")).strip()
            if content:
                user_messages.append(content)
        elif role=="assistant":
            content=str(msg.get("content")).strip()
            if content:
                assistant_messages.append(content)
        elif role=="assistant_tool_call":
            tool_name=str(msg.get("tool_name")).strip()
            if tool_name and tool_name not in tool_names:
                tool_names.append(tool_name)

    # 没有消息时给空摘要
    if not user_messages and not assistant_messages and not tool_names:
        return ""
    
    parts: list[str]=[]

     # 第一部分：用户当前主要目标
    if user_messages:
        latest_user_goal = _shorten(user_messages[-1], 80)
        parts.append(f"当前目标：{latest_user_goal}")

    # 第二部分：最近一次明确回答
    if assistant_messages:
        latest_answer = _shorten(assistant_messages[-1], 80)
        parts.append(f"最近结论：{latest_answer}")

    # 第三部分：本会话用到的工具
    if tool_names:
        tools_text = "、".join(tool_names[-5:])
        parts.append(f"已使用工具：{tools_text}")

    return "；".join(parts)
