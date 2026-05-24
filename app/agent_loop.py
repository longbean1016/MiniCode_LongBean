from app.prompt import build_system_prompt
from app.tooling import ToolRegistry
from app.types import AgentStep, ChatMessage, ModelAdapter, ToolContext


def run_agent_once(
    user_input: str,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    history: list[ChatMessage] | None = None,
) -> tuple[AgentStep, list[ChatMessage]]:  # type: ignore
    """
    执行一次最小 agent 主流程。

    当前版本支持：
    1. 普通文本回答
    2. 单轮工具调用后再生成最终回答
    """

    # 没有历史消息时，先初始化为空列表
    if history is None:
        history = []

    # 根据当前工具注册表生成系统提示词
    system_prompt = build_system_prompt(tool_registry)

    # 先构造本轮要发给模型的完整消息
    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # 第一次调用模型，判断是直接回答还是先调用工具
    step = model.next(messages)

    # 如果模型直接返回普通回答，就结束本轮
    if step.type == "assistant":
        new_history = history + [
            {
                "role": "user",
                "content": user_input,
            },
            {
                "role": "assistant",
                "content": step.content,
            },
        ]
        return step, new_history

    # 如果模型要求调用工具，就先执行工具
    if step.type == "tool_calls":
        working_history = history + [
            {
                "role": "user",
                "content": user_input,
            }
        ]

        for call in step.calls:
            # 记录模型发起的工具调用
            working_history.append(
                {
                    "role": "assistant_tool_call",
                    "tool_use_id": call["id"],
                    "tool_name": call["tool_name"],
                    "input": call["input"],
                }
            )

            # 真正执行工具
            result = tool_registry.execute_tool(
                tool_name=call["tool_name"],
                input_data=call["input"],
                context=tool_context,
            )

            # 记录工具执行结果
            working_history.append(
                {
                    "role": "tool_result",
                    "tool_use_id": call["id"],
                    "tool_name": call["tool_name"],
                    "content": result.output,
                    "is_error": not result.ok,
                }
            )

        # 把工具结果再发给模型，让模型生成最终回答
        follow_up_messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        follow_up_messages.extend(working_history)

        final_step = model.next(follow_up_messages)

        if final_step.type == "assistant":
            new_history = working_history + [
                {
                    "role": "assistant",
                    "content": final_step.content,
                }
            ]
            return final_step, new_history

        # 第一版先不处理多轮连续工具调用，这里做一个兜底返回
        fallback_step = AgentStep(
            type="assistant",
            content="本轮工具调用已完成，但暂未生成最终回答。",
            kind="final",
        )
        return fallback_step, working_history

    # 理论上不会走到这里，先加一个保险兜底
    fallback_step = AgentStep(
        type="assistant",
        content="未识别的模型返回结果。",
        kind="final",
    )
    return fallback_step, history