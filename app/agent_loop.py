from app.message_builder import MessageBuilder
from app.prompt import build_system_prompt
from app.tooling import ToolRegistry
from app.types import AgentStep, ChatMessage, ModelAdapter, ToolContext


def run_agent_once(
    user_input: str,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    history: list[ChatMessage] | None = None,
    max_steps: int = 8,
) -> tuple[AgentStep, list[ChatMessage]]:
    """执行一轮 agent 主循环：模型 -> 工具 -> 再模型，直到完成或达到上限。"""
    if history is None:
        history = []

    system_prompt = build_system_prompt(tool_registry)

    # 用 MessageBuilder 统一管理本轮消息
    builder = MessageBuilder()
    builder.extend(history)
    builder.add_user(user_input)

    # 限制循环次数，防止模型和工具一直来回
    for step_index in range(max_steps):
        print(f"这是第 {step_index + 1} 轮循环")

        # 每一轮都把最新历史发给模型
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        messages.extend(builder.build())

        step = model.next(messages=messages)

        # 情况一：模型直接返回最终回答
        if step.type == "assistant":
            builder.add_assistant(step.content)
            return step, builder.build()

        # 情况二：模型要求调用一个或多个工具
        if step.type == "tool_calls":
            # 特殊情况：模型返回空工具调用
            if not step.calls:
                fallback = AgentStep(
                    type="assistant",
                    content="模型返回了空的工具调用。",
                    kind="final",
                )
                builder.add_assistant(fallback.content)
                return fallback, builder.build()

            # 先记录工具调用，再执行工具，再记录工具结果
            for call in step.calls:
                print(f"准备调用工具: {call['tool_name']}")
                builder.add_tool_call(
                    tool_use_id=call["id"],
                    tool_name=call["tool_name"],
                    input_data=call["input"],
                )

                result = tool_registry.execute_tool(
                    tool_name=call["tool_name"],
                    input_data=call["input"],
                    context=tool_context,
                )
                print(f"工具{call['tool_name']}返回: {result.ok}")

                builder.add_tool_result(
                    tool_use_id=call["id"],
                    tool_name=call["tool_name"],
                    content=result.output,
                    is_error=not result.ok,
                )

            # 工具执行后继续下一轮，把结果喂回模型
            continue

        # 情况三：未知返回类型，直接兜底退出
        fallback = AgentStep(
            type="assistant",
            content="未识别的模型返回类型。",
            kind="final",
        )
        builder.add_assistant(fallback.content)
        return fallback, builder.build()

    # 达到最大步数，防止死循环
    fallback = AgentStep(
        type="assistant",
        content="已达到最大循环步数，本轮已停止。",
        kind="final",
    )
    builder.add_assistant(fallback.content)
    return fallback, builder.build()
