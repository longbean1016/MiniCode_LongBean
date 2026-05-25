from __future__ import annotations

import time

from app.logger import log_event
from app.message_builder import MessageBuilder
from app.prompt import build_system_prompt
from app.tooling import ToolRegistry
from app.types import AgentStep, ChatMessage, ModelAdapter, ToolContext, ToolResult


def run_agent_once(
    user_input: str,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    history: list[ChatMessage] | None = None,
    max_steps: int = 8,
    session_id: str = "",
) -> tuple[AgentStep, list[ChatMessage]]:
    """执行一轮 agent 主循环：模型 -> 工具 -> 再模型，直到完成或达到上限。"""
    # 记录整轮请求开始时间
    loop_started_at = time.perf_counter()

    # 没有历史时用空列表兜底
    if history is None:
        history = []

    # 构建系统提示词
    system_prompt = build_system_prompt(tool_registry)

    # 用 MessageBuilder 统一管理本轮消息
    builder = MessageBuilder()
    builder.extend(history)
    builder.add_user(user_input)

    # 记录本轮请求开始
    log_event(
        f"[session={session_id or '-'}] 开始一轮 Agent 请求"
    )

    # 限制循环步数，防止模型和工具来回打转
    for step_index in range(max_steps):
        # 记录当前 step 开始时间
        step_started_at = time.perf_counter()

        # 记录当前是第几轮循环
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮循环开始"
        )

        # 每一轮都把系统提示词和最新历史发给模型
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        messages.extend(builder.build())

        try:
            # 记录即将请求模型
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮开始请求模型"
            )

            # 请求模型给出下一步动作
            step = model.next(messages=messages)

            # 记录模型返回类型
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮模型返回类型: {step.type}"
            )
        except Exception as error:
            # 模型调用异常时兜底为最终回答，避免主循环直接崩掉
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮模型调用异常: {error}"
            )
            fallback = AgentStep(
                type="assistant",
                content=f"模型调用失败: {error}",
                kind="final",
            )
            builder.add_assistant(fallback.content)
            return fallback, builder.build()

        # 情况一：模型直接返回最终答案
        if step.type == "assistant":
            builder.add_assistant(step.content)

            # 记录当前 step 和整轮总耗时
            step_cost = time.perf_counter() - step_started_at
            total_cost = time.perf_counter() - loop_started_at
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮直接返回答案 "
                f"step耗时={step_cost:.3f}s 总耗时={total_cost:.3f}s"
            )
            return step, builder.build()

        # 情况二：模型要求调用一个或多个工具
        if step.type == "tool_calls":
            # 特殊情况：模型返回了空工具调用
            if not step.calls:
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具调用为空"
                )
                fallback = AgentStep(
                    type="assistant",
                    content="模型返回了空的工具调用。",
                    kind="final",
                )
                builder.add_assistant(fallback.content)
                return fallback, builder.build()

            # 依次记录、执行并回写每个工具调用结果
            for call in step.calls:
                tool_name = call["tool_name"]
                tool_input = call["input"]
                tool_use_id = call["id"]

                # 先把工具调用请求记到历史里
                builder.add_tool_call(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    input_data=tool_input,
                )

                # 记录即将调用哪个工具
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮准备调用工具: {tool_name}"
                )

                # 记录单个工具开始时间
                tool_started_at = time.perf_counter()

                try:
                    # 统一通过 registry 执行工具
                    result = tool_registry.execute_tool(
                        tool_name=tool_name,
                        input_data=tool_input,
                        context=tool_context,
                    )

                    # 记录工具执行结果和耗时
                    tool_cost = time.perf_counter() - tool_started_at
                    log_event(
                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具 {tool_name} "
                        f"执行完成 ok={result.ok} 耗时={tool_cost:.3f}s"
                    )
                except Exception as error:
                    # 理论上 registry 已经兜底，这里是主循环最后一层保险
                    tool_cost = time.perf_counter() - tool_started_at
                    log_event(
                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具 {tool_name} "
                        f"执行异常: {error} 耗时={tool_cost:.3f}s"
                    )
                    result = ToolResult(
                        ok=False,
                        output=f"工具调用发生未捕获异常: {error}",
                        error="UNCAUGHT_TOOL_ERROR",
                        meta={"tool_name": tool_name},
                    )

                # 记录工具返回状态
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具 {tool_name} 返回 ok={result.ok}"
                )

                # 始终把工具结果喂回模型
                builder.add_tool_result(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    content=result.output,
                    is_error=not result.ok,
                )

            # 记录当前工具阶段结束
            step_cost = time.perf_counter() - step_started_at
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具阶段结束 step耗时={step_cost:.3f}s"
            )
            continue

        # 情况三：遇到未知返回类型时兜底退出
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮出现未识别返回类型: {step.type}"
        )
        fallback = AgentStep(
            type="assistant",
            content="未识别的模型返回类型。",
            kind="final",
        )
        builder.add_assistant(fallback.content)
        return fallback, builder.build()

    # 达到最大步数时停止，防止死循环
    total_cost = time.perf_counter() - loop_started_at
    log_event(
        f"[session={session_id or '-'}] 达到最大循环步数 {max_steps} 总耗时={total_cost:.3f}s"
    )
    fallback = AgentStep(
        type="assistant",
        content="已达到最大循环步数，本轮已停止。",
        kind="final",
    )
    builder.add_assistant(fallback.content)
    return fallback, builder.build()
