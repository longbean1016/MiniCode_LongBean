from __future__ import annotations

import time

from app.logger import log_event
from app.message_builder import MessageBuilder
from app.prompt import build_system_prompt
from app.tooling import ToolRegistry
from app.types import AgentStep, ApprovalRequest, ChatMessage, ModelAdapter, ToolContext, ToolResult
from app.working_memory import WorkingMemory


def run_agent_once(
    user_input: str,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    working_memory: WorkingMemory,
    history: list[ChatMessage] | None = None,
    max_steps: int = 8,
    session_id: str = "",
) -> tuple[AgentStep, list[ChatMessage]]:
    """执行一轮 agent 主循环：模型 -> 工具 -> 再模型，直到完成或达到上限。"""
    # 没有历史时用空列表兜底
    if history is None:
        history = []

    # 用 MessageBuilder 统一管理本轮消息
    builder = MessageBuilder()
    builder.extend(history)
    builder.add_user(user_input)

    # 从“新用户输入”开始进入主循环
    return _run_agent_loop(
        builder=builder,
        model=model,
        tool_registry=tool_registry,
        tool_context=tool_context,
        max_steps=max_steps,
        working_memory=working_memory,
        session_id=session_id,
    )


def continue_agent_from_history(
    history: list[ChatMessage],
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    working_memory: WorkingMemory,
    tool_context: ToolContext,
    max_steps: int = 8,
    session_id: str = "",
) -> tuple[AgentStep, list[ChatMessage]]:
    """基于已有历史继续主循环，不再追加新的 user 消息。"""
    builder = MessageBuilder()
    builder.extend(history)
    return _run_agent_loop(
        builder=builder,
        model=model,
        tool_registry=tool_registry,
        tool_context=tool_context,
        max_steps=max_steps,
        working_memory=working_memory,
        session_id=session_id,
    )


def _run_agent_loop(
    builder: MessageBuilder,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    max_steps: int,
    working_memory: WorkingMemory,
    session_id: str,
) -> tuple[AgentStep, list[ChatMessage]]:
    """执行真正的模型/工具循环，既可用于新请求，也可用于授权后的继续执行。"""
    # 记录整轮请求开始时间
    loop_started_at = time.perf_counter()

    # 构建系统提示词
    system_prompt = build_system_prompt(tool_registry)

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
            # 记录最近一次模型失败，后面做 prompt 注入时可以提醒模型避坑
            working_memory.add_failure(f"模型调用失败: {error}")
            fallback = AgentStep(
                type="assistant",
                content=f"模型调用失败: {error}",
                kind="final",
            )
            builder.add_assistant(fallback.content)
            return fallback, builder.build() # type: ignore

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
                # 如果工具参数里带 path，说明这个路径是当前任务的活跃路径
                if isinstance(tool_input, dict):
                    raw_path = tool_input.get("path")
                    if isinstance(raw_path, str) and raw_path.strip():
                        working_memory.add_active_path(raw_path)

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

                tool_cost = time.perf_counter() - tool_started_at
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具 {tool_name} "
                    f"返回 ok={result.ok} error={result.error} 耗时={tool_cost:.3f}s"
                )
                # 工具失败时，把错误记进短期工作记忆
                if not result.ok:
                    failure_text = result.error or result.output
                    working_memory.add_failure(f"{tool_name}: {failure_text}")

                # 命中“需要授权”时，不继续喂模型，而是把审批请求返回给 main
                if result.error=="PERMISSION_REQUIRED":
                    command=str(result.meta.get("command", ""))
                    reason = str(result.meta.get("reason", ""))
                    action_key = str(result.meta.get("action_key", ""))

                    # 授权中断前也要补一条 tool_result，避免历史里只留下 tool_call
                    # 否则下一轮把这段历史再发给模型时，会因为协议断链而报 400
                    builder.add_tool_result(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        content="该操作需要用户授权，当前尚未执行。",
                        is_error=True,
                    )

                    approval_message = (
                        "该操作需要用户授权。\n"
                        f"工具: {tool_name}\n"
                        f"命令: {command}\n"
                        f"原因: {reason}"
                    )

                    approval_step = AgentStep(
                        type="approval",
                        content=approval_message,
                        approval=ApprovalRequest(
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            action_key=action_key,
                            message=approval_message,
                            input_data=tool_input,
                        ),
                    )
                    return approval_step, builder.build()

                # 正常情况才把工具结果写回消息历史
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
