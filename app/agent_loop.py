from __future__ import annotations

import time

from app.compaction_policy import build_compaction_policy
from app.history_window import build_older_history_summary, select_history_window
from app.history_summarizer import OlderHistorySummarizer
from app.logger import log_event
from app.memory_context_builder import build_memory_context
from app.memory_store import MemoryStore
from app.message_builder import MessageBuilder
from app.prompt import build_system_prompt
from app.session import SessionData
from app.tooling import ToolRegistry
from app.types import AgentStep, ApprovalRequest, ChatMessage, ModelAdapter, ToolContext, ToolResult
from app.working_memory import WorkingMemory
from app.working_memory_updater import (
    extract_active_paths,
    extract_decision_from_assistant,
    summarize_failure,
)


def run_agent_once(
    user_input: str,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    session: SessionData,
    working_memory: WorkingMemory,
    memory_store: MemoryStore | None,
    history_summarizer: OlderHistorySummarizer | None = None,
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
        session=session,
        max_steps=max_steps,
        working_memory=working_memory,
        memory_store=memory_store,
        history_summarizer=history_summarizer,
        session_id=session_id,
    )


def continue_agent_from_history(
    history: list[ChatMessage],
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    session: SessionData,
    working_memory: WorkingMemory,
    memory_store: MemoryStore | None,
    history_summarizer: OlderHistorySummarizer | None = None,
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
        session=session,
        max_steps=max_steps,
        working_memory=working_memory,
        memory_store=memory_store,
        history_summarizer=history_summarizer,
        session_id=session_id,
    )


def _run_agent_loop(
    builder: MessageBuilder,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    session: SessionData,
    max_steps: int,
    working_memory: WorkingMemory,
    memory_store: MemoryStore | None,
    history_summarizer: OlderHistorySummarizer | None,
    session_id: str,
) -> tuple[AgentStep, list[ChatMessage]]:
    """执行真正的模型/工具循环，既可用于新请求，也可用于授权后的继续执行。"""
    # 记录整轮请求开始时间
    loop_started_at = time.perf_counter()

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

        # 先取出当前完整历史。
        # 这份 full_history 只在本轮内部用于切窗口和生成摘要，
        # 不会整段原样发给模型。
        full_history = list(builder.build())

        # 按 user 消息把完整历史切成多轮，只保留最近几轮完整消息。
        # 更老的历史转交给摘要层，避免上下文无限增长。
        compaction_policy = build_compaction_policy(session)

        history_window = select_history_window(
            history=full_history,
            keep_rounds=compaction_policy.keep_rounds,
        )

        # 旧历史只保留主线摘要，不再原样透传。
        # 旧历史摘要优先走模型版摘要器。
        # 如果外面没有传摘要器，就自动退回现有的规则摘要。
        if history_summarizer is not None:
            older_history_summary = history_summarizer.summarize(
                session=session,
                older_messages=history_window.older_messages,
                older_round_count=history_window.older_round_count,
            )
        else:
            older_history_summary = build_older_history_summary(
                history_window.older_messages,
            )

        # 记录本轮上下文裁剪结果，方便观察最近窗口策略是否生效。
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮上下文窗口: "
            f"level={compaction_policy.level} keep_rounds={compaction_policy.keep_rounds} older={len(history_window.older_messages)} recent={len(history_window.recent_messages)}"
        )

        # 用最近几轮原始消息构造临时会话快照。
        # 这样 session_snapshot 更接近本轮真正要发给模型的原始消息窗口。
        session_snapshot = SessionData(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            workspace=session.workspace,
            messages=list(history_window.recent_messages),
            extra=dict(session.extra),
        )

        # 把“会话摘要 + 短期工作记忆 + 长期记忆检索结果”拼成一段辅助上下文。
        # 这段内容会被注入 system prompt，而不是直接改写原始 history 结构。
        memory_context = build_memory_context(
            user_input=working_memory.current_goal,
            session=session_snapshot,
            working_memory=working_memory,
            memory_store=memory_store,
            session_summary_override=older_history_summary,
        )

        # 每一轮都按最新记忆上下文重新构造系统提示词，
        # 这样模型能看到刚刚更新过的 summary / working memory / 长期记忆。
        system_prompt = build_system_prompt(
            tool_registry=tool_registry,
            memory_context=memory_context,
        )

        # 每一轮都只发送最近几轮完整消息。
        # 更老的历史已经进入 older_history_summary，不再重复占上下文窗口。
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        messages.extend(history_window.recent_messages)

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

            # 从最终 assistant 回复里尝试抽一条关键决策。
            # 这不是为了记录所有回答，而是尽量保留“已经确认的方向或约束”。
            decision = extract_decision_from_assistant(step.content)
            if decision:
                working_memory.add_decision(decision)

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

                # 从工具输入里尽量提取活跃路径。
                # 这一步会覆盖 path / file_path / directory / run_command 等常见形式。
                for path in extract_active_paths(tool_name, tool_input):
                    working_memory.add_active_path(path)

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

                # 工具失败时，把错误压成短摘要写进短期工作记忆。
                if not result.ok:
                    working_memory.add_failure(
                        summarize_failure(tool_name, result)
                    )

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
