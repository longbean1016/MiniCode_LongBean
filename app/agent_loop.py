from __future__ import annotations

"""Agent 主循环，负责模型调用、工具执行、审批中断与结果回写。"""

import time

from app.analysis_guard import (
    NUDGE_ANALYSIS_CONVERGE,
    NUDGE_ANALYSIS_STRUCTURE_FIRST,
    NUDGE_ANALYSIS_TOOL_PRIORITY,
    _build_analysis_convergence_nudge,
    _build_analysis_fact_correction_nudge,
    _build_analysis_force_answer_nudge,
    _build_analysis_target_resolution_nudge,
    _create_analysis_tracker,
    _find_unobserved_answer_function_names,
    _find_unsupported_analysis_claims,
    _has_sufficient_analysis_evidence,
    _is_code_analysis_request,
    _normalize_analysis_answer_content,
    _record_analysis_evidence,
    _should_block_redundant_analysis_calls,
    _should_redirect_analysis_to_structure_first,
)
from app.context_reactive_compact import (
    is_context_overflow_error,
    recover_from_context_overflow,
)
from app.context_runtime import prepare_agent_context, persist_post_response_working_memory_state
from app.history_summarizer import OlderHistorySummarizer
from app.logger import log_event
from app.memory_pipeline import MemoryPipeline
from app.message_builder import MessageBuilder
from app.session import SessionData
from app.tooling import ToolRegistry
from app.types import AgentStep, ApprovalRequest, ChatMessage, ModelAdapter, ToolContext, ToolResult
from app.working_memory import WorkingMemory
from app.working_memory_updater import (
    extract_active_paths,
    extract_decisions_from_assistant,
    summarize_failure,
)

NUDGE_CONTINUE = (
    "继续推进：如果现有信息已经足够，请直接给出最终答案；"
    "否则只执行下一步最必要的动作，不要重复读取相同文件、符号或目录。"
)
NUDGE_AFTER_TOOL_RESULT = (
    "你已经拿到了工具结果。请先判断现有证据是否已经足够："
    "足够就直接给最终答案；不够才继续一次最必要的工具调用。不要重复刚看到的内容。"
)
EXPLORATION_TOOL_NAMES = {
    "read_file",
    "file_overview",
    "find_symbols",
    "find_references",
    "locate_symbol",
    "get_ast_info",
    "list_files",
    "grep_files",
}


def _append_transient_user_nudge(
    messages: list[ChatMessage],
    content: str | None,
) -> list[ChatMessage]:
    """只在当前模型请求里追加临时引导语，不写入会话历史。"""
    if not content:
        return messages

    if messages:
        last_message = messages[-1]
        if last_message.get("role") == "user" and str(last_message.get("content", "")) == content:
            return messages

    result = list(messages)
    result.append(
        {
            "role": "user",
            "content": content,
        }
    )
    return result


def _extract_tool_target(tool_input: object) -> str:
    """尽量把工具输入归一成“当前在看什么”。"""
    if not isinstance(tool_input, dict):
        return ""

    for key in ("path", "file_path", "symbol_path", "directory", "root", "cwd"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("symbol", "name", "keyword", "pattern", "query"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _is_exploration_tool(tool_name: str) -> bool:
    return tool_name in EXPLORATION_TOOL_NAMES


def _extract_latest_real_user_message(messages: list[ChatMessage]) -> str:
    """拿到当前任务的真实用户请求，跳过系统注入的临时提示。"""
    synthetic_messages = {
        NUDGE_CONTINUE,
        NUDGE_AFTER_TOOL_RESULT,
        NUDGE_ANALYSIS_CONVERGE,
        NUDGE_ANALYSIS_TOOL_PRIORITY,
        NUDGE_ANALYSIS_STRUCTURE_FIRST,
    }
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if not content or content in synthetic_messages:
            continue
        return content
    return ""


def _should_inject_convergence_nudge(
    *,
    exploration_history: list[tuple[str, str]],
    step_index: int,
    max_steps: int,
) -> bool:
    """在重复探索或接近上限时，提醒模型基于现有工具结果继续收敛。"""
    if not exploration_history:
        return False

    remaining_steps = max_steps - step_index - 1
    recent_explorations = exploration_history[-3:]
    recent_targets = [target for _, target in recent_explorations if target]

    repeated_target = (
        len(recent_targets) >= 2
        and len(set(recent_targets[-2:])) == 1
    )
    near_limit_after_multiple_explorations = remaining_steps <= 1 and len(recent_explorations) >= 2

    return repeated_target or near_limit_after_multiple_explorations


def run_agent_once(
    user_input: str,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    session: SessionData,
    working_memory: WorkingMemory,
    memory_pipeline: MemoryPipeline | None,
    history_summarizer: OlderHistorySummarizer | None = None,
    history: list[ChatMessage] | None = None,
    max_steps: int = 20,
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
        memory_pipeline=memory_pipeline,
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
    memory_pipeline: MemoryPipeline | None,
    history_summarizer: OlderHistorySummarizer | None = None,
    max_steps: int = 20,
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
        memory_pipeline=memory_pipeline,
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
    memory_pipeline: MemoryPipeline | None,
    history_summarizer: OlderHistorySummarizer | None,
    session_id: str,
) -> tuple[AgentStep, list[ChatMessage]]:
    """执行真正的模型/工具循环，既可用于新请求，也可用于授权后的继续执行。"""
    # 记录整轮请求开始时间
    loop_started_at = time.perf_counter()
    exploration_history: list[tuple[str, str]] = []
    pending_user_nudge: str | None = None
    blocked_analysis_tool_call_count = 0
    task_text = _extract_latest_real_user_message(builder.build())
    analysis_tracker: dict[str, object] | None = None
    if _is_code_analysis_request(task_text):
        # 分析类任务会额外维护一份“证据账本”。
        # 后面每次读文件/查符号，都会往里面沉淀可验证事实，
        # 最终用它判断是否该停止探索、是否允许直接放行答案。
        analysis_tracker = _create_analysis_tracker(task_text)
        target_resolution_nudge = _build_analysis_target_resolution_nudge(analysis_tracker)
        pending_user_nudge = NUDGE_ANALYSIS_TOOL_PRIORITY
        if target_resolution_nudge:
            pending_user_nudge = pending_user_nudge + "\n" + target_resolution_nudge

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

        # 把上下文准备工作委托给专门模块，避免主循环里塞入过多策略细节。
        prepared_context = prepare_agent_context(
            full_history=full_history,
            session=session,
            tool_registry=tool_registry,
            working_memory=working_memory,
            memory_pipeline=memory_pipeline,
            history_summarizer=history_summarizer,
        )

        # 记录本轮上下文裁剪结果和 token 占用情况，便于观察策略是否生效。
        microcompact_reason = str(
            prepared_context.compaction_history_entry.get("microcompact_reason", "")
        ).strip()
        microcompact_tool_results = prepared_context.compaction_history_entry.get(
            "microcompact_tool_results", 0
        )
        microcompact_keep_recent = prepared_context.compaction_history_entry.get(
            "microcompact_keep_recent", 0
        )
        microcompact_cooldown_remaining = prepared_context.compaction_history_entry.get(
            "microcompact_cooldown_remaining_seconds", 0.0
        )
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮上下文窗口: "
            f"level={prepared_context.policy.level} keep_rounds={prepared_context.policy.keep_rounds} "
            f"older={len(prepared_context.history_window.older_messages)} recent={len(prepared_context.history_window.recent_messages)} "
            f"tool_truncated={prepared_context.compaction_result.truncated_tool_results} "
            f"tool_cleared={prepared_context.compaction_result.cleared_old_tool_results} "
            f"microcompact_reason={microcompact_reason or 'unknown'} "
            f"microcompact_tool_results={microcompact_tool_results} "
            f"microcompact_keep_recent={microcompact_keep_recent} "
            f"microcompact_cooldown_left={float(microcompact_cooldown_remaining):.0f}s "
            f"steps={','.join(prepared_context.pipeline_steps)}"
        )
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮压缩前预估: "
            f"preview_total={prepared_context.preview_stats.total_tokens} "
            f"preview_usage={prepared_context.preview_stats.usage_ratio:.1%} "
            f"preview_budget={prepared_context.preview_stats.usable_budget} "
            f"preview_recent={prepared_context.preview_stats.recent_tokens} "
            f"preview_memory={prepared_context.preview_stats.memory_tokens} "
            f"preview_tool_results={prepared_context.preview_stats.tool_result_tokens}"
        )
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮 token统计: "
            f"total={prepared_context.stats.total_tokens} usage={prepared_context.stats.usage_ratio:.1%} "
            f"budget={prepared_context.stats.usable_budget} system={prepared_context.stats.system_tokens} "
            f"recent={prepared_context.stats.recent_tokens} memory={prepared_context.stats.memory_tokens} "
            f"tool_results={prepared_context.stats.tool_result_tokens}"
        )

        messages = _append_transient_user_nudge(
            prepared_context.messages,
            pending_user_nudge,
        )
        # nudge 只对“这一次模型请求”生效，不会写回 builder 历史。
        # 这样既能引导下一步动作，又不会污染后续真正的会话记录。
        if pending_user_nudge:
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮注入临时引导提示"
            )
            pending_user_nudge = None

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
            if is_context_overflow_error(error):
                # 只有明确判断为上下文溢出时，才会尝试改写消息后重试模型。
                recovery_result = recover_from_context_overflow(
                    messages=messages,
                    usable_budget=prepared_context.stats.usable_budget,
                )
                if recovery_result.recovered:
                    log_event(
                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮触发 Reactive Compact Recover: "
                        f"strategy={recovery_result.strategy} "
                        f"before={recovery_result.tokens_before} after={recovery_result.tokens_after}"
                    )
                    try:
                        step = model.next(messages=recovery_result.messages)
                        log_event(
                            f"[session={session_id or '-'}] 第 {step_index + 1} 轮恢复后模型返回类型: {step.type}"
                        )
                    except Exception as retry_error:
                        error = retry_error
                    else:
                        if step.type == "assistant":
                            if step.kind == "progress":
                                builder.add_progress(step.content)
                                pending_user_nudge = NUDGE_CONTINUE
                                log_event(
                                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮恢复后收到 progress，继续下一轮"
                                )
                                continue

                            # 对代码分析类回答做最后一道事实校验。
                            # 这里拦的不是“回答不好”，而是“引用了未观察到的函数名/统计数字/step.type”。
                            if (
                                analysis_tracker is not None
                                and _has_sufficient_analysis_evidence(analysis_tracker)
                            ):
                                invalid_names = _find_unobserved_answer_function_names(
                                    analysis_tracker,
                                    step.content,
                                )
                                invalid_claims = _find_unsupported_analysis_claims(
                                    analysis_tracker,
                                    step.content,
                                )
                                if (invalid_names or invalid_claims) and step_index < max_steps - 1:
                                    pending_user_nudge = _build_analysis_fact_correction_nudge(
                                        analysis_tracker,
                                        invalid_names,
                                        invalid_claims,
                                    )
                                    log_event(
                                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮恢复后检测到缺少证据支撑的分析结论，要求模型自纠: "
                                        f"{', '.join((invalid_names + invalid_claims)[:6])}"
                                    )
                                    continue

                            final_content = step.content
                            if analysis_tracker is not None:
                                final_content = _normalize_analysis_answer_content(
                                    analysis_tracker,
                                    step.content,
                                )
                            step.content = final_content
                            builder.add_assistant(final_content)

                            if memory_pipeline is not None:
                                memory_pipeline.record_assistant_reply(
                                    working_memory,
                                    content=final_content,
                                )
                            else:
                                decisions = extract_decisions_from_assistant(final_content)
                                for decision in decisions:
                                    working_memory.protect(
                                        decision,
                                        entry_type="key_decision",
                                        ttl_seconds=3600,
                                        importance=0.95,
                                    )
                                    working_memory.protect(
                                        decision,
                                        entry_type="reflection_decision",
                                        ttl_seconds=3600,
                                        importance=0.95,
                                    )

                            persist_post_response_working_memory_state(
                                session=session,
                                working_memory=working_memory,
                            )
                            step_cost = time.perf_counter() - step_started_at
                            total_cost = time.perf_counter() - loop_started_at
                            log_event(
                                f"[session={session_id or '-'}] 第 {step_index + 1} 轮恢复后直接返回答案 "
                                f"step耗时={step_cost:.3f}s 总耗时={total_cost:.3f}s"
                            )
                            return step, builder.build()

            # 模型调用异常时兜底为最终回答，避免主循环直接崩掉
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮模型调用异常: {error}"
            )
            # 记录最近一次模型失败，后面做 prompt 注入时可以提醒模型避坑
            working_memory.protect(
                f"模型调用失败: {error}",
                entry_type="error_context",
                ttl_seconds=1800,
                importance=0.9,
            )
            working_memory.protect(
                f"模型调用失败: {error}",
                entry_type="reflection_failure",
                ttl_seconds=1800,
                importance=0.9,
            )
            fallback = AgentStep(
                type="assistant",
                content=f"模型调用失败: {error}",
                kind="final",
            )
            builder.add_assistant(fallback.content)
            return fallback, builder.build() # type: ignore

        # 情况一：模型直接返回最终答案
        if step.type == "assistant":
            if step.kind == "progress":
                builder.add_progress(step.content)
                # progress 代表模型认为还没收敛。
                # 这里不结束本轮，而是把进度消息写入历史后继续驱动下一轮决策。
                if analysis_tracker is not None and _has_sufficient_analysis_evidence(analysis_tracker):
                    pending_user_nudge = _build_analysis_convergence_nudge(analysis_tracker)
                else:
                    pending_user_nudge = NUDGE_CONTINUE
                step_cost = time.perf_counter() - step_started_at
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮收到 progress，继续下一轮 "
                    f"step耗时={step_cost:.3f}s"
                )
                continue

            # 对代码分析类回答做最后一道符号校验，避免把猜测出的函数名直接放行。
            if (
                analysis_tracker is not None
                and _has_sufficient_analysis_evidence(analysis_tracker)
            ):
                invalid_names = _find_unobserved_answer_function_names(
                    analysis_tracker,
                    step.content,
                )
                invalid_claims = _find_unsupported_analysis_claims(
                    analysis_tracker,
                    step.content,
                )
                if (invalid_names or invalid_claims) and step_index < max_steps - 1:
                    pending_user_nudge = _build_analysis_fact_correction_nudge(
                        analysis_tracker,
                        invalid_names,
                        invalid_claims,
                    )
                    log_event(
                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮检测到缺少证据支撑的分析结论，要求模型自纠: "
                        f"{', '.join((invalid_names + invalid_claims)[:6])}"
                    )
                    continue

            final_content = step.content
            if analysis_tracker is not None:
                final_content = _normalize_analysis_answer_content(
                    analysis_tracker,
                    step.content,
                )
                step.content = final_content

            builder.add_assistant(final_content)

            # 从最终 assistant 回复里尝试抽一条关键决策。
            # 这不是为了记录所有回答，而是尽量保留“已经确认的方向或约束”。
            if memory_pipeline is not None:
                memory_pipeline.record_assistant_reply(
                    working_memory,
                    content=final_content,
                )
            else:
                decisions = extract_decisions_from_assistant(final_content)
                for decision in decisions:
                    working_memory.protect(
                        decision,
                        entry_type="key_decision",
                        ttl_seconds=3600,
                        importance=0.95,
                    )
                    working_memory.protect(
                        decision,
                        entry_type="reflection_decision",
                        ttl_seconds=3600,
                        importance=0.95,
                    )

            persist_post_response_working_memory_state(
                session=session,
                working_memory=working_memory,
            )
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

            if (
                analysis_tracker is not None
                and _should_redirect_analysis_to_structure_first(analysis_tracker, step.calls)
            ):
                # 链路分析早期如果一上来就 read_file，很容易在半截源码上脑补流程。
                # 先强制拿一份 file_overview / AST / symbol 级结构，再决定读哪一段源码。
                pending_user_nudge = NUDGE_ANALYSIS_TOOL_PRIORITY + "\n" + NUDGE_ANALYSIS_STRUCTURE_FIRST
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮检测到分析任务过早直接 read_file，要求先获取结构化证据"
                )
                continue

            if (
                analysis_tracker is not None
                and _should_block_redundant_analysis_calls(
                    analysis_tracker,
                    calls=step.calls,
                    step_index=step_index,
                    max_steps=max_steps,
                    is_exploration_tool=_is_exploration_tool,
                )
            ):
                blocked_analysis_tool_call_count += 1
                # 第一次拦截时先温和提示“证据够了可以回答”；
                # 连续第二次还想继续探索，就升级成强制直接作答。
                if blocked_analysis_tool_call_count >= 2:
                    pending_user_nudge = _build_analysis_force_answer_nudge(analysis_tracker)
                else:
                    pending_user_nudge = _build_analysis_convergence_nudge(analysis_tracker)
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮检测到证据已足够，拦截重复探索工具并要求直接作答"
                )
                continue

            # 依次记录、执行并回写每个工具调用结果
            blocked_analysis_tool_call_count = 0
            for call in step.calls:
                tool_name = call["tool_name"]
                tool_input = call["input"]
                tool_use_id = call["id"]
                tool_target = _extract_tool_target(tool_input)

                # 这里只沉淀“当前任务正在碰哪些路径/命令/目标”。
                # 后面的 working_memory 和 memory_pipeline 会把这些线索变成短期上下文。
                if memory_pipeline is not None:
                    memory_pipeline.record_tool_call(
                        working_memory,
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                else:
                    for path in extract_active_paths(tool_name, tool_input):
                        working_memory.protect(
                            path,
                            entry_type="active_task",
                            ttl_seconds=1800,
                            importance=0.8,
                        )
                        working_memory.protect(
                            path,
                            entry_type="reflection_file",
                            ttl_seconds=1800,
                            importance=0.7,
                        )

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

                if _is_exploration_tool(tool_name):
                    # exploration_history 只是一个轻量信号，
                    # 用来判断是不是连续在探索同一个目标。
                    exploration_history.append((tool_name, tool_target))

                if analysis_tracker is not None:
                    # 真正严格的分析证据都沉淀在 tracker 里，
                    # 例如哪些函数名是观察到的、哪段 read_file 已读过、是否已完整读完目标文件。
                    _record_analysis_evidence(
                        analysis_tracker,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        result=result,
                    )

                # 工具失败时，把错误压成短摘要写进短期工作记忆。
                if not result.ok:
                    if memory_pipeline is not None:
                        memory_pipeline.record_tool_failure(
                            working_memory,
                            tool_name=tool_name,
                            result=result,
                        )
                    else:
                        failure_summary = summarize_failure(tool_name, result)
                        working_memory.protect(
                            failure_summary,
                            entry_type="error_context",
                            ttl_seconds=1800,
                            importance=0.9,
                        )
                        working_memory.protect(
                            failure_summary,
                            entry_type="reflection_failure",
                            ttl_seconds=1800,
                            importance=0.9,
                        )

                # 命中“需要授权”时，不继续喂模型，而是把审批请求返回给 main
                if result.error=="PERMISSION_REQUIRED":
                    command=str(result.meta.get("command", ""))
                    reason = str(result.meta.get("reason", ""))
                    action_key = str(result.meta.get("action_key", ""))

                    # 授权前先补一条占位 tool_result，保证消息协议完整。
                    # 否则历史里只剩 tool_call 没有 tool_result，下一次继续会话时会断链。
                    builder.add_tool_result(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        content="该操作需要用户授权，当前尚未执行。",
                        is_error=True,
                        meta=dict(result.meta),
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
                context_output = result.meta.get("context_output", result.output)
                if not isinstance(context_output, str) or not context_output.strip():
                    context_output = result.output
                builder.add_tool_result(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    content=context_output,
                    is_error=not result.ok,
                    meta=dict(result.meta),
                )

            if _should_inject_convergence_nudge(
                exploration_history=exploration_history,
                step_index=step_index,
                max_steps=max_steps,
            ):
                # 这层提示比分析专项护栏更通用：
                # 只要检测到连续探索或接近步数上限，就提醒模型先判断是否已经够答。
                if analysis_tracker is not None and _has_sufficient_analysis_evidence(analysis_tracker):
                    pending_user_nudge = _build_analysis_convergence_nudge(analysis_tracker)
                else:
                    pending_user_nudge = NUDGE_AFTER_TOOL_RESULT
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮注入临时工具结果提示"
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
