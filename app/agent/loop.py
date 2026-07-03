from __future__ import annotations

"""Agent 主循环，负责模型调用、工具执行、审批中断与结果回写。"""

import time

from app.agent.analysis_guard import (
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
from app.context.auto_compact import (
    is_context_overflow_error,
    recover_from_context_overflow,
)
from app.context.runtime import prepare_agent_context
from app.context.history_summarizer import OlderHistorySummarizer
from app.logger import log_event
from app.agent.message_builder import MessageBuilder
from app.state.session import SessionData
from app.agent.tooling import ToolRegistry
from app.types import AgentStep, ApprovalRequest, ChatMessage, ModelAdapter, ToolContext, ToolResult
from app.tui.events import (
    AgentEvent,
    ApprovalEvent,
    DoneEvent,
    ErrorEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolRunningEvent,
    UsageEvent,
)

NUDGE_CONTINUE = (
    "继续推进：如果现有信息已经足够，请直接给出最终答案；"
    "否则只执行下一步最必要的动作，不要重复读取相同文件、符号或目录。"
)
NUDGE_AFTER_TOOL_RESULT = (
    "你已经拿到了工具结果。请先判断现有证据是否已经足够："
    "足够就直接给最终答案；不够才继续一次最必要的工具调用。不要重复刚看到的内容。"
)
# 探索类工具名称集合（对齐新 8 核心工具集）
EXPLORATION_TOOL_NAMES = {
    "read_file",
    "grep_files",
    "glob_files",
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
    """尽量把工具输入归一成"当前在看什么"。"""
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
    history_summarizer: OlderHistorySummarizer | None = None,
    history: list[ChatMessage] | None = None,
    max_steps: int = 50,
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

    # 从"新用户输入"开始进入主循环
    return _run_agent_loop(
        builder=builder,
        model=model,
        tool_registry=tool_registry,
        tool_context=tool_context,
        session=session,
        max_steps=max_steps,
        history_summarizer=history_summarizer,
        session_id=session_id,
    )


def continue_agent_from_history(
    history: list[ChatMessage],
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    session: SessionData,
    history_summarizer: OlderHistorySummarizer | None = None,
    max_steps: int = 50,
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
        # 分析类任务会额外维护一份"证据账本"。
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
        context_started_at = time.perf_counter()
        prepared_context = prepare_agent_context(
            full_history=full_history,
            session=session,
            tool_registry=tool_registry,
            history_summarizer=history_summarizer,
        )
        context_cost = time.perf_counter() - context_started_at

        # 记录本轮上下文裁剪结果和 token 占用情况，便于观察策略是否生效。
        microcompact_reason = str(
            prepared_context.compaction_history_entry.get("microcompact_reason", "")
        ).strip()
        microcompact_tool_rounds = prepared_context.compaction_history_entry.get(
            "microcompact_tool_rounds", 0
        )
        microcompact_keep_recent_rounds = prepared_context.compaction_history_entry.get(
            "microcompact_keep_recent_rounds", 0
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
            f"microcompact_tool_rounds={microcompact_tool_rounds} "
            f"microcompact_keep_recent_rounds={microcompact_keep_recent_rounds} "
            f"microcompact_cooldown_left={float(microcompact_cooldown_remaining):.0f}s "
            f"steps={','.join(prepared_context.pipeline_steps)}"
        )

        messages = _append_transient_user_nudge(
            prepared_context.messages,
            pending_user_nudge,
        )
        # nudge 只对"这一次模型请求"生效，不会写回 builder 历史。
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
            model_started_at = time.perf_counter()
            step = model.next(messages=messages)
            model_cost = time.perf_counter() - model_started_at

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
                        model_cost = time.perf_counter() - model_started_at
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
                            # 这里拦的不是"回答不好"，而是"引用了未观察到的函数名/统计数字/step.type"。
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

                            post_started_at = time.perf_counter()
                            post_cost = time.perf_counter() - post_started_at
                            step_cost = time.perf_counter() - step_started_at
                            total_cost = time.perf_counter() - loop_started_at
                            log_event(
                                f"[session={session_id or '-'}] 第 {step_index + 1} 轮恢复后直接返回答案 "
                                f"step耗时={step_cost:.3f}s context={context_cost:.3f}s "
                                f"model={model_cost:.3f}s post={post_cost:.3f}s 总耗时={total_cost:.3f}s"
                            )
                            _check_memory_review(list(builder.build()), session)
                            return step, builder.build()

            # 模型调用异常时兜底为最终回答，避免主循环直接崩掉
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮模型调用异常: {error}"
            )
            # 记录最近一次模型失败
            log_event(
                f"[session={session_id or '-'}] 模型调用失败记录: {error}"
            )
            fallback = AgentStep(
                type="assistant",
                content=f"模型调用失败: {error}",
                kind="final",
            )
            builder.add_assistant(fallback.content)
            _check_memory_review(list(builder.build()), session)
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
                    f"step耗时={step_cost:.3f}s context={context_cost:.3f}s model={model_cost:.3f}s"
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

            post_started_at = time.perf_counter()
            post_cost = time.perf_counter() - post_started_at
            # 记录当前 step 和整轮总耗时
            step_cost = time.perf_counter() - step_started_at
            total_cost = time.perf_counter() - loop_started_at
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮直接返回答案 "
                f"step耗时={step_cost:.3f}s context={context_cost:.3f}s "
                f"model={model_cost:.3f}s post={post_cost:.3f}s 总耗时={total_cost:.3f}s"
            )
            _check_memory_review(list(builder.build()), session)
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
                # 先强制拿一份 grep_files / glob_files 级结构，再决定读哪一段源码。
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
                # 第一次拦截时先温和提示"证据够了可以回答"；
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
            tool_total_cost = 0.0
            for call in step.calls:
                tool_name = call["tool_name"]
                tool_input = call["input"]
                tool_use_id = call["id"]
                tool_target = _extract_tool_target(tool_input)

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
                tool_total_cost += tool_cost
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

                # 工具失败时记录日志
                if not result.ok:
                    log_event(
                        f"[session={session_id or '-'}] 工具 {tool_name} 执行失败: {result.error}",
                    )

                # 命中"需要授权"时，不继续喂模型，而是把审批请求返回给 main
                if result.error=="PERMISSION_REQUIRED":
                    command=str(result.meta.get("command", ""))
                    reason = str(result.meta.get("reason", ""))
                    action_key = str(result.meta.get("action_key", ""))
                    suggestions = result.meta.get("suggestions")  # 规则建议列表

                    # 授权前先补一条占位 tool_result，保证消息协议完整。
                    # 否则历史里只剩 tool_call 没有 tool_result，下一次继续会话时会断链。
                    builder.add_tool_result(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        content="该操作需要用户授权，当前尚未执行。",
                        is_error=True,
                        meta=dict(result.meta),
                    )

                    # ── 构建审批消息（包含规则建议信息）──
                    approval_message = (
                        "该操作需要用户授权。\n"
                        f"工具: {tool_name}\n"
                        f"命令: {command}\n"
                        f"原因: {reason}"
                    )
                    if suggestions:
                        approval_message += f"\n建议规则: {len(suggestions)} 条"

                    approval_step = AgentStep(
                        type="approval",
                        content=approval_message,
                        approval=ApprovalRequest(
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            action_key=action_key,
                            message=approval_message,
                            input_data=tool_input,
                            # 传递规则建议给 TUI 展示
                            suggestions=suggestions,
                        ),
                    )
                    return approval_step, builder.build()

                # 命中"工作目录越界"时，暂停并让用户选择是否加入工作目录
                if result.error == "WORKSPACE_ACCESS_REQUIRED":
                    raw_path = str(result.meta.get("path", ""))
                    reason = str(result.meta.get("reason", ""))
                    action_key = str(result.meta.get("action_key", ""))

                    # 授权前先补一条占位 tool_result，保证消息协议完整
                    builder.add_tool_result(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        content="目标路径不在工作目录范围内，需要用户授权。",
                        is_error=True,
                        meta=dict(result.meta),
                    )

                    approval_message = (
                        "目标路径不在工作目录范围内。\n"
                        f"工具: {tool_name}\n"
                        f"路径: {raw_path}\n"
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
                            approval_type="workspace_access",
                            workspace_path=raw_path,
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
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具阶段结束 "
                f"step耗时={step_cost:.3f}s context={context_cost:.3f}s "
                f"model={model_cost:.3f}s tools={tool_total_cost:.3f}s"
            )
            continue

        # 情况三：遇到未知返回类型时兜底退出
        fallback = AgentStep(
            type="assistant",
            content="未识别的模型返回类型。",
            kind="final",
        )
        builder.add_assistant(fallback.content)
        _check_memory_review(list(builder.build()), session)
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
    _check_memory_review(list(builder.build()), session)
    return fallback, builder.build()


def stream_agent(
    user_input: str,
    model: object,  # OpenAIModelAdapter，避免循环依赖不写类型注解
    tool_registry: object,  # ToolRegistry
    tool_context: object,  # ToolContext
    session: object,  # SessionData
    history_summarizer: object | None,  # OlderHistorySummarizer | None
    history: list | None,  # list[ChatMessage] | None
    max_steps: int = 50,  # 最大循环步数，防止死循环
    session_id: str = "",  # 当前会话 ID
) -> object:  # Generator[AgentEvent, None, None]
    """流式执行一轮 Agent 请求。

    与 run_agent_once 使用相同的底层逻辑（prepare_agent_context、工具执行），
    但通过 stream_chat 把模型输出拆成流式 AgentEvent 逐条 yield，
    而不是等所有步骤完成后一次性返回。

    调用方用 for event in stream_agent(...) 消费事件流：
        for event in stream_agent(...):
            if isinstance(event, ThinkingEvent): ...
            elif isinstance(event, TextEvent): ...
            ...

    中文注释：这个函数是 TUI 改造的核心——把原来阻塞式的
    "调模型 → 等全部结果 → print" 流程改为
    "调模型 → 边出边 yield → TUI 边收边渲染" 的流式流水线。
    """
    import json

    from app.context.auto_compact import (
        is_context_overflow_error,
        recover_from_context_overflow,
    )
    from app.context.runtime import (
        prepare_agent_context,
    )
    from app.agent.message_builder import MessageBuilder
    from app.infra.model_registry import OpenAIModelAdapter
    from app.types import (
        AgentStep,
        ApprovalRequest,
        ChatMessage,
        ToolContext,
        ToolResult,
    )

    # 没有历史时用空列表兜底
    if history is None:
        history = []

    # 用 MessageBuilder 统一管理本轮消息
    builder = MessageBuilder()
    builder.extend(history)
    # 审批后继续执行时 user_input 为空，不应追加空的 user 消息
    if user_input and user_input.strip():
        builder.add_user(user_input)

    loop_started_at = time.perf_counter()
    pending_user_nudge: str | None = None
    exploration_history: list[tuple[str, str]] = []

    # ---- 分析护栏：初始化证据追踪器 ----
    # 对代码分析类任务创建 tracker，后续每次工具调用都会往里面沉淀观察到的函数名、文件、行数。
    # 最终回答前用 tracker 校验是否引用了未经观察的符号，防止模型"脑补"。
    task_text = _extract_latest_real_user_message(builder.build())
    analysis_tracker: dict[str, object] | None = None
    blocked_analysis_tool_call_count = 0
    if _is_code_analysis_request(task_text):
        analysis_tracker = _create_analysis_tracker(task_text)
        target_resolution_nudge = _build_analysis_target_resolution_nudge(analysis_tracker)
        pending_user_nudge = NUDGE_ANALYSIS_TOOL_PRIORITY
        if target_resolution_nudge:
            pending_user_nudge = pending_user_nudge + "\n" + target_resolution_nudge

    log_event(
        f"[session={session_id or '-'}] 开始一轮 Agent 流式请求",
        echo=False,
    )

    # 主循环：每轮 = 准备上下文 → 流式调模型 → 处理工具调用
    for step_index in range(max_steps):
        step_started_at = time.perf_counter()

        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮循环开始",
            echo=False,
        )

        # ---- 第一步：准备上下文窗口 ----
        if step_index == 0:
            yield ThinkingEvent("正在分析请求并准备上下文...")

        context_started_at = time.perf_counter()
        prepared_context = prepare_agent_context(
            full_history=list(builder.build()),
            session=session,
            tool_registry=tool_registry,
            history_summarizer=history_summarizer,
        )
        context_cost = time.perf_counter() - context_started_at

        # 上下文准备完成，进入流式循环

        messages = _append_transient_user_nudge(
            prepared_context.messages,
            pending_user_nudge,
        )
        if pending_user_nudge:
            pending_user_nudge = None

        # ---- 第二步：流式调模型 ----
        if step_index == 0:
            yield ThinkingEvent("正在等待模型响应...")

        # 收集流式结果的缓冲区
        collected_text = ""  # 累积的文本回答
        _api_total_tokens = 0  # API 返回的真实 token 总数（用于后续 token 预估基线）
        tool_calls_buf: dict[int, dict] = {}  # tool_index → {id, name, args_str}

        model_started_at = time.perf_counter()
        try:
            # 逐 chunk 消费模型的流式输出
            for chunk in model.stream_chat(messages=messages):
                if chunk.type == "text":
                    # 文本片段 → 累积到 collected_text 并实时 yield
                    collected_text += chunk.text
                    yield TextEvent(text=chunk.text)

                elif chunk.type == "tool_call_name":
                    # 工具调用第一块：包含 id 和函数名
                    # 参数在后续 tool_call_args 块中逐步到达，这里先不 yield Event
                    idx = chunk.tool_index
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {"id": "", "name": "", "args_str": ""}
                    tool_calls_buf[idx]["id"] = chunk.tool_id
                    tool_calls_buf[idx]["name"] = chunk.text

                elif chunk.type == "tool_call_args":
                    # 工具调用后续块：arguments JSON 增量片段
                    idx = chunk.tool_index
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {"id": "", "name": "", "args_str": ""}
                    tool_calls_buf[idx]["args_str"] += chunk.text

                elif chunk.type == "usage":
                    # API 返回的 token 用量 + prompt cache 命中统计
                    # chunk.text 格式: "total,cache_hit,cache_miss"
                    parts = chunk.text.split(",")
                    total_tokens = int(parts[0]) if len(parts) > 0 and parts[0].lstrip("-").isdigit() else 0
                    cache_hit = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 0
                    cache_miss = int(parts[2]) if len(parts) > 2 and parts[2].lstrip("-").isdigit() else 0
                    yield UsageEvent(
                        total_tokens=total_tokens,
                        cache_hit_tokens=cache_hit,
                        cache_miss_tokens=cache_miss,
                    )
                    _api_total_tokens = total_tokens  # 保存到外层，写入 assistant 消息 meta
                    # 记录缓存命中率到 debug.log，方便排查和性能分析
                    cache_total = cache_hit + cache_miss
                    if cache_total > 0:
                        rate = (cache_hit / cache_total) * 100
                        log_event(
                            f"[session={session_id or '-'}] prompt_cache: hit={cache_hit} miss={cache_miss} rate={rate:.1f}% total={total_tokens}",
                            echo=False,
                        )
            model_cost = time.perf_counter() - model_started_at

        except Exception as error:
            # 模型调用失败
            if is_context_overflow_error(error):
                # 尝试上下文溢出恢复
                recovery_result = recover_from_context_overflow(
                    messages=messages,
                    usable_budget=prepared_context.stats.usable_budget,
                )
                if recovery_result.recovered:
                    log_event(
                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮触发上下文溢出自动恢复",
                        echo=False,
                    )
                    yield ErrorEvent(
                        message="上下文溢出，已自动压缩后重试。"
                    )
                    continue

            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮模型调用异常: {error}",
                echo=False,
            )
            yield ErrorEvent(message=f"模型调用失败: {error}")
            # 兜底：返回错误信息
            fallback = AgentStep(
                type="assistant",
                content=f"模型调用失败: {error}",
                kind="final",
            )
            builder.add_assistant(fallback.content)
            _check_memory_review(list(builder.build()), session)
            yield DoneEvent(step=fallback, history=builder.build())
            return

        # ---- 第三步：处理工具调用 ----
        if tool_calls_buf:
            # 把收集到的流式工具调用片段组装成完整 ToolCall
            calls = []
            for idx in sorted(tool_calls_buf.keys()):
                info = tool_calls_buf[idx]
                tool_name = info["name"]
                tool_use_id = info["id"]
                args_str = info["args_str"]

                # 安全解析 JSON 参数
                try:
                    parsed_input = json.loads(args_str) if args_str.strip() else {}
                except json.JSONDecodeError:
                    parsed_input = {}

                # 现在 args 已完整收集 → yield ToolCallEvent 带完整参数
                yield ToolCallEvent(name=tool_name, args=parsed_input)

                calls.append({
                    "tool_name": tool_name,
                    "input": parsed_input,
                    "id": tool_use_id,
                })

            # ---- 分析护栏：工具调用前的结构优先与冗余拦截 ----
            if (
                analysis_tracker is not None
                and _should_redirect_analysis_to_structure_first(analysis_tracker, calls)
            ):
                pending_user_nudge = (
                    NUDGE_ANALYSIS_TOOL_PRIORITY + "\n" + NUDGE_ANALYSIS_STRUCTURE_FIRST
                )
                continue

            if (
                analysis_tracker is not None
                and _should_block_redundant_analysis_calls(
                    analysis_tracker,
                    calls=calls,
                    step_index=step_index,
                    max_steps=max_steps,
                    is_exploration_tool=_is_exploration_tool,
                )
            ):
                blocked_analysis_tool_call_count += 1
                if blocked_analysis_tool_call_count >= 2:
                    pending_user_nudge = _build_analysis_force_answer_nudge(analysis_tracker)
                else:
                    pending_user_nudge = _build_analysis_convergence_nudge(analysis_tracker)
                continue

            # ── 并行执行工具（对标 Claude Code isConcurrencySafe）──
            # 将工具按并发安全性分类：只读工具可并行，写工具必须串行
            # 并发安全的工具集合（对标 Claude Code 各工具的 isConcurrencySafe 返回值）
            _CONCURRENCY_SAFE = {
                "read_file", "grep_files", "glob_files",
                "web_search", "web_fetch", "ask_user", "agent_dispatch",
                "memory",
            }
            parallel_calls: list[dict] = []
            sequential_calls: list[dict] = []

            for call in calls:
                tool_name = call["tool_name"]
                if tool_name in _CONCURRENCY_SAFE:
                    parallel_calls.append(call)
                elif tool_name == "run_command":
                    # run_command 按命令风险分级：只读命令可并行，写/高风险命令串行
                    raw_cmd = str(call.get("input", {}).get("command", ""))
                    if raw_cmd:
                        from app.permissions.command_safety import classify_command_risk
                        risk = classify_command_risk(raw_cmd)
                        if risk == "read_only":
                            parallel_calls.append(call)
                        else:
                            sequential_calls.append(call)
                    else:
                        sequential_calls.append(call)
                else:
                    # edit_file / write_file 等写工具必须串行，避免文件冲突
                    sequential_calls.append(call)

            # 所有工具调用统一写入消息历史（保持协议顺序）
            for call in calls:
                builder.add_tool_call(
                    tool_use_id=call["id"],
                    tool_name=call["tool_name"],
                    input_data=call["input"],
                )

            # ── 阶段1：线程池并发执行只读工具 ──
            if parallel_calls:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                # 通知 UI 所有并行工具开始执行
                for call in parallel_calls:
                    yield ToolRunningEvent(name=call["tool_name"])

                # 并发提交所有只读工具到线程池
                parallel_results: dict[int, ToolResult] = {}
                with ThreadPoolExecutor(max_workers=min(8, len(parallel_calls))) as executor:
                    future_to_idx = {}
                    for idx, call in enumerate(parallel_calls):
                        future = executor.submit(
                            tool_registry.execute_tool,
                            tool_name=call["tool_name"],
                            input_data=call["input"],
                            context=tool_context,
                        )
                        future_to_idx[future] = (idx, call)

                    # 等所有工具完成
                    for future in as_completed(future_to_idx):
                        idx, call = future_to_idx[future]
                        try:
                            result = future.result()
                        except Exception as tool_error:
                            result = ToolResult(
                                ok=False,
                                output=f"工具调用发生未捕获异常: {tool_error}",
                                error="UNCAUGHT_TOOL_ERROR",
                                meta={"tool_name": call["tool_name"]},
                            )
                        parallel_results[idx] = result

                # 按原始调用顺序处理结果（保证消息协议一致性）
                for idx, call in enumerate(parallel_calls):
                    result = parallel_results[idx]
                    tool_name = call["tool_name"]
                    tool_input = call["input"]
                    tool_use_id = call["id"]

                    log_event(
                        f"[session={session_id or '-'}] 工具 {tool_name} "
                        f"返回 ok={result.ok} error={result.error}",
                        echo=False,
                    )

                    # 构建结果摘要并通知 UI
                    raw = str(result.output).strip()
                    preview = raw[:120].replace("\n", " ") + ("..." if len(raw) > 120 else "")
                    summary = preview
                    if result.error:
                        summary += f"  错误 {result.error}"
                    yield ToolResultEvent(name=tool_name, summary=summary, ok=result.ok)

                    # 分析护栏和探索历史记录
                    if analysis_tracker is not None:
                        _record_analysis_evidence(
                            analysis_tracker, tool_name=tool_name,
                            tool_input=tool_input, result=result,
                        )
                    if _is_exploration_tool(tool_name):
                        tool_target = _extract_tool_target(tool_input)
                        exploration_history.append((tool_name, tool_target))

                    # 权限检查：并行工具中如有需要授权的，中断流程
                    if result.error == "PERMISSION_REQUIRED":
                        command = str(result.meta.get("command", ""))
                        reason = str(result.meta.get("reason", ""))
                        action_key = str(result.meta.get("action_key", ""))
                        suggestions = result.meta.get("suggestions")
                        builder.add_tool_result(
                            tool_use_id=tool_use_id, tool_name=tool_name,
                            content="该操作需要用户授权，当前尚未执行。",
                            is_error=True, meta=dict(result.meta),
                        )
                        approval_message = (
                            "该操作需要用户授权。\n"
                            f"工具: {tool_name}\n命令: {command}\n原因: {reason}"
                        )
                        if suggestions:
                            approval_message += f"\n建议规则: {len(suggestions)} 条"
                        approval_step = AgentStep(
                            type="approval", content=approval_message,
                            approval=ApprovalRequest(
                                tool_name=tool_name, tool_use_id=tool_use_id,
                                action_key=action_key, message=approval_message,
                                input_data=tool_input, suggestions=suggestions,
                            ),
                        )
                        # 为所有未处理的工具补占位 tool_result，保证消息协议完整
                        for _j, _oc in enumerate(parallel_calls):
                            if _j == idx:
                                continue
                            _r = parallel_results.get(_j)
                            if _r is not None:
                                builder.add_tool_result(
                                    tool_use_id=_oc["id"], tool_name=_oc["tool_name"],
                                    content=str(_r.output), is_error=not _r.ok,
                                    meta=dict(_r.meta),
                                )
                        # 串行工具还没执行，统一补占位
                        for _sc in sequential_calls:
                            builder.add_tool_result(
                                tool_use_id=_sc["id"], tool_name=_sc["tool_name"],
                                content="工具执行被中断（前置工具需要授权）",
                                is_error=True, meta={},
                            )
                        yield ApprovalEvent(approval=approval_step.approval)
                        yield DoneEvent(step=approval_step, history=builder.build())
                        return

                    # 将成功结果写入消息历史
                    context_output = result.meta.get("context_output", result.output)
                    if not isinstance(context_output, str) or not context_output.strip():
                        context_output = result.output
                    builder.add_tool_result(
                        tool_use_id=tool_use_id, tool_name=tool_name,
                        content=context_output, is_error=not result.ok,
                        meta=dict(result.meta),
                    )

                    if not result.ok:
                        log_event(
                            f"[session={session_id or '-'}] 工具 {tool_name} 执行失败: {result.error}",
                            echo=False,
                        )

            # ── 阶段2：串行执行写工具（edit_file/write_file/高风险命令等）──
            for call in sequential_calls:
                tool_name = call["tool_name"]
                tool_input = call["input"]
                tool_use_id = call["id"]

                # 通知 UI：工具开始执行
                yield ToolRunningEvent(name=tool_name)

                # 执行工具
                tool_started_at = time.perf_counter()
                try:
                    result = tool_registry.execute_tool(
                        tool_name=tool_name,
                        input_data=tool_input,
                        context=tool_context,
                    )
                except Exception as tool_error:
                    tool_cost = time.perf_counter() - tool_started_at
                    log_event(
                        f"[session={session_id or '-'}] 工具 {tool_name} 执行异常: {tool_error} "
                        f"耗时={tool_cost:.3f}s",
                        echo=False,
                    )
                    result = ToolResult(
                        ok=False,
                        output=f"工具调用发生未捕获异常: {tool_error}",
                        error="UNCAUGHT_TOOL_ERROR",
                        meta={"tool_name": tool_name},
                    )

                tool_cost = time.perf_counter() - tool_started_at
                log_event(
                    f"[session={session_id or '-'}] 工具 {tool_name} "
                    f"返回 ok={result.ok} error={result.error} 耗时={tool_cost:.3f}s",
                    echo=False,
                )

                # 构造工具结果摘要，通知 UI
                raw = str(result.output).strip()
                preview = raw[:120].replace("\n", " ") + ("..." if len(raw) > 120 else "")
                summary = preview
                if result.error:
                    summary += f"  错误 {result.error}"

                yield ToolResultEvent(
                    name=tool_name,
                    summary=summary,
                    ok=result.ok,
                )

                # 分析护栏：记录观察到的文件/符号/行数
                if analysis_tracker is not None:
                    _record_analysis_evidence(
                        analysis_tracker,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        result=result,
                    )

                # 记录探索历史，用于后续判断是否连续重复探索同一目标
                if _is_exploration_tool(tool_name):
                    tool_target = _extract_tool_target(tool_input)
                    exploration_history.append((tool_name, tool_target))

                # ---- 权限检查：高风险操作需要用户授权 ----
                if result.error == "PERMISSION_REQUIRED":
                    command = str(result.meta.get("command", ""))
                    reason = str(result.meta.get("reason", ""))
                    action_key = str(result.meta.get("action_key", ""))
                    suggestions = result.meta.get("suggestions")

                    # 写入占位 tool_result，保证消息协议完整
                    builder.add_tool_result(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        content="该操作需要用户授权，当前尚未执行。",
                        is_error=True,
                        meta=dict(result.meta),
                    )

                    # 构建审批消息（包含规则建议信息）
                    approval_message = (
                        "该操作需要用户授权。\n"
                        f"工具: {tool_name}\n"
                        f"命令: {command}\n"
                        f"原因: {reason}"
                    )
                    if suggestions:
                        approval_message += f"\n建议规则: {len(suggestions)} 条"

                    approval_step = AgentStep(
                        type="approval",
                        content=approval_message,
                        approval=ApprovalRequest(
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            action_key=action_key,
                            message=approval_message,
                            input_data=tool_input,
                            suggestions=suggestions,
                        ),
                    )
                    # 把审批请求推送给 UI，暂停等待用户确认
                    yield ApprovalEvent(approval=approval_step.approval)
                    yield DoneEvent(step=approval_step, history=builder.build())
                    return

                # ---- 路径权限检查：工作目录越界需要用户授权 ----
                if result.error == "WORKSPACE_ACCESS_REQUIRED":
                    raw_path = str(result.meta.get("path", ""))
                    reason = str(result.meta.get("reason", ""))
                    action_key = str(result.meta.get("action_key", ""))

                    # 写入占位 tool_result，保证消息协议完整
                    builder.add_tool_result(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        content="目标路径不在工作目录范围内，需要用户授权。",
                        is_error=True,
                        meta=dict(result.meta),
                    )

                    approval_message = (
                        "目标路径不在工作目录范围内。\n"
                        f"工具: {tool_name}\n"
                        f"路径: {raw_path}\n"
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
                            approval_type="workspace_access",
                            workspace_path=raw_path,
                        ),
                    )
                    # 把审批请求推送给 UI，暂停等待用户确认
                    yield ApprovalEvent(approval=approval_step.approval)
                    yield DoneEvent(step=approval_step, history=builder.build())
                    return

                # 把成功工具结果写入消息历史
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

                # 工具失败时记录日志
                if not result.ok:
                    log_event(
                        f"[session={session_id or '-'}] 工具 {tool_name} 执行失败: {result.error}",
                        echo=False,
                    )

            # ---- 分析护栏：重置冗余拦截计数器，注入收敛提示 ----
            blocked_analysis_tool_call_count = 0
            if _should_inject_convergence_nudge(
                exploration_history=exploration_history,
                step_index=step_index,
                max_steps=max_steps,
            ):
                if (
                    analysis_tracker is not None
                    and _has_sufficient_analysis_evidence(analysis_tracker)
                ):
                    pending_user_nudge = _build_analysis_convergence_nudge(analysis_tracker)
                else:
                    pending_user_nudge = NUDGE_AFTER_TOOL_RESULT

            # 工具阶段结束，继续下一轮循环
            step_cost = time.perf_counter() - step_started_at
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具阶段结束 "
                f"step耗时={step_cost:.3f}s context={context_cost:.3f}s "
                f"model={model_cost:.3f}s",
                echo=False,
            )
            continue

        # ---- 第四步：返回最终回答 ----
        if collected_text.strip():
            # ---- 分析护栏：事实校验 ----
            # 对代码分析类回答做最后一层检查：
            # 1. 证据是否足够（至少读过目标文件、观察到过关键符号）
            # 2. 回答中是否引用了未经 read_file/grep_files 确认的函数名
            # 3. 是否有无事实依据的分析断言
            if (
                analysis_tracker is not None
                and _has_sufficient_analysis_evidence(analysis_tracker)
            ):
                invalid_names = _find_unobserved_answer_function_names(
                    analysis_tracker,
                    collected_text,
                )
                invalid_claims = _find_unsupported_analysis_claims(
                    analysis_tracker,
                    collected_text,
                )
                if (invalid_names or invalid_claims) and step_index < max_steps - 1:
                    # 发现编造内容，注入纠正提示，要求模型自纠后再答
                    pending_user_nudge = _build_analysis_fact_correction_nudge(
                        analysis_tracker,
                        invalid_names,
                        invalid_claims,
                    )
                    continue

                # 校验通过，规范化答案（去掉可能的幻觉修改）
                collected_text = _normalize_analysis_answer_content(
                    analysis_tracker,
                    collected_text,
                )

            # 把流式收集到的文本写入历史，附带 API 真实 token 用量供后续预估
            builder.add_assistant(collected_text)
            if _api_total_tokens > 0:
                builder.history[-1]["_api_total_tokens"] = _api_total_tokens  # type: ignore[index]

        step = AgentStep(
            type="assistant",
            content=collected_text,
            kind="final",
        )
        step_cost = time.perf_counter() - step_started_at
        total_cost = time.perf_counter() - loop_started_at
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮流式回答完成 "
            f"step耗时={step_cost:.3f}s context={context_cost:.3f}s "
            f"model={model_cost:.3f}s 总耗时={total_cost:.3f}s",
            echo=False,
        )
        _check_memory_review(list(builder.build()), session)
        yield DoneEvent(step=step, history=builder.build())
        return

    # 达到最大步数时停止，防止死循环
    total_cost = time.perf_counter() - loop_started_at
    log_event(
        f"[session={session_id or '-'}] 达到最大循环步数 {max_steps} "
        f"总耗时={total_cost:.3f}s",
        echo=False,
    )
    fallback = AgentStep(
        type="assistant",
        content="已达到最大循环步数，本轮已停止。",
        kind="final",
    )
    builder.add_assistant(fallback.content)
    _check_memory_review(list(builder.build()), session)
    yield DoneEvent(step=fallback, history=builder.build())


# ---------------------------------------------------------------------------
# 后台记忆反思触发器
# ---------------------------------------------------------------------------

_REVIEW_NUDGE_INTERVAL = 10
_review_runner: object = None

# 计数器 key，存放在 session.extra 中，跨会话重启保持
_TURNS_KEY = "_turns_since_memory_review"


def configure_review_runner(runner: object) -> None:
    """注入后台反思 runner，loop 在被调用时自动触发反思。"""
    global _review_runner
    _review_runner = runner


def _check_memory_review(history: list[ChatMessage], session: object) -> None:
    """累计用户轮次，达到阈值时 spawn 后台记忆反思线程（参照 Hermes _memory_nudge_interval）。

    计数器存放在 session.extra 中，跨会话重启保持。
    """
    if _review_runner is None or session is None:
        return
    extra = getattr(session, "extra", {})
    if not isinstance(extra, dict):
        extra = {}
    count = int(extra.get(_TURNS_KEY, 0)) + 1
    extra[_TURNS_KEY] = count
    if count >= _REVIEW_NUDGE_INTERVAL:
        extra[_TURNS_KEY] = 0
        session_id = getattr(session, "session_id", "")
        try:
            getattr(_review_runner, "spawn_review")(history, session_id=session_id)
        except Exception:
            pass
