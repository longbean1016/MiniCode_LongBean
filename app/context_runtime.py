from __future__ import annotations

from dataclasses import dataclass

"""运行时上下文装配层，负责拼接消息窗口、记忆注入和压缩结果。"""

from app.compaction_policy import build_compaction_policy
from app.context_compact_memory import (
    CompactMemorySnapshot,
    build_active_context_snapshot,
    render_active_context_summary,
)
from app.context_auto_compact import AUTO_COMPACT_TRIGGER_RATIO
from app.context_compactor import CompactionResult
from app.context_compactor_pipeline import ContextCompactorPipeline, ContextPipelineResult
from app.context_signal_resolver import (
    resolve_project_constraints,
    resolve_recent_risks,
    resolve_user_preferences,
)
from app.context_manager import (
    DEFAULT_USABLE_CONTEXT_BUDGET,
    ContextPolicy,
    ContextStats,
    collect_context_stats,
    decide_context_policy,
)
from app.context_state import (
    ContextStateData,
    build_history_fingerprint,
    build_token_stats_snapshot,
    load_context_state,
    save_context_state,
)
from app.history_summarizer import OlderHistorySummarizer
from app.history_window import HistoryWindow, build_older_history_summary, select_history_window
from app.memory_pipeline import MemoryPipeline
from app.prompt import build_system_prompt
from app.session import SessionData
from app.tooling import ToolRegistry
from app.types import ChatMessage
from app.user_profile import ResolvedUserPolicy, UserPolicyRule, load_user_profile
from app.working_memory import WorkingMemory

_ANALYSIS_MODE_KEYWORDS = (
    "分析",
    "链路",
    "调用链",
    "串联",
    "流程",
    "入口",
    "结构",
    "梳理",
    "trace",
    "analyze",
    "analysis",
    "call chain",
    "workflow",
    "walkthrough",
)
_ANALYSIS_PINNED_TOOL_NAMES = {
    "file_overview",
    "get_ast_info",
    "find_symbols",
    "locate_symbol",
    "find_references",
}


@dataclass(slots=True)
class PreparedAgentContext:
    """保存一次模型请求前已经准备好的上下文结果。"""

    messages: list[ChatMessage]
    policy: ContextPolicy
    preview_stats: ContextStats
    stats: ContextStats
    active_context_summary: str
    active_context_snapshot: CompactMemorySnapshot
    older_history_summary: str
    resolved_user_preferences: list[str]
    resolved_user_policy: ResolvedUserPolicy
    active_user_rules: list[UserPolicyRule]
    resolved_project_constraints: list[str]
    recent_risks: list[str]
    history_window: HistoryWindow
    compaction_result: CompactionResult
    pipeline_steps: list[str]
    memory_context: str
    user_profile_context: str


def prepare_agent_context(
    *,
    full_history: list[ChatMessage],
    session: SessionData,
    tool_registry: ToolRegistry,
    working_memory: WorkingMemory,
    memory_pipeline: MemoryPipeline | None,
    history_summarizer: OlderHistorySummarizer | None,
) -> PreparedAgentContext:
    """
    组装一次模型调用前的上下文。

    主链路已经统一为 active_context_*：
    1. 先根据预估压力决定压缩级别。
    2. 再构造 active context 摘要与快照。
    3. 最后交给 pipeline 处理 recent window、tool_result 和 auto compact。
    """
    usable_budget = _resolve_usable_budget(session)
    analysis_mode = _infer_analysis_mode_from_history(full_history)
    cached_state = load_context_state(session.workspace, session.session_id)
    resolved_user_preferences = resolve_user_preferences(workspace=session.workspace)
    resolved_user_policy = _resolve_user_policy(session.workspace)
    active_user_rules = resolved_user_policy.active_rules_for(
        _build_user_policy_task_context(
            full_history=full_history,
            working_memory=working_memory,
        )
    )
    user_profile_context = resolved_user_policy.to_prompt_section(active_user_rules)
    base_system_prompt = build_system_prompt(
        tool_registry=tool_registry,
        memory_context="",
        user_profile_context=user_profile_context,
    )

    # 先做压缩前预估，用来判断这一轮的真实上下文压力。
    initial_policy = build_compaction_policy(session)
    preview_recent_messages, preview_summary = _build_preview_source(
        full_history=full_history,
        session=session,
        cached_state=cached_state,
        initial_keep_rounds=max(6, initial_policy.keep_rounds),
    )
    preview_memory_context = _build_preview_memory_context(
        active_context_summary=preview_summary,
        working_memory=working_memory,
    )
    preview_stats = collect_context_stats(
        system_prompt=base_system_prompt,
        recent_messages=preview_recent_messages,
        memory_context=preview_memory_context,
        usable_budget=usable_budget,
    )

    # 先基于“预估上下文”选压缩级别，而不是等真正超预算了再补救。
    policy = decide_context_policy(preview_stats, analysis_mode=analysis_mode)
    session.extra["compaction_level"] = policy.level

    # 命中可复用的 active context 时，优先走增量恢复，
    # 避免每一轮都从 full_history 重新切一次 recent window。
    restored_recent_messages = _try_restore_recent_messages(
        full_history=full_history,
        cached_state=cached_state,
    )
    if restored_recent_messages is not None:
        history_window = HistoryWindow(
            older_messages=[],
            recent_messages=restored_recent_messages,
            older_round_count=0,
            recent_round_count=0,
        )
        older_history_summary = cached_state.older_history_summary if cached_state else ""
    else:
        history_window = select_history_window(
            history=full_history,
            keep_rounds=policy.keep_rounds,
        )
        older_history_summary = _build_compaction_summary(
            history_window=history_window,
            session=session,
            history_summarizer=history_summarizer,
        )

    resolved_project_constraints = resolve_project_constraints(
        memory_pipeline=memory_pipeline,
        working_memory=working_memory,
    )
    recent_risks = resolve_recent_risks(
        working_memory=working_memory,
        cached_risks=cached_state.recent_risks if cached_state is not None else [],
    )

    active_context_snapshot, active_context_summary = _resolve_active_context(
        older_history_summary=older_history_summary,
        working_memory=working_memory,
        cached_state=cached_state,
        resolved_user_preferences=resolved_user_preferences,
        resolved_project_constraints=resolved_project_constraints,
        recent_risks=recent_risks,
    )

    fixed_overhead_tokens = max(0, preview_stats.total_tokens - preview_stats.recent_tokens)
    pipeline = ContextCompactorPipeline()
    # pipeline 负责真正把“摘要、memory、tool_result 裁剪、系统提示词”拼成最终请求。
    # prepare_agent_context 自己只做编排，不直接改写消息细节。
    pipeline_result, memory_context, final_stats, messages = _build_compacted_request(
        pipeline=pipeline,
        source_recent_messages=history_window.recent_messages,
        summary_source_messages=select_history_window(
            history=full_history,
            keep_rounds=policy.keep_rounds,
        ).recent_messages,
        policy=policy,
        session=session,
        working_memory=working_memory,
        memory_pipeline=memory_pipeline,
        active_context_summary=active_context_summary,
        active_context_snapshot=active_context_snapshot,
        tool_registry=tool_registry,
        usable_budget=usable_budget,
        fixed_overhead_tokens=fixed_overhead_tokens,
        base_system_prompt=base_system_prompt,
        user_profile_context=user_profile_context,
        # 把缓存状态直接传给压缩流水线，让 microcompact 节流能接上旧轮次。
        cached_state=cached_state,
        analysis_mode=analysis_mode,
    )
    active_context_snapshot = (
        pipeline_result.resolved_active_context_snapshot or active_context_snapshot
    )
    active_context_summary = (
        pipeline_result.resolved_active_context_summary or active_context_summary
    )

    # 如果 memory 注入之后上下文还是高压，就再强制走一次 auto compact。
    # 这一步是第二道保险，避免“摘要已经做了，但 memory 一加回来又超压”。
    if (
        final_stats.usage_ratio >= AUTO_COMPACT_TRIGGER_RATIO
        and pipeline_result.auto_compact_result.strategy != "full"
    ):
        pipeline_result, memory_context, final_stats, messages = _build_compacted_request(
            pipeline=pipeline,
            source_recent_messages=history_window.recent_messages,
            summary_source_messages=select_history_window(
                history=full_history,
                keep_rounds=policy.keep_rounds,
            ).recent_messages,
            policy=policy,
            session=session,
            working_memory=working_memory,
            memory_pipeline=memory_pipeline,
            active_context_summary=active_context_summary,
            active_context_snapshot=active_context_snapshot,
            tool_registry=tool_registry,
            usable_budget=usable_budget,
            fixed_overhead_tokens=fixed_overhead_tokens,
            base_system_prompt=base_system_prompt,
            user_profile_context=user_profile_context,
            force_auto_compact=True,
            # 第二次强制 auto compact 时也继续复用同一个缓存状态，
            # 避免第一次处理刚写下的 microcompact 节流信息丢失。
            cached_state=cached_state,
            analysis_mode=analysis_mode,
        )
        active_context_snapshot = (
            pipeline_result.resolved_active_context_snapshot or active_context_snapshot
        )
        active_context_summary = (
            pipeline_result.resolved_active_context_summary or active_context_summary
        )

    _save_active_context_state(
        full_history=full_history,
        session=session,
        active_context_summary=active_context_summary,
        active_context_snapshot=active_context_snapshot,
        older_history_summary=older_history_summary,
        resolved_user_preferences=resolved_user_preferences,
        resolved_project_constraints=resolved_project_constraints,
        recent_risks=recent_risks,
        compacted_messages=pipeline_result.messages,
        preview_stats=preview_stats,
        final_stats=final_stats,
        pipeline_history_entry=pipeline_result.compaction_history_entry,
        cached_state=cached_state,
        compaction_level=policy.level,
        # 把最新 microcompact 时间写回 context_state，供下一轮增量恢复继续复用。
        last_microcompact_at=pipeline_result.last_microcompact_at,
        auto_compact_failure_count=pipeline_result.auto_compact_failure_count,
        auto_compact_suppressed_until=pipeline_result.auto_compact_suppressed_until,
    )

    return PreparedAgentContext(
        messages=messages,
        policy=policy,
        preview_stats=preview_stats,
        stats=final_stats,
        active_context_summary=active_context_summary,
        active_context_snapshot=active_context_snapshot,
        older_history_summary=older_history_summary,
        resolved_user_preferences=resolved_user_preferences,
        resolved_user_policy=resolved_user_policy,
        active_user_rules=active_user_rules,
        resolved_project_constraints=resolved_project_constraints,
        recent_risks=recent_risks,
        history_window=history_window,
        compaction_result=pipeline_result.compaction_result,
        pipeline_steps=list(pipeline_result.steps_taken),
        memory_context=memory_context,
        user_profile_context=user_profile_context,
    )


def _resolve_usable_budget(session: SessionData) -> int:
    """解析当前会话可用的上下文预算。"""
    raw_value = session.extra.get("usable_context_budget", DEFAULT_USABLE_CONTEXT_BUDGET)
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_USABLE_CONTEXT_BUDGET


def _infer_analysis_mode_from_history(full_history: list[ChatMessage]) -> bool:
    """根据最近真实用户请求判断当前是否是代码分析任务。"""
    for message in reversed(full_history):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        lowered = content.lower()
        if any(keyword in content for keyword in _ANALYSIS_MODE_KEYWORDS):
            return True
        if any(keyword in lowered for keyword in _ANALYSIS_MODE_KEYWORDS):
            return True
        return False
    return False


def _build_preview_memory_context(
    *,
    active_context_summary: str,
    working_memory: WorkingMemory,
) -> str:
    """构造预估阶段的轻量 memory_context，不触发长期记忆检索。"""
    sections: list[str] = []
    if active_context_summary.strip():
        sections.append(active_context_summary.strip())

    try:
        working_memory_text = _build_preview_working_memory_brief(working_memory)
    except Exception:
        working_memory_text = ""

    if working_memory_text:
        sections.append(working_memory_text)

    return "\n".join(sections).strip()


def _build_preview_working_memory_brief(working_memory: WorkingMemory) -> str:
    """预估阶段只保留少量高价值 working memory，避免预估值过度膨胀。"""
    sections: list[str] = []
    slot_specs = (
        ("当前任务", "active_task", 2),
        ("关键决策", "key_decision", 2),
        ("最近风险", "recent_risk", 2),
        ("错误上下文", "error_context", 1),
    )
    for title, entry_type, limit in slot_specs:
        entries = working_memory.get_entries_by_type(entry_type)[-limit:]
        if not entries:
            continue
        sections.append(f"## {title}")
        for entry in entries:
            content = " ".join(str(entry.content).strip().split())
            if content:
                sections.append(f"- {content[:120]}")
    return "\n".join(sections).strip()


def _build_compaction_summary(
    *,
    history_window: HistoryWindow,
    session: SessionData,
    history_summarizer: OlderHistorySummarizer | None,
) -> str:
    """
    构造 older history 的压缩摘要。

    这段摘要只作为构建 active context 的原料，不再直接主导模型注入上下文。
    """
    if history_summarizer is not None:
        return history_summarizer.summarize(
            session=session,
            older_messages=history_window.older_messages,
            older_round_count=history_window.older_round_count,
        )
    return build_older_history_summary(history_window.older_messages)


def _resolve_user_policy(workspace: str) -> ResolvedUserPolicy:
    """从工作区 USER.md 解析结构化用户策略。"""
    profile = load_user_profile(workspace)
    if profile is None:
        return ResolvedUserPolicy()
    return profile.build_resolved_policy()


def _build_user_policy_task_context(
    *,
    full_history: list[ChatMessage],
    working_memory: WorkingMemory,
) -> str:
    """提取当前任务文本，用于筛选命中的路径规则。"""
    parts: list[str] = []
    primary_intent = working_memory.get_primary_user_intent().strip()
    if primary_intent:
        parts.append(primary_intent)

    recent_user_messages: list[str] = []
    for message in reversed(full_history):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        recent_user_messages.append(content)
        if len(recent_user_messages) >= 3:
            break

    parts.extend(reversed(recent_user_messages))
    return "\n".join(part for part in parts if part).strip()


def _resolve_active_context(
    *,
    older_history_summary: str,
    working_memory: WorkingMemory,
    cached_state: ContextStateData | None,
    resolved_user_preferences: list[str],
    resolved_project_constraints: list[str],
    recent_risks: list[str],
) -> tuple[CompactMemorySnapshot, str]:
    """基于当前 working memory 重建 active context，并吸收上一版基线。"""
    snapshot = build_active_context_snapshot(
        older_history_summary=older_history_summary,
        working_memory=working_memory,
        previous_snapshot=(
            cached_state.active_context_snapshot if cached_state is not None else None
        ),
        previous_active_context_summary=(
            cached_state.active_context_summary if cached_state is not None else ""
        ),
        resolved_user_preferences=resolved_user_preferences,
        resolved_project_constraints=resolved_project_constraints,
        recent_risks=recent_risks,
    )
    return snapshot, render_active_context_summary(snapshot)


def _build_preview_source(
    *,
    full_history: list[ChatMessage],
    session: SessionData,
    cached_state: ContextStateData | None,
    initial_keep_rounds: int,
) -> tuple[list[ChatMessage], str]:
    """构造压缩前预估用的 recent 消息和摘要基线。"""
    restored_recent_messages = _try_restore_recent_messages(
        full_history=full_history,
        cached_state=cached_state,
    )
    if restored_recent_messages is not None and cached_state is not None:
        return restored_recent_messages, cached_state.active_context_summary

    preview_window = select_history_window(
        history=full_history,
        keep_rounds=initial_keep_rounds,
    )
    preview_summary = build_older_history_summary(preview_window.older_messages)
    return preview_window.recent_messages, preview_summary


def _try_restore_recent_messages(
    *,
    full_history: list[ChatMessage],
    cached_state: ContextStateData | None,
) -> list[ChatMessage] | None:
    """
    尝试从 active context state 恢复 recent messages。

    思路和 MiniCode 类似：
    - 老历史保留为 compact 后的 active context
    - 新增消息只做增量追加
    """
    if cached_state is None:
        return None

    source_count = cached_state.source_message_count
    if source_count < 0 or source_count > len(full_history):
        return None

    source_history = full_history[:source_count]
    source_fingerprint = build_history_fingerprint(source_history)
    if source_fingerprint != cached_state.source_history_fingerprint:
        return None

    delta_messages = full_history[source_count:]
    restored = list(cached_state.compacted_messages)
    restored.extend(delta_messages)
    return restored


def _save_active_context_state(
    *,
    full_history: list[ChatMessage],
    session: SessionData,
    active_context_summary: str,
    active_context_snapshot: CompactMemorySnapshot,
    older_history_summary: str,
    resolved_user_preferences: list[str],
    resolved_project_constraints: list[str],
    recent_risks: list[str],
    compacted_messages: list[ChatMessage],
    preview_stats: ContextStats,
    final_stats: ContextStats,
    pipeline_history_entry: dict[str, object],
    cached_state: ContextStateData | None,
    compaction_level: int,
    last_microcompact_at: float,
    auto_compact_failure_count: int,
    auto_compact_suppressed_until: float,
) -> None:
    """把当前 compact 后的 active context 状态单独落盘。"""
    history_entries: list[dict[str, object]] = []
    if cached_state is not None:
        history_entries.extend(cached_state.compaction_history[-9:])
    history_entries.append(dict(pipeline_history_entry))

    state = ContextStateData(
        session_id=session.session_id,
        source_message_count=len(full_history),
        source_history_fingerprint=build_history_fingerprint(full_history),
        compacted_messages=list(compacted_messages),
        active_context_summary=active_context_summary,
        active_context_snapshot=dict(active_context_snapshot),
        older_history_summary=older_history_summary,
        resolved_user_preferences=list(resolved_user_preferences),
        resolved_project_constraints=list(resolved_project_constraints),
        recent_risks=list(recent_risks),
        compaction_level=compaction_level,
        auto_compact_failure_count=max(0, int(auto_compact_failure_count)),
        auto_compact_suppressed_until=max(
            0.0,
            float(auto_compact_suppressed_until or 0.0),
        ),
        # 这里保存的是“本轮结束后”的最近一次 microcompact 时间。
        # 下一轮如果直接命中 context_state，就能继续做时间节流判断。
        last_microcompact_at=max(0.0, float(last_microcompact_at or 0.0)),
        compaction_history=history_entries,
        last_token_stats=build_token_stats_snapshot(
            preview_stats=preview_stats,
            final_stats=final_stats,
        ),
    )
    # 同步到 session.extra，兼容仍从 extra 读取摘要基线的旧路径。
    session.extra["active_context_summary"] = active_context_summary
    session.extra["active_context_snapshot"] = dict(active_context_snapshot)
    session.extra["older_history_summary"] = older_history_summary
    save_context_state(session.workspace, state)


def _build_compacted_request(
    *,
    pipeline: ContextCompactorPipeline,
    source_recent_messages: list[ChatMessage],
    summary_source_messages: list[ChatMessage],
    policy: ContextPolicy,
    session: SessionData,
    working_memory: WorkingMemory,
    memory_pipeline: MemoryPipeline | None,
    active_context_summary: str,
    active_context_snapshot: CompactMemorySnapshot,
    tool_registry: ToolRegistry,
    usable_budget: int,
    fixed_overhead_tokens: int,
    base_system_prompt: str,
    user_profile_context: str,
    cached_state: ContextStateData | None,
    force_auto_compact: bool = False,
    analysis_mode: bool = False,
) -> tuple[ContextPipelineResult, str, ContextStats, list[ChatMessage]]:
    """统一构造一次 compact 后的请求消息，避免 runtime 内重复拼装。"""
    pipeline_result = pipeline.process_request(
        messages=source_recent_messages,
        summary_source_messages=summary_source_messages,
        max_recent_tool_results=policy.max_recent_tool_results,
        truncate_tool_result_chars=policy.truncate_tool_result_chars,
        workspace=session.workspace,
        usable_budget=usable_budget,
        fixed_overhead_tokens=fixed_overhead_tokens,
        auto_compact_summary=active_context_summary,
        auto_compact_snapshot=active_context_snapshot,
        force_auto_compact=force_auto_compact,
        pinned_tool_names=(_ANALYSIS_PINNED_TOOL_NAMES if analysis_mode else None),
        # 命中缓存状态时，把上一次 microcompact 时间透传进 pipeline，
        # 让轻量清理也具备跨轮次的连续性。
        last_microcompact_at=(
            cached_state.last_microcompact_at if cached_state is not None else 0.0
        ),
        auto_compact_failure_count=(
            cached_state.auto_compact_failure_count if cached_state is not None else 0
        ),
        auto_compact_suppressed_until=(
            cached_state.auto_compact_suppressed_until if cached_state is not None else 0.0
        ),
    )

    session_snapshot = SessionData(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        workspace=session.workspace,
        messages=list(pipeline_result.messages),
        extra=dict(session.extra),
    )

    memory_context = ""
    if memory_pipeline is not None:
        memory_result = memory_pipeline.build_prompt_context(
            user_input=working_memory.get_primary_user_intent(),
            session=session_snapshot,
            working_memory=working_memory,
            # memory 检索也切到 active context baseline，避免继续由 older_history_summary 主导。
            session_summary_override=active_context_summary,
            top_k=policy.memory_top_k,
            retrieval_top_k=policy.retrieval_top_k,
            max_memory_chars_per_item=policy.memory_item_chars,
        )
        memory_context = memory_result.prompt_context

    system_prompt = build_system_prompt(
        tool_registry=tool_registry,
        memory_context=memory_context,
        user_profile_context=user_profile_context,
    )
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
    messages.extend(pipeline_result.messages)

    final_stats = collect_context_stats(
        system_prompt=base_system_prompt,
        recent_messages=pipeline_result.messages,
        memory_context=memory_context,
        usable_budget=usable_budget,
    )
    return pipeline_result, memory_context, final_stats, messages
