from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

"""运行时上下文装配层，负责拼接消息窗口、记忆注入和压缩结果。"""

from app.context.auto_compact import (
    AUTO_COMPACT_TRIGGER_RATIO,
    AUTO_COMPACT_TARGET_RATIO,
    AutoCompactDispatcher,
    AutoCompactDispatcherConfig,
    AutoCompactResult,
)
from app.context.compactor import (
    CompactionResult,
    compact_recent_messages,
    microcompact_old_tool_results,
)
from app.context.message_safety import normalize_tool_call_pairs
from app.context.signal_resolver import (
    resolve_project_constraints,
    resolve_recent_risks,
    resolve_user_preferences,
)
from app.context.manager import (
    DEFAULT_USABLE_CONTEXT_BUDGET,
    ContextPolicy,
    ContextStats,
    collect_context_stats,
    decide_context_policy,
    estimate_messages_tokens,
)
from app.context.history_summarizer import OlderHistorySummarizer
from app.context.history_summarizer import HistoryWindow, build_older_history_summary, select_history_window
from app.memory.memory_store import MemoryStore
from app.memory.memory_tool import get_memory_store
from app.agent.prompt import build_system_prompt
from app.state.session import SessionData
from app.agent.tooling import ToolRegistry
from app.types import ChatMessage
from app.state.user_profile import ResolvedUserPolicy, UserPolicyRule, load_user_profile

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
# 分析模式下优先保留上下文结果的工具（对齐新工具集）
_ANALYSIS_PINNED_TOOL_NAMES = {
    "read_file",
    "grep_files",
    "glob_files",
}


@dataclass(slots=True)
class PreparedAgentContext:
    """保存一次模型请求前已经准备好的上下文结果。"""

    messages: list[ChatMessage]
    policy: ContextPolicy
    stats: ContextStats
    active_context_summary: str
    active_context_snapshot: dict
    older_history_summary: str
    resolved_user_preferences: list[str]
    resolved_user_policy: ResolvedUserPolicy
    active_user_rules: list[UserPolicyRule]
    resolved_project_constraints: list[str]
    recent_risks: list[str]
    history_window: HistoryWindow
    compaction_result: CompactionResult
    compaction_history_entry: dict[str, object]
    pipeline_steps: list[str]
    memory_context: str
    user_profile_context: str


def prepare_agent_context(
    *,
    full_history: list[ChatMessage],
    session: SessionData,
    tool_registry: ToolRegistry,
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
    cached_state = None  # context_state 已删除，不再从本地缓存恢复
    resolved_user_preferences = resolve_user_preferences(workspace=session.workspace)
    resolved_user_policy = _resolve_user_policy(session.workspace)
    active_user_rules = resolved_user_policy.active_rules_for(
        _build_user_policy_task_context(
            full_history=full_history,
        )
    )
    user_profile_context = resolved_user_policy.to_prompt_section(active_user_rules)
    base_system_prompt = build_system_prompt(
        tool_registry=tool_registry,
        memory_context="",
        user_profile_context=user_profile_context,
    )

    policy = decide_context_policy(analysis_mode=analysis_mode)
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
        memory_snapshot=None,
    )
    recent_risks = resolve_recent_risks(
        cached_risks=cached_state.recent_risks if cached_state is not None else [],
    )

    active_context_snapshot, active_context_summary = {}, ""

    # 固定 system + memory 开销估计（原基于 preview_stats 计算，简化为常数）
    fixed_overhead_tokens = int(usable_budget * 0.12)
    pipeline = ContextCompactorPipeline()
    # pipeline 负责真正把"摘要、memory、tool_result 裁剪、系统提示词"拼成最终请求。
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
        active_context_summary=active_context_summary,
        active_context_snapshot=active_context_snapshot,
        tool_registry=tool_registry,
        usable_budget=usable_budget,
        fixed_overhead_tokens=fixed_overhead_tokens,
        base_system_prompt=base_system_prompt,
        user_profile_context=user_profile_context,
        # 把缓存状态直接传给压缩流水线，让 microcompact 节流能接上旧轮次。
        cached_state=cached_state,
        history_summarizer=history_summarizer,
        analysis_mode=analysis_mode,
    )
    active_context_snapshot = (
        pipeline_result.resolved_active_context_snapshot or active_context_snapshot
    )
    active_context_summary = (
        pipeline_result.resolved_active_context_summary or active_context_summary
    )

    # 如果 memory 注入之后上下文还是高压，就再强制走一次 auto compact。
    # 这一步是第二道保险，避免"摘要已经做了，但 memory 一加回来又超压"。
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
            history_summarizer=history_summarizer,
            analysis_mode=analysis_mode,
        )
        active_context_snapshot = (
            pipeline_result.resolved_active_context_snapshot or active_context_snapshot
        )
        active_context_summary = (
            pipeline_result.resolved_active_context_summary or active_context_summary
        )

    _save_active_context_state()

    return PreparedAgentContext(
        messages=messages,
        policy=policy,
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
        compaction_history_entry=dict(pipeline_result.compaction_history_entry),
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
    working_memory: object = None,
) -> str:
    """构造预估阶段的轻量 memory_context，不触发长期记忆检索。"""
    sections: list[str] = []
    if active_context_summary.strip():
        # 预估阶段也尽量复用正式注入时的结构，
        # 让 token 预估更接近真实请求的上下文形态。
        sections.append("## 当前会话压缩基线\n" + active_context_summary.strip())

    try:
        working_memory_text = _build_preview_working_memory_brief(working_memory)
    except Exception:
        working_memory_text = ""

    if working_memory_text:
        # preview 里不做长期记忆检索，只保留轻量 working memory 补充位。
        sections.append("## 当前工作记忆\n" + working_memory_text)

    return "\n".join(sections).strip()


def _build_preview_working_memory_brief(working_memory: object) -> str:
    """预估阶段只保留少量高价值 working memory，避免预估值过度膨胀。"""
    if working_memory is None:
        return ""
    sections: list[str] = []
    slot_specs = (
        # 这里和 MemoryReadPipeline 保持同一组核心槽位，
        # 避免 preview 与真实注入阶段关注点错位。
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
    working_memory: object = None,
) -> str:
    """提取当前任务文本，用于筛选命中的路径规则。"""
    parts: list[str] = []
    if working_memory is None:
        return ""
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


def _save_active_context_state(**kwargs) -> None:
    """context_state 已删除，不再落盘。"""
    pass


def _resolve_active_context_outputs(
    *,
    auto_compact_summary: str,
    auto_compact_snapshot: dict | None,
    auto_compact_result: AutoCompactResult,
) -> tuple[dict, str]:
    """返回空的 active context 输出（compact_memory 已删除）。"""
    return {}, ""


def _merge_compaction_result(*, base: CompactionResult, overlay: CompactionResult) -> None:
    """
    把额外阶段的统计并回主结果。

    这里继续沿用当前项目已有的 CompactionResult，
    避免 runtime、日志、测试、state 结构一起被迫改动。
    """

    base.messages = list(overlay.messages)
    base.truncated_tool_results += overlay.truncated_tool_results
    base.cleared_old_tool_results += overlay.cleared_old_tool_results
    base.deduped_read_results += overlay.deduped_read_results
    base.semantic_compacted_pairs += overlay.semantic_compacted_pairs
    base.dropped_progress_messages += overlay.dropped_progress_messages
    base.priority_dropped_messages += overlay.priority_dropped_messages
    base.tokens_freed_estimate += overlay.tokens_freed_estimate
