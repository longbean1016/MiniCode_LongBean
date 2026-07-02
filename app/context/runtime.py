from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

"""运行时上下文装配层，负责拼接消息窗口、记忆注入和压缩结果。"""

from app.context.compact_memory import (
    CompactMemorySnapshot,
    build_active_context_snapshot,
    merge_active_context_snapshots,
    parse_active_context_summary,
    render_active_context_summary,
)
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
from app.context.state import (
    ContextStateData,
    build_history_fingerprint,
    build_token_stats_snapshot,
    load_context_state,
    merge_context_state_snapshot,
    save_context_state,
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
    active_context_snapshot: CompactMemorySnapshot
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
    cached_state = load_context_state(session.workspace, session.session_id)
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

    active_context_snapshot, active_context_summary = _resolve_active_context(
        older_history_summary=older_history_summary,
        cached_state=cached_state,
        resolved_user_preferences=resolved_user_preferences,
        resolved_project_constraints=resolved_project_constraints,
        recent_risks=recent_risks,
    )

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


def _resolve_active_context(
    *,
    older_history_summary: str,
    cached_state: ContextStateData | None,
    resolved_user_preferences: list[str],
    resolved_project_constraints: list[str],
    recent_risks: list[str],
) -> tuple[CompactMemorySnapshot, str]:
    """基于 memory snapshot 重建 active context，并吸收上一版基线。"""
    snapshot = build_active_context_snapshot(
        older_history_summary=older_history_summary,
        working_memory=None,
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
        # 这里保存的是"本轮结束后"的最近一次 microcompact 时间。
        # 下一轮如果直接命中 context_state，就能继续做时间节流判断。
        last_microcompact_at=max(0.0, float(last_microcompact_at or 0.0)),
        compaction_history=history_entries,
        last_token_stats=build_token_stats_snapshot(
            final_stats=final_stats,
        ),
    )
    # 同步到 session.extra，兼容仍从 extra 读取摘要基线的旧路径。
    session.extra["active_context_summary"] = active_context_summary
    session.extra["active_context_snapshot"] = dict(active_context_snapshot)
    session.extra["older_history_summary"] = older_history_summary
    save_context_state(session.workspace, state)


def persist_post_response_working_memory_state(
    *,
    session: SessionData,
    working_memory: object = None,
) -> None:
    """
    保存点 B：模型回复写入 WM 后，立即把增量快照合并进 context_state。

    保存点 A 负责请求前完整基线；这里只补当轮新产生的决策、风险、约束等 WM 增量，
    降低进程在两轮之间退出时丢失最新语义的概率。
    """
    if working_memory is None:
        return
    protected_snapshot = working_memory.build_protected_snapshot()
    if not protected_snapshot:
        return
    merge_context_state_snapshot(
        session.workspace,
        session.session_id,
        protected_snapshot,
    )


def _build_compacted_request(
    *,
    pipeline: ContextCompactorPipeline,
    source_recent_messages: list[ChatMessage],
    summary_source_messages: list[ChatMessage],
    policy: ContextPolicy,
    session: SessionData,
    active_context_summary: str,
    active_context_snapshot: CompactMemorySnapshot,
    tool_registry: ToolRegistry,
    usable_budget: int,
    fixed_overhead_tokens: int,
    base_system_prompt: str,
    user_profile_context: str,
    cached_state: ContextStateData | None,
    history_summarizer: OlderHistorySummarizer | None,
    force_auto_compact: bool = False,
    analysis_mode: bool = False,
) -> tuple[ContextPipelineResult, str, ContextStats, list[ChatMessage]]:
    """统一构造一次 compact 后的请求消息，避免 runtime 内重复拼装。"""
    pipeline_result = pipeline.process_request(
        messages=source_recent_messages,
        summary_source_messages=summary_source_messages,
        max_recent_tool_results=policy.max_recent_tool_results,
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
        # session/full compact 摘要优先尝试模型生成结构化结果，失败再退回规则摘要。
        semantic_summarizer=history_summarizer,
    )
    active_context_summary = (
        pipeline_result.resolved_active_context_summary or active_context_summary
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
    # 改用 MemoryStore 快照注入，不再走向量检索 pipeline
    memory_context = get_memory_store().get_prompt_context()

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


# ── 上下文压缩流水线（原 compactor_pipeline.py） ──

# 对标 Claude Code timeBasedMCConfig.keepRecent（默认保留最近 5 条 tool_result）
_MICROCOMPACT_KEEP_RECENT_TOOL_RESULTS = 5
_MICROCOMPACT_INTERVAL_SECONDS = 60 * 60
# 对标 Claude Code timeBasedMCConfig.gapThresholdMinutes（默认 60 分钟）
_MICROCOMPACT_GAP_THRESHOLD_MINUTES = 60


def _count_tool_rounds(messages: list[ChatMessage]) -> int:
    """统计有工具调用的轮数。

    一轮 = 两次 user 消息之间的所有消息。
    只统计至少包含一条 tool_result 的轮。
    """
    rounds = 0
    current_has_tools = False
    for msg in messages:
        role = str(msg.get("role", ""))
        if role == "user":
            if current_has_tools:
                rounds += 1
            current_has_tools = False
        elif role == "tool_result":
            current_has_tools = True
    if current_has_tools:
        rounds += 1
    return rounds


def _gap_since_last_assistant_minutes(
    messages: list[ChatMessage],
    now: float | None = None,
) -> float | None:
    """计算距离最后一条 assistant 消息已经过了多少分钟。

       对标 Claude Code evaluateTimeBasedTrigger() 的 gapMinutes 计算。
       使用 ChatMessage.created_at 作为消息时间戳。
       返回 None 表示消息列表中没有任何 assistant 消息。
    """
    if now is None:
        now = time.time()

    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            created_at = msg.get("created_at")
            if isinstance(created_at, (int, float)) and created_at > 0:
                return (now - created_at) / 60.0
            # 旧消息没有 created_at 字段，保守返回 0
            return 0.0
    return None


@dataclass(slots=True)
class ContextPipelineResult:
    """统一返回每轮请求前的上下文治理结果。"""

    messages: list[ChatMessage]
    compaction_result: CompactionResult
    auto_compact_result: AutoCompactResult
    steps_taken: list[str] = field(default_factory=list)
    compaction_history_entry: dict[str, Any] = field(default_factory=dict)
    resolved_active_context_snapshot: CompactMemorySnapshot = field(default_factory=dict)
    resolved_active_context_summary: str = ""
    last_microcompact_at: float = 0.0
    auto_compact_failure_count: int = 0
    auto_compact_suppressed_until: float = 0.0
    protected_recent_messages: int = 6


@dataclass(slots=True)
class LightweightCompactionConfig:
    """
    轻量治理阶段的配置。

    这层对应新版 MiniCode 里的请求前小步优化：
    - 先控制 tool_result 体积
    - 先去掉重复扫描噪声
    - 先压掉最近窗口里最容易膨胀的部分
    """

    max_recent_tool_results: int
    protected_recent_messages: int = 6


@dataclass(slots=True)
class MicrocompactState:
    """microcompact 的跨轮次状态。

       对标 Claude Code MicrocompactResult 的状态跟踪。
       keep_recent_tool_results 默认 5，按条数保留最近 tool_result。
    """

    last_time_based_compact: float = 0.0
    time_based_interval: float = _MICROCOMPACT_INTERVAL_SECONDS
    keep_recent_tool_results: int = _MICROCOMPACT_KEEP_RECENT_TOOL_RESULTS
    total_tokens_cleared: int = 0


@dataclass(slots=True)
class MicrocompactResult:
    """microcompact 阶段的结果。"""

    messages: list[ChatMessage]
    applied: bool = False
    cleared_count: int = 0
    tokens_freed_estimate: int = 0
    last_compact_at: float = 0.0
    # 记录 microcompact 决策细节，便于日志和 context_state 直接定位原因。
    reason: str = "not_evaluated"
    tool_round_count: int = 0
    keep_recent_tool_results: int = 0
    cooldown_remaining_seconds: float = 0.0


class LightweightContextPhase:
    """对应新版 MiniCode 的 tool budget / read dedup / recent cleanup 阶段。"""

    def run(
        self,
        *,
        messages: list[ChatMessage],
        config: LightweightCompactionConfig,
        usable_budget: int,
        fixed_overhead_tokens: int,
        pinned_tool_names: set[str] | None,
    ) -> tuple[CompactionResult, list[str]]:
        target_tokens = max(
            1,
            int(usable_budget * AUTO_COMPACT_TARGET_RATIO) - fixed_overhead_tokens,
        )
        result = compact_recent_messages(
            messages,
            pinned_tool_names=pinned_tool_names,
            target_tokens=target_tokens,
            protected_recent_messages=config.protected_recent_messages,
        )
        return result, ["tool_budget", "read_dedup", "recent_tool_cleanup"]


class MicrocompactEngine:
    """
    轻量 microcompact 引擎。

    行为上尽量贴近新版 MiniCode：
    - 不生成 assistant 摘要
    - 不破坏 tool_call / tool_result 协议
    - 只把较旧的 tool_result 正文替换成占位说明
    - 使用时间窗口做节流，避免每轮重复清理同一批历史结果
    """

    def __init__(self, state: MicrocompactState | None = None):
        self._state = state or MicrocompactState()

    @property
    def state(self) -> MicrocompactState:
        return self._state

    def run(
        self,
        *,
        messages: list[ChatMessage],
        pinned_tool_names: set[str] | None,
        semantic_summarizer: OlderHistorySummarizer | None = None,
    ) -> MicrocompactResult:
        """对标 Claude Code microcompactMessages → maybeTimeBasedMicrocompact：
           纯空闲时间触发，不依赖 usable_budget 或工具轮数。"""
        now = time.time()
        decision = self._evaluate(
            messages=messages,
            now=now,
        )
        if not bool(decision["should_apply"]):
            return MicrocompactResult(
                messages=list(messages),
                last_compact_at=self._state.last_time_based_compact,
                reason=str(decision["reason"]),
                tool_round_count=int(decision["tool_round_count"]),
                keep_recent_tool_results=int(decision["keep_recent_tool_results"]),
                cooldown_remaining_seconds=float(decision["cooldown_remaining_seconds"]),
            )

        compaction_result = microcompact_old_tool_results(
            messages,
            keep_recent_tool_results=self._state.keep_recent_tool_results,
            pinned_tool_names=pinned_tool_names,
            semantic_summarizer=semantic_summarizer,
        )
        if compaction_result.cleared_old_tool_results <= 0:
            return MicrocompactResult(
                messages=list(compaction_result.messages),
                last_compact_at=self._state.last_time_based_compact,
                                reason="no_old_tool_results",
                tool_round_count=int(decision["tool_round_count"]),
                keep_recent_tool_results=int(decision["keep_recent_tool_results"]),
            )

        self._state.last_time_based_compact = now
        self._state.total_tokens_cleared += compaction_result.tokens_freed_estimate
        return MicrocompactResult(
            messages=list(compaction_result.messages),
            applied=True,
            cleared_count=compaction_result.cleared_old_tool_results,
            tokens_freed_estimate=compaction_result.tokens_freed_estimate,
            last_compact_at=now,
            carried_tool_findings=list(compaction_result.carried_tool_findings),
            carried_open_issues=list(compaction_result.carried_open_issues),
            carried_key_decisions=list(compaction_result.carried_key_decisions),
            reason="applied",
            tool_round_count=int(decision["tool_round_count"]),
            keep_recent_tool_results=int(decision["keep_recent_tool_results"]),
        )

    def _should_apply(
        self,
        *,
        messages: list[ChatMessage],
        now: float,
    ) -> bool:
        """基于空闲时间判断是否需要触发 microcompact。

           对标 Claude Code evaluateTimeBasedTrigger()：
           唯一触发条件 — 最后一条 assistant 消息距今超过 60 分钟。
           注意：ChatMessage 暂无 timestamp 字段，gap 计算依赖 future 添加。
           当前回退策略：gap 不可用时不触发 microcompact。
        """
        gap_minutes = _gap_since_last_assistant_minutes(messages, now)
        if gap_minutes is None:
            return False
        return gap_minutes > _MICROCOMPACT_GAP_THRESHOLD_MINUTES

    def _evaluate(
        self,
        *,
        messages: list[ChatMessage],
        now: float,
    ) -> dict[str, float | int | str | bool]:
        """评估是否触发 microcompact。对标 Claude Code evaluateTimeBasedTrigger()。"""
        tool_rounds = _count_tool_rounds(messages)
        keep_recent = max(0, self._state.keep_recent_tool_results)
        gap_minutes = _gap_since_last_assistant_minutes(messages, now)

        if gap_minutes is None:
            return {
                "should_apply": False,
                "reason": "no_assistant_message",
                "tool_round_count": tool_rounds,
                "keep_recent_tool_results": keep_recent,
                "cooldown_remaining_seconds": 0.0,
            }

        if gap_minutes <= _MICROCOMPACT_GAP_THRESHOLD_MINUTES:
            return {
                "should_apply": False,
                "reason": "gap_too_small",
                "tool_round_count": tool_rounds,
                "keep_recent_tool_results": keep_recent,
                "cooldown_remaining_seconds": (_MICROCOMPACT_GAP_THRESHOLD_MINUTES - gap_minutes) * 60.0,
            }

        return {
            "should_apply": True,
            "reason": "ready",
            "tool_round_count": tool_rounds,
            "keep_recent_tool_results": keep_recent,
            "cooldown_remaining_seconds": 0.0,
        }


class ContextCompactor:
    """
    统一的请求前上下文编排器。

    这个类是对齐新版 MiniCode 结构的核心：
    1. 轻量治理阶段
    2. microcompact
    3. auto compact 调度

    但对外仍然产出当前项目已有的 ContextPipelineResult，
    这样 runtime、context_state、日志、测试都不需要重写。
    """

    def __init__(self):
        self._lightweight_phase = LightweightContextPhase()

    def process_request(
        self,
        *,
        messages: list[ChatMessage],
        summary_source_messages: list[ChatMessage] | None,
        lightweight_config: LightweightCompactionConfig,
        workspace: str,
        usable_budget: int,
        fixed_overhead_tokens: int,
        auto_compact_summary: str,
        auto_compact_snapshot: CompactMemorySnapshot | None,
        force_auto_compact: bool,
        pinned_tool_names: set[str] | None,
        microcompact_state: MicrocompactState,
        auto_compact_failure_count: int = 0,
        auto_compact_suppressed_until: float = 0.0,
        semantic_summarizer: OlderHistorySummarizer | None = None,
        model_name: str = "",
    ) -> ContextPipelineResult:
        auto_compact = AutoCompactDispatcher(
            config=AutoCompactDispatcherConfig(),
            failure_count=auto_compact_failure_count,
            suppressed_until=auto_compact_suppressed_until,
        )
        compaction_result, steps_taken = self._lightweight_phase.run(
            messages=messages,
            config=lightweight_config,
            usable_budget=usable_budget,
            fixed_overhead_tokens=fixed_overhead_tokens,
            pinned_tool_names=pinned_tool_names,
        )

        microcompact_engine = MicrocompactEngine(microcompact_state)
        microcompact_result = microcompact_engine.run(
            messages=compaction_result.messages,
            pinned_tool_names=pinned_tool_names,
            semantic_summarizer=semantic_summarizer,
        )
        if microcompact_result.applied:
            overlay = CompactionResult(
                messages=list(microcompact_result.messages),
                cleared_old_tool_results=microcompact_result.cleared_count,
                tokens_freed_estimate=microcompact_result.tokens_freed_estimate,
                carried_tool_findings=list(microcompact_result.carried_tool_findings),
                carried_open_issues=list(microcompact_result.carried_open_issues),
                carried_key_decisions=list(microcompact_result.carried_key_decisions),
            )
            _merge_compaction_result(base=compaction_result, overlay=overlay)
            steps_taken.append("microcompact")

        recent_tokens_after_compaction = estimate_messages_tokens(compaction_result.messages)
        auto_compact_result = auto_compact.dispatch(
            messages=compaction_result.messages,
            usable_budget=usable_budget,
            summary_base=auto_compact_summary,
            summary_snapshot=auto_compact_snapshot,
            summary_source_messages=summary_source_messages or messages,
            fixed_overhead_tokens=fixed_overhead_tokens,
            force_full=force_auto_compact,
            semantic_summarizer=semantic_summarizer,
            model_name=model_name,
        )
        if auto_compact_result.applied:
            steps_taken.append(f"auto_compact:{auto_compact_result.strategy}")

        output_messages = (
            auto_compact_result.messages
            if auto_compact_result.applied
            else compaction_result.messages
        )
        normalized_messages = normalize_tool_call_pairs(output_messages)
        resolved_snapshot, resolved_context = _resolve_active_context_outputs(
            auto_compact_summary=auto_compact_summary,
            auto_compact_snapshot=auto_compact_snapshot,
            auto_compact_result=auto_compact_result,
        )

        history_entry = {
            "truncated_tool_results": compaction_result.truncated_tool_results,
            "cleared_old_tool_results": compaction_result.cleared_old_tool_results,
            "deduped_read_results": compaction_result.deduped_read_results,
            "semantic_compacted_pairs": compaction_result.semantic_compacted_pairs,
            "dropped_progress_messages": compaction_result.dropped_progress_messages,
            "priority_dropped_messages": compaction_result.priority_dropped_messages,
            "recent_tokens_after_compaction": recent_tokens_after_compaction,
            "tokens_freed_estimate": (
                compaction_result.tokens_freed_estimate
                + auto_compact_result.tokens_freed_estimate
            ),
            "microcompact_applied": microcompact_result.applied,
            "microcompact_reason": microcompact_result.reason,
            "microcompact_tool_rounds": microcompact_result.tool_round_count,
            "microcompact_keep_recent_rounds": microcompact_result.keep_recent_tool_results,
            "microcompact_cooldown_remaining_seconds": (
                microcompact_result.cooldown_remaining_seconds
            ),
            "last_microcompact_at": microcompact_result.last_compact_at,
            "auto_compact_applied": auto_compact_result.applied,
            "auto_compact_strategy": auto_compact_result.strategy,
            "auto_compact_tokens_before": auto_compact_result.tokens_before,
            "auto_compact_tokens_after": auto_compact_result.tokens_after,
            "steps_taken": list(steps_taken),
        }
        return ContextPipelineResult(
            messages=normalized_messages,
            compaction_result=compaction_result,
            auto_compact_result=auto_compact_result,
            steps_taken=steps_taken,
            compaction_history_entry=history_entry,
            resolved_active_context_snapshot=resolved_snapshot,
            resolved_active_context_summary=resolved_context,
            last_microcompact_at=microcompact_result.last_compact_at,
            auto_compact_failure_count=auto_compact_result.failure_count,
            auto_compact_suppressed_until=auto_compact_result.suppressed_until,
        )


class ContextCompactorPipeline:
    """
    当前项目的兼容入口。

    对外接口保持不变，但内部已经切成更接近新版 MiniCode 的结构：
    `LightweightContextPhase -> MicrocompactEngine -> AutoCompactDispatcher -> ContextCompactor`
    """

    def __init__(self):
        self._compactor = ContextCompactor()

    def process_request(
        self,
        *,
        messages: list[ChatMessage],
        summary_source_messages: list[ChatMessage] | None = None,
        max_recent_tool_results: int,
        workspace: str,
        usable_budget: int,
        fixed_overhead_tokens: int,
        auto_compact_summary: str,
        auto_compact_snapshot: CompactMemorySnapshot | None = None,
        force_auto_compact: bool = False,
        pinned_tool_names: set[str] | None = None,
        last_microcompact_at: float = 0.0,
        auto_compact_failure_count: int = 0,
        auto_compact_suppressed_until: float = 0.0,
        semantic_summarizer: OlderHistorySummarizer | None = None,
        model_name: str = "",
    ) -> ContextPipelineResult:
        return self._compactor.process_request(
            messages=messages,
            summary_source_messages=summary_source_messages,
            lightweight_config=LightweightCompactionConfig(
                max_recent_tool_results=max_recent_tool_results,
            ),
            workspace=workspace,
            usable_budget=usable_budget,
            fixed_overhead_tokens=fixed_overhead_tokens,
            auto_compact_summary=auto_compact_summary,
            auto_compact_snapshot=auto_compact_snapshot,
            force_auto_compact=force_auto_compact,
            pinned_tool_names=pinned_tool_names,
            microcompact_state=MicrocompactState(
                last_time_based_compact=max(0.0, float(last_microcompact_at or 0.0)),
                keep_recent_tool_results=_MICROCOMPACT_KEEP_RECENT_TOOL_RESULTS,
            ),
            auto_compact_failure_count=auto_compact_failure_count,
            auto_compact_suppressed_until=auto_compact_suppressed_until,
            semantic_summarizer=semantic_summarizer,
            model_name=model_name,
        )


def _resolve_active_context_outputs(
    *,
    auto_compact_summary: str,
    auto_compact_snapshot: CompactMemorySnapshot | None,
    auto_compact_result: AutoCompactResult,
) -> tuple[CompactMemorySnapshot, str]:
    """统一解析 active context 的最终输出，避免散落在不同阶段重复拼接。"""
    resolved_snapshot = auto_compact_snapshot or parse_active_context_summary(
        auto_compact_summary
    )
    resolved_context = auto_compact_summary.strip()
    if auto_compact_result.summary_snapshot:
        resolved_snapshot = auto_compact_result.summary_snapshot
        resolved_context = (
            auto_compact_result.summary_text.strip()
            or render_active_context_summary(resolved_snapshot)
        )
    elif resolved_snapshot and not resolved_context:
        resolved_context = render_active_context_summary(resolved_snapshot)
    return resolved_snapshot, resolved_context


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
