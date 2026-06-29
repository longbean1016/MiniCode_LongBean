from __future__ import annotations

import re
from dataclasses import dataclass

"""压缩期记忆快照模块，负责提炼上下文压缩前后的核心记忆语义。"""

from app.context.manager import estimate_tokens
from app.types import ChatMessage
from app.state.working_memory import WorkingMemory, WorkingMemoryEntry

# compact memory 是跨轮次续带的结构化基线，预算不能低到只够放 1-2 条句子。
# 这里参考 minicode 的分层摘要 / working memory 量级，给到更合理的中等预算。
COMPACT_MEMORY_MAX_TOKENS = 2000
COMPACT_MEMORY_SECTION_ENTRY_LIMIT = 6
_CARRY_FORWARD_BLOCKED_SECTIONS = {"项目约束", "当前任务", "用户意图"}
_CARRY_FORWARD_REJECT_PHRASES = (
    "本次",
    "这次",
    "本轮",
    "验收测试",
    "工具调用",
    "完成后",
    "立即停止",
    "不要继续探索",
    "写文件",
    "写入 tmp",
    "tmp/",
    "先读取",
    "最后只回复",
    "最多允许",
)
_SECTION_KEY_TO_TITLE = {
    "preferences": "用户偏好",
    "stable_constraints": "项目约束",
    "active_tasks": "当前任务",
    "decisions": "关键决策",
    "open_issues": "未解决问题（最近风险）",
    "tool_findings": "关键工具发现（上次压缩延续）",
    "history_summary": "旧对话摘要",
}
_TITLE_TO_SECTION_KEY = {
    "用户偏好": "preferences",
    "项目约束": "stable_constraints",
    "当前任务": "active_tasks",
    "关键决策": "decisions",
    "未解决问题": "open_issues",
    "未解决问题（最近风险）": "open_issues",
    "最近风险": "open_issues",
    "关键工具发现": "tool_findings",
    "旧对话摘要": "history_summary",
    "上次压缩延续": "tool_findings",
    "稳定事实": "tool_findings",
}
_SNAPSHOT_SECTION_SPECS = (
    ("decisions", 4, 160, 128),
    ("open_issues", 3, 100, 54),
    ("tool_findings", 4, 110, 50),
    ("active_tasks", 2, 80, 34),
    ("preferences", 2, 90, 42),
    ("stable_constraints", 3, 100, 54),
    ("history_summary", 1, 140, 40),
)
_DEFAULT_RENDER_SECTION_SPECS = (
    ("decisions", 4, 160, 128),
    ("open_issues", 3, 100, 54),
    ("tool_findings", 2, 110, 50),
    ("active_tasks", 2, 80, 34),
    ("preferences", 2, 90, 42),
    ("stable_constraints", 3, 100, 54),
    ("history_summary", 1, 140, 40),
)
_FULL_COMPACT_RENDER_SECTION_SPECS = (
    ("active_tasks", 1, 72, 42),
    ("open_issues", 2, 90, 72),
    ("decisions", 2, 96, 84),
    ("tool_findings", 2, 100, 110),
    ("stable_constraints", 1, 90, 42),
)
_NOISE_PREFIXES = ("│", "├", "└", "dir ", "file ")
_FOCUS_STOP_TOKENS = {
    "当前",
    "继续",
    "问题",
    "目标",
    "风险",
    "主题",
    "结论",
    "分析",
    "需要",
    "应该",
    "优先",
    "保住",
    "语义",
}
_MARKDOWN_TABLE_RE = re.compile(r"^\|.+\|$")
_CODE_LIKE_PREFIXES = ("def ", "class ", "return ", "import ", "from ", "if ", "for ", "while ")
_TOOL_RESULT_META_PREFIXES = (
    "FILE:",
    "OFFSET:",
    "END:",
    "TOTAL_CHARS:",
    "TRUNCATED:",
    "PATH:",
    "TOOL:",
    "路径:",
    "工具:",
    "SEARCH_ROOT:",
    "PATTERN:",
    "ROOT:",
    "TOTAL_ENTRIES:",
    "RETURNED_ENTRIES:",
    "LOCALROOT:",
)
_TOOL_RESULT_SKIP_TOKENS = (
    "--- preview",
    "--- 预览",
    "省略",
    "omitted",
    "output truncated",
    "已落盘",
    "旧工具结果已省略",
    "原始字符数",
    "补充说明",
    "filler block",
    "alpha filler",
    "beta filler",
    "delta filler",
    "gamma filler",
    "offset=",
    "limit=",
    "read_repeat_blocked",
    "read_policy_blocked",
    "同一轮里已经读取过相同区间",
    "目录创建成功",
    "文件写入成功",
)
_SEMANTIC_HINT_TOKENS = (
    "压缩",
    "上下文",
    "recent window",
    "tool_result",
    "memory",
    "working memory",
    "constraint",
    "constraints",
    "decision",
    "risk",
    "issue",
    "error",
    "结论",
    "发现",
    "风险",
    "原因",
    "建议",
    "需要",
    "应该",
    "优先",
    "避免",
    "问题",
    "策略",
    "验证",
    "保留",
    "分离",
    "污染",
)

CompactMemorySnapshot = dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class _SectionSpec:
    """定义 compact memory 各个层的优先级和预算。"""

    title: str
    entry_types: tuple[str, ...]
    max_entries: int
    max_item_chars: int
    max_section_tokens: int


_SECTION_SPECS = (
    _SectionSpec("用户偏好", ("user_preference",), 2, 90, 48),
    _SectionSpec("项目约束", ("project_constraint",), 3, 100, 54),
    _SectionSpec("最近风险", ("recent_risk", "error_context"), 3, 100, 54),
    _SectionSpec("当前任务", ("active_task",), 2, 80, 34),
    _SectionSpec("关键决策", ("key_decision",), 4, 160, 128),
    _SectionSpec("用户意图", ("user_intent",), 1, 90, 22),
)


def build_active_context_summary(
    *,
    older_history_summary: str,
    working_memory: WorkingMemory | None = None,
    previous_active_context_summary: str = "",
    previous_snapshot: CompactMemorySnapshot | None = None,
    resolved_user_preferences: list[str] | None = None,
    resolved_project_constraints: list[str] | None = None,
    recent_risks: list[str] | None = None,
) -> str:
    """
    构造“活动上下文摘要”。

    这是新版链路里真正给下一轮请求复用的主摘要基线，不再把旧 compact memory
    视为主接口；旧接口只保留兼容包装。
    """
    snapshot = build_active_context_snapshot(
        older_history_summary=older_history_summary,
        working_memory=working_memory,
        previous_snapshot=previous_snapshot,
        previous_active_context_summary=previous_active_context_summary,
        resolved_user_preferences=resolved_user_preferences,
        resolved_project_constraints=resolved_project_constraints,
        recent_risks=recent_risks,
    )
    return render_active_context_summary(snapshot)


def build_active_context_snapshot(
    *,
    older_history_summary: str,
    working_memory: WorkingMemory | None = None,
    previous_snapshot: CompactMemorySnapshot | None = None,
    previous_active_context_summary: str = "",
    resolved_user_preferences: list[str] | None = None,
    resolved_project_constraints: list[str] | None = None,
    recent_risks: list[str] | None = None,
) -> CompactMemorySnapshot:
    """
    构造结构化活动上下文快照。

    这里沿用新版 MiniCode 思路：
    1. 优先复用结构化快照，而不是整段自由文本回灌。
    2. older_history_summary 只作为低优先级原料，不再主导最终注入基线。
    3. 只有兼容旧数据时，才从 legacy active context 摘要文本 解析少量可续带内容。
    """
    carry_snapshot = _normalize_snapshot(previous_snapshot)
    if not carry_snapshot:
        carry_snapshot = _normalize_snapshot(
            _snapshot_from_previous_context(previous_active_context_summary)
        )
    protected_snapshot: CompactMemorySnapshot = {}
    if working_memory is not None:
        protected_snapshot = _normalize_snapshot(working_memory.build_protected_snapshot())
    snapshot: CompactMemorySnapshot = {}

    snapshot["preferences"] = _merge_snapshot_lines(
        primary=(
            list(resolved_user_preferences or [])
            or protected_snapshot.get("preferences", [])
        ),
        secondary=carry_snapshot.get("preferences", []),
        max_entries=2,
        max_item_chars=90,
    )
    snapshot["stable_constraints"] = _merge_snapshot_lines(
        primary=(
            list(resolved_project_constraints or [])
            or protected_snapshot.get("stable_constraints", [])
        ),
        secondary=carry_snapshot.get("stable_constraints", []),
        max_entries=3,
        max_item_chars=100,
    )
    snapshot["active_tasks"] = _merge_snapshot_lines(
        primary=protected_snapshot.get("active_tasks", []),
        secondary=[],
        max_entries=2,
        max_item_chars=80,
    )
    snapshot["decisions"] = _merge_snapshot_lines(
        primary=protected_snapshot.get("decisions", []),
        secondary=[],
        max_entries=4,
        max_item_chars=160,
    )

    issue_lines = _merge_snapshot_lines(
        primary=list(recent_risks or []),
        secondary=[],
        max_entries=3,
        max_item_chars=100,
    )
    if not issue_lines:
        issue_lines = _merge_snapshot_lines(
            primary=protected_snapshot.get("open_issues", []),
            secondary=[],
            max_entries=3,
            max_item_chars=100,
        )
    snapshot["open_issues"] = issue_lines

    snapshot["tool_findings"] = _merge_ranked_snapshot_lines(
        primary=(
            protected_snapshot.get("tool_findings", [])
            or _collect_tool_findings(working_memory)
        ),
        secondary=carry_snapshot.get("tool_findings", []),
        max_entries=2,
        max_item_chars=110,
        priority_fn=_tool_finding_priority,
    )

    normalized_summary = older_history_summary.strip()
    if normalized_summary:
        snapshot["history_summary"] = [_shorten(normalized_summary, 140)]
    elif carry_snapshot.get("history_summary"):
        snapshot["history_summary"] = carry_snapshot["history_summary"][:1]

    return {key: lines for key, lines in snapshot.items() if lines}


def parse_active_context_summary(text: str) -> CompactMemorySnapshot:
    """把活动上下文摘要解析回结构化快照，供恢复和全量压缩复用。"""
    return _normalize_snapshot(_snapshot_from_previous_context(text))


def merge_active_context_snapshots(
    *,
    base_snapshot: CompactMemorySnapshot | None,
    overlay_snapshot: CompactMemorySnapshot | None,
) -> CompactMemorySnapshot:
    """把事件快照叠加到稳定基线上，仍按各槽位预算做裁剪。"""
    base = _normalize_snapshot(base_snapshot or {})
    overlay = _normalize_snapshot(overlay_snapshot or {})
    merged: CompactMemorySnapshot = {}

    for key, max_entries, max_item_chars, _ in _SNAPSHOT_SECTION_SPECS:
        merge_fn = _merge_ranked_snapshot_lines if key == "tool_findings" else _merge_snapshot_lines
        merge_kwargs = {
            "primary": overlay.get(key, []),
            "secondary": base.get(key, []),
            "max_entries": max_entries,
            "max_item_chars": max_item_chars,
        }
        if key == "tool_findings":
            merge_kwargs["priority_fn"] = _tool_finding_priority
        merged_lines = merge_fn(**merge_kwargs)
        if merged_lines:
            merged[key] = merged_lines
    return merged


def prioritize_snapshot_for_current_focus(
    snapshot: CompactMemorySnapshot,
    *,
    focus_lines: list[str],
    drop_unaligned_tool_findings: bool = False,
) -> CompactMemorySnapshot:
    """按当前主题重排快照，优先保留与当前主线对齐的决策/风险/工具发现。"""
    normalized_snapshot = _normalize_snapshot(snapshot)
    # focus_lines 不是直接拿来展示，而是先拆成可匹配的主题词，
    # 后面用这些主题词给 decisions / open_issues / tool_findings 重新排序。
    focus_terms = _extract_focus_terms(focus_lines)
    if not focus_terms:
        return normalized_snapshot

    prioritized = dict(normalized_snapshot)
    for key in ("decisions", "open_issues", "tool_findings"):
        lines = list(prioritized.get(key, []))
        if not lines:
            continue
        base_priority_fn = _tool_finding_priority if key == "tool_findings" else _generic_focus_priority
        reranked = _rerank_lines_by_focus(
            lines,
            focus_terms=focus_terms,
            base_priority_fn=base_priority_fn,
        )
        if key == "tool_findings" and drop_unaligned_tool_findings:
            # tool_findings 最容易把旧主题噪音带进下一轮，所以在 full compact 下允许更激进：
            # 只保留和当前 focus 至少有一点对齐的工具发现。
            aligned = [
                line for line in reranked
                if _focus_alignment_score(line, focus_terms) > 0
            ]
            if aligned:
                reranked = aligned
        prioritized[key] = reranked
    return prioritized


def build_active_context_event_snapshot(
    *,
    removed_messages: list[ChatMessage],
) -> CompactMemorySnapshot:
    """按轮次事件提取压缩快照，避免按消息片段自由文本拼接。"""
    snapshot: CompactMemorySnapshot = {}
    active_tasks: list[str] = []
    decisions: list[str] = []
    open_issues: list[str] = []
    tool_calls: list[str] = []
    tool_findings: list[str] = []

    for message in removed_messages:
        role = str(message.get("role", "")).strip()
        raw_content = str(message.get("content", "")).strip()
        content = " ".join(raw_content.split())
        if role == "user" and content and not _looks_like_structured_noise(content):
            active_tasks.append(_shorten(content, 60))
            continue
        if role == "assistant" and content and not _looks_like_structured_noise(content):
            decisions.extend(_extract_assistant_decision_points(raw_content))
            continue
        if role == "assistant_tool_call":
            tool_name = str(message.get("tool_name", "")).strip() or "unknown"
            tool_calls.append(tool_name)
            continue
        if role != "tool_result" or not content:
            continue

        tool_name = str(message.get("tool_name", "")).strip() or "unknown"
        findings = _extract_tool_result_findings(
            tool_name=tool_name,
            raw_content=raw_content,
        )
        if bool(message.get("is_error")):
            if findings:
                open_issues.extend(findings)
            else:
                preview = _shorten(content, 70)
                open_issues.append(f"{tool_name}：{preview}")
        else:
            if findings:
                tool_findings.extend(findings)
            else:
                preview = _shorten(content, 70)
                if not _looks_like_low_value_tool_finding(preview):
                    tool_findings.append(f"{tool_name}：{preview}")

    if active_tasks:
        snapshot["active_tasks"] = _dedupe_lines(active_tasks)[:2]
    if decisions:
        snapshot["decisions"] = _dedupe_lines(decisions)[:4]
    if open_issues:
        snapshot["open_issues"] = _dedupe_lines(open_issues)[:3]
    if tool_findings:
        snapshot["tool_findings"] = _merge_ranked_snapshot_lines(
            primary=tool_findings,
            secondary=[],
            max_entries=4,
            max_item_chars=100,
            priority_fn=_tool_finding_priority,
        )
    elif tool_calls:
        snapshot["tool_findings"] = _dedupe_lines(
            [f"调用工具：{tool_name}" for tool_name in tool_calls]
        )[:2]
    return snapshot


def render_active_context_summary(
    snapshot: CompactMemorySnapshot,
    *,
    section_specs: tuple[tuple[str, int, int, int], ...] = _DEFAULT_RENDER_SECTION_SPECS,
    max_tokens: int = COMPACT_MEMORY_MAX_TOKENS,
) -> str:
    """
    把结构化快照按预算渲染成活动上下文摘要。

    这份文本会直接作为下一轮请求的基线，因此优先保留决策、风险、工具结论，
    而不是把整段 older_history_summary 原样塞回去。
    """
    normalized_snapshot = _normalize_snapshot(snapshot)
    lines: list[str] = ["结构化压缩记忆", "压缩记忆基线"]

    for key, max_entries, max_item_chars, max_section_tokens in section_specs:
        section_lines = _build_snapshot_section(
            key=key,
            values=normalized_snapshot.get(key, []),
            max_entries=max_entries,
            max_item_chars=max_item_chars,
            max_section_tokens=max_section_tokens,
        )
        if not section_lines:
            continue
        if not _append_with_global_budget(lines, section_lines, max_tokens=max_tokens):
            break

    return "\n".join(lines).strip()


def render_full_active_context_summary(snapshot: CompactMemorySnapshot) -> str:
    """full compact 场景下优先渲染语义核心，避免任务/偏好占掉摘要预算。"""
    return render_active_context_summary(
        snapshot,
        section_specs=_FULL_COMPACT_RENDER_SECTION_SPECS,
    )


def _build_snapshot_section(
    *,
    key: str,
    values: list[str],
    max_entries: int,
    max_item_chars: int,
    max_section_tokens: int,
) -> list[str]:
    """按结构化字段构造单个 section。"""
    if not values:
        return []

    title = _SECTION_KEY_TO_TITLE.get(key, key)
    lines = [f"## {title}"]
    kept_count = 0

    for content in values:
        if kept_count >= max_entries:
            break
        candidate_line = f"- {_shorten(content, max_item_chars)}"
        candidate_lines = [*lines, candidate_line]
        if estimate_tokens("\n".join(candidate_lines)) > max_section_tokens:
            if kept_count == 0:
                lines.append(candidate_line)
            break
        lines.append(candidate_line)
        kept_count += 1

    return lines if len(lines) > 1 else []


def _build_structured_section(
    *,
    working_memory: WorkingMemory | None = None,
    title: str,
    entry_types: tuple[str, ...],
    max_entries: int,
    max_item_chars: int,
    max_section_tokens: int,
    explicit_lines: list[str],
) -> list[str]:
    """按类型收集高价值条目，并在 section 内再做一次预算裁剪。"""
    entries = _collect_ranked_entries(
        working_memory=working_memory,
        entry_types=entry_types,
    ) if not explicit_lines else []
    if not explicit_lines and not entries:
        return []

    lines = [f"## {title}"]
    kept_count = 0
    candidate_sources = (
        [str(item).strip() for item in explicit_lines if str(item).strip()]
        if explicit_lines
        else [entry.content for entry in entries]
    )
    for content in candidate_sources:
        if kept_count >= max_entries:
            break

        candidate_line = f"- {_shorten(content, max_item_chars)}"
        candidate_lines = [*lines, candidate_line]
        if estimate_tokens("\n".join(candidate_lines)) > max_section_tokens:
            if kept_count == 0:
                lines.append(candidate_line)
            break

        lines.append(candidate_line)
        kept_count += 1

    return lines if len(lines) > 1 else []


def _collect_ranked_entries(
    *,
    working_memory: WorkingMemory | None = None,
    entry_types: tuple[str, ...],
) -> list[WorkingMemoryEntry]:
    """对不同类型的 working memory 条目做去重和优先级排序。"""
    raw_entries: list[WorkingMemoryEntry] = []
    for entry_type in entry_types:
        raw_entries.extend(working_memory.get_entries_by_type(entry_type))

    deduped = _dedupe_entries(raw_entries)
    deduped.sort(
        key=lambda entry: (
            _entry_type_priority(entry.entry_type),
            entry.importance,
            entry.created_at,
        ),
        reverse=True,
    )
    return deduped[:COMPACT_MEMORY_SECTION_ENTRY_LIMIT]


def _collect_snapshot_entries(
    *,
    working_memory: WorkingMemory | None = None,
    entry_types: tuple[str, ...],
    max_entries: int,
    max_item_chars: int,
) -> list[str]:
    """从 working memory 抽指定类型的结构化条目。"""
    entries = _collect_ranked_entries(
        working_memory=working_memory,
        entry_types=entry_types,
    )
    result: list[str] = []
    for entry in entries:
        if _looks_like_structured_noise(entry.content):
            continue
        result.append(_shorten(entry.content, max_item_chars))
        if len(result) >= max_entries:
            break
    return result


def _collect_tool_findings(working_memory: WorkingMemory) -> list[str]:
    """提取值得续带的工具发现。"""
    findings = _collect_snapshot_entries(
        working_memory=working_memory,
        entry_types=("reflection_file", "error_context"),
        max_entries=2,
        max_item_chars=110,
    )
    ranked = sorted(
        findings,
        key=_tool_finding_priority,
        reverse=True,
    )
    return ranked[:2]


def _extract_assistant_decision_points(raw_content: str) -> list[str]:
    """把 assistant 结论拆成更小的语义点，避免整段说明被当成单条决策。"""
    points: list[str] = []
    decision_tokens = (
        "负责",
        "顺序",
        "优先",
        "需要",
        "应该",
        "避免",
        "治理",
        "拆成",
        "保留",
        "session memory compact",
        "full compact",
        "tool result budget",
        "read dedup",
        "working memory",
        "tool_findings",
        "context",
        "pipeline",
        "compact",
    )
    for raw_line in raw_content.splitlines():
        normalized = re.sub(r"^\s*(?:[-*•]+|\d+\.)\s*", "", raw_line).strip()
        normalized = " ".join(normalized.split())
        if not normalized or len(normalized) < 12:
            continue
        if normalized.endswith(("：", ":")):
            continue
        lowered = normalized.lower()
        if not any(token in normalized or token in lowered for token in decision_tokens):
            continue
        if _looks_like_structured_noise(normalized):
            continue
        points.append(_shorten(normalized, 110))
    if points:
        return _dedupe_lines(points)[:4]
    fallback = " ".join(raw_content.strip().split())
    if 12 <= len(fallback) <= 120 and not _looks_like_structured_noise(fallback):
        return [_shorten(fallback, 110)]
    return []


def _extract_focus_terms(lines: list[str]) -> set[str]:
    terms: set[str] = set()
    for raw_line in lines:
        normalized = " ".join(str(raw_line).strip().lower().split())
        if not normalized:
            continue
        # 英文 token、数字串、中文短语都保留一份。
        # 中文额外切 2-4 字 ngram，是为了提升“偏题/主线漂移/压缩语义”这种短词的对齐命中率。
        for token in re.findall(r"[a-z][a-z0-9_.-]{2,}", normalized):
            if token not in _FOCUS_STOP_TOKENS:
                terms.add(token)
        for token in re.findall(r"[a-z0-9_-]{4,}", normalized):
            if token not in _FOCUS_STOP_TOKENS:
                terms.add(token)
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,24}", normalized):
            compact_chunk = chunk.strip()
            if len(compact_chunk) < 2:
                continue
            if compact_chunk not in _FOCUS_STOP_TOKENS:
                terms.add(compact_chunk)
            max_ngram = min(4, len(compact_chunk))
            for size in range(2, max_ngram + 1):
                for index in range(0, len(compact_chunk) - size + 1):
                    gram = compact_chunk[index:index + size]
                    if gram in _FOCUS_STOP_TOKENS:
                        continue
                    terms.add(gram)
    return terms


def _focus_alignment_score(text: str, focus_terms: set[str]) -> int:
    normalized = " ".join(str(text).strip().lower().split())
    if not normalized or not focus_terms:
        return 0

    score = 0
    matched = 0
    # 长词优先，避免“问题/目标/风险”这种过泛短词把排序带偏。
    for term in sorted(focus_terms, key=len, reverse=True):
        if term not in normalized:
            continue
        matched += 1
        score += 1 if len(term) <= 2 else 2 if len(term) <= 4 else 3
        if matched >= 8:
            break
    return score


def _generic_focus_priority(text: str) -> tuple[int, int]:
    return _semantic_line_score(text), len(text)


def _rerank_lines_by_focus(
    lines: list[str],
    *,
    focus_terms: set[str],
    base_priority_fn,
) -> list[str]:
    ranked = sorted(
        lines,
        key=lambda line: (
            _focus_alignment_score(line, focus_terms),
            *base_priority_fn(line),
        ),
        reverse=True,
    )
    return _dedupe_lines(ranked)


def _extract_tool_result_findings(*, tool_name: str, raw_content: str) -> list[str]:
    """从 tool_result 中优先抽取语义结论，而不是文件头和路径元信息。"""
    candidates: list[tuple[int, str]] = []

    for raw_line in raw_content.splitlines():
        normalized = _normalize_tool_result_candidate(raw_line)
        if not normalized:
            continue
        if _looks_like_structured_noise(normalized) or _looks_like_path_only(normalized):
            continue
        # 规则压缩里最危险的是把“目录头 / filler / preview 提示”误当成结论继续承接。
        # 这里先做一层低价值过滤，宁可少带一条，也不要把噪音写进后续基线。
        if _looks_like_low_value_tool_finding(normalized):
            continue

        score = _semantic_line_score(normalized)
        if score <= 0:
            continue
        candidates.append((score, _shorten(normalized, 100)))

    deduped: list[str] = []
    seen: set[str] = set()
    for _, item in sorted(candidates, key=lambda pair: (pair[0], len(pair[1])), reverse=True):
        dedupe_key = " ".join(item.lower().split())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(item)
        if len(deduped) >= 2:
            break
    return deduped


def _looks_like_low_value_tool_finding(text: str) -> bool:
    """过滤目录头、预览提示和 filler 句，避免它们被误当成核心工具结论。"""
    normalized = " ".join(str(text).strip().lower().split())
    if not normalized:
        return True
    if normalized.startswith(
        ("root:", "total_entries:", "returned_entries:", "search_root:", "pattern:")
    ):
        return True
    if any(token in normalized for token in ("filler", "preview", "omitted", "truncated: no")):
        return True
    if normalized.endswith("继续拉高上下文压力。") or normalized.endswith("继续拉高上下文压力"):
        return True
    return False


def _normalize_tool_result_candidate(raw_line: str) -> str:
    line = raw_line.strip()
    if not line:
        return ""

    lowered = line.lower()
    if any(token in lowered for token in _TOOL_RESULT_SKIP_TOKENS):
        return ""
    if line.startswith("... [") or line.startswith("...（"):
        return ""

    line = re.sub(r"^\d+\.\s+", "", line)
    if any(line.startswith(prefix) for prefix in _TOOL_RESULT_META_PREFIXES):
        return ""
    if line.startswith("ERROR:"):
        return line.split(":", 1)[1].strip()

    if ":" in line:
        head, tail = line.split(":", 1)
        normalized_head = head.strip()
        normalized_tail = tail.strip()
        if _is_tool_result_header(normalized_head) and normalized_tail:
            return normalized_tail
    return line


def _is_tool_result_header(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if normalized.upper() == normalized and re.search(r"[A-Z_]", normalized):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9_\-]+", normalized))


def _semantic_line_score(text: str) -> int:
    lowered = text.lower()
    score = 0

    if len(text) >= 12:
        score += 1
    if any(token in lowered for token in _SEMANTIC_HINT_TOKENS):
        score += 4
    if any(token in text for token in ("：", "，", "。", "；")):
        score += 2
    if any(token in lowered for token in ("应", "应该", "需要", "避免", "导致", "验证")):
        score += 2
    if _looks_like_path_only(text):
        score -= 5
    if re.search(r"\b(tmp|app|tests)[/\\]", lowered):
        score -= 3
    if len(text) < 10 or len(text) > 140:
        score -= 2
    return score


def _tool_finding_priority(text: str) -> tuple[int, int]:
    score = _semantic_line_score(text)
    if _looks_like_path_only(text):
        score -= 4
    lowered = text.lower()
    if "已落盘" in text or "原始字符数" in text or lowered.startswith("[工具结果已"):
        score -= 6
    return score, len(text)


def _looks_like_path_only(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if re.fullmatch(r"[A-Za-z]:[/\\].+", normalized):
        return True
    if re.fullmatch(r"[\w.\- /\\]+", normalized) and ("/" in normalized or "\\" in normalized):
        alpha_count = sum(1 for ch in normalized if ch.isalpha())
        return alpha_count <= max(8, len(normalized) // 3)
    return False


def _merge_snapshot_lines(
    *,
    primary: list[str],
    secondary: list[str],
    max_entries: int,
    max_item_chars: int,
) -> list[str]:
    """合并当前值和续带值，当前值优先。"""
    result: list[str] = []
    seen: set[str] = set()

    for line in [*primary, *secondary]:
        normalized = _shorten(str(line).strip(), max_item_chars)
        dedupe_key = " ".join(normalized.lower().split())
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result.append(normalized)
        if len(result) >= max_entries:
            break

    return result


def _merge_ranked_snapshot_lines(
    *,
    primary: list[str],
    secondary: list[str],
    max_entries: int,
    max_item_chars: int,
    priority_fn,
) -> list[str]:
    """按内容质量排序合并，避免路径类新条目挤掉旧的高价值语义事实。"""
    candidates: list[tuple[tuple[int, int, int], str, str]] = []
    for index, line in enumerate(primary):
        normalized = _shorten(str(line).strip(), max_item_chars)
        dedupe_key = " ".join(normalized.lower().split())
        if not dedupe_key:
            continue
        priority = priority_fn(normalized)
        candidates.append(((priority[0], priority[1], 1_000 - index), dedupe_key, normalized))
    for index, line in enumerate(secondary):
        normalized = _shorten(str(line).strip(), max_item_chars)
        dedupe_key = " ".join(normalized.lower().split())
        if not dedupe_key:
            continue
        priority = priority_fn(normalized)
        candidates.append(((priority[0], priority[1], 100 - index), dedupe_key, normalized))

    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    result: list[str] = []
    seen: set[str] = set()
    for _, dedupe_key, normalized in ranked:
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result.append(normalized)
        if len(result) >= max_entries:
            break
    return result


def _build_summary_fallback_section(
    *,
    older_history_summary: str,
    previous_active_context_summary: str,
) -> list[str]:
    """
    构造低优先级兜底层。

    只有在高价值结构化信号放完之后，才用旧摘要和上一版 compact baseline 补足。
    """
    lines: list[str] = []

    normalized_summary = older_history_summary.strip()
    if normalized_summary:
        lines.append("## 旧对话摘要")
        lines.append(_shorten(normalized_summary, 140))

    carry_lines = _extract_previous_baseline_lines(previous_active_context_summary)
    if carry_lines:
        lines.append("## 上次压缩延续")
        lines.extend(carry_lines[:3])

    return lines


def _snapshot_from_previous_context(previous_active_context_summary: str) -> CompactMemorySnapshot:
    """兼容旧数据：把上一版自由文本基线尽量解析回结构化快照。"""
    snapshot: CompactMemorySnapshot = {}
    current_key = ""

    for raw_line in previous_active_context_summary.splitlines():
        line = raw_line.strip()
        if not line or line in {"压缩记忆基线", "结构化压缩记忆"}:
            continue
        if line.startswith("## "):
            current_key = _TITLE_TO_SECTION_KEY.get(line[3:].strip(), "")
            continue
        if not current_key:
            continue
        if line.startswith("- "):
            content = line[2:].strip()
        elif re.match(r"^\d+\.\s+", line):
            content = re.sub(r"^\d+\.\s+", "", line)
        else:
            continue
        if current_key == "stable_constraints":
            normalized = " ".join(content.lower().split())
            if any(phrase in normalized for phrase in _CARRY_FORWARD_REJECT_PHRASES):
                continue
        if _looks_like_structured_noise(content):
            continue
        snapshot.setdefault(current_key, []).append(content)

    return snapshot


def _normalize_snapshot(snapshot: CompactMemorySnapshot | object) -> CompactMemorySnapshot:
    """清洗结构化快照，去掉空白和无效 key。"""
    if not isinstance(snapshot, dict):
        return {}

    normalized: CompactMemorySnapshot = {}
    for key in _SECTION_KEY_TO_TITLE:
        raw_lines = snapshot.get(key, [])
        if not isinstance(raw_lines, list):
            continue
        lines = [str(item).strip() for item in raw_lines if str(item).strip()]
        deduped = _dedupe_lines(lines)
        if deduped:
            normalized[key] = deduped
    return normalized


def _extract_previous_baseline_lines(previous_active_context_summary: str) -> list[str]:
    """从上一版 compact memory 中提取可延续的有效内容，避免递归复制标题。"""
    result: list[str] = []
    current_section = ""
    for raw_line in previous_active_context_summary.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"压缩记忆基线", "结构化压缩记忆"}:
            continue
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue
        if current_section in _CARRY_FORWARD_BLOCKED_SECTIONS:
            continue
        if _looks_like_structured_noise(line):
            continue
        normalized = " ".join(line.lstrip("- ").strip().lower().split())
        if any(phrase in normalized for phrase in _CARRY_FORWARD_REJECT_PHRASES):
            continue
        result.append(f"- {_shorten(line.lstrip('- ').strip(), 90)}")
    return _dedupe_lines(result)


def _append_with_global_budget(
    lines: list[str],
    section_lines: list[str],
    *,
    max_tokens: int,
) -> bool:
    """把 section 追加到最终上下文中，同时保证总 token 不超预算。"""
    candidate = "\n".join([*lines, *section_lines]).strip()
    if estimate_tokens(candidate) <= max_tokens:
        lines.extend(section_lines)
        return True

    partial = list(lines)
    for line in section_lines:
        next_candidate = "\n".join([*partial, line]).strip()
        if estimate_tokens(next_candidate) > max_tokens:
            return False
        partial.append(line)

    lines[:] = partial
    return True


def _dedupe_entries(entries: list[WorkingMemoryEntry]) -> list[WorkingMemoryEntry]:
    """按归一化内容去重，避免同一句在多个来源里重复出现。"""
    result: list[WorkingMemoryEntry] = []
    seen: set[str] = set()
    for entry in entries:
        normalized = " ".join(entry.content.strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(entry)
    return result


def _dedupe_lines(lines: list[str]) -> list[str]:
    """去掉重复的 carry-over 行。"""
    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = " ".join(line.strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(line)
    return result


def _entry_type_priority(entry_type: str) -> int:
    """定义 compact memory 内部的类型优先级。"""
    priority_map = {
        "user_preference": 100,
        "project_constraint": 95,
        "recent_risk": 90,
        "error_context": 88,
        "key_decision": 82,
        "active_task": 78,
        "user_intent": 72,
    }
    return priority_map.get(entry_type, 50)


def _shorten(text: str, max_chars: int) -> str:
    """统一裁剪摘要段落，防止 compact memory 本身反过来撑爆上下文。"""
    normalized = " ".join(str(text).strip().split())
    if len(normalized) <= max_chars:
        return normalized
    suffix = " ...[已截断]"
    head_limit = max(0, max_chars - len(suffix))
    return f"{normalized[:head_limit]}{suffix}"


def _looks_like_structured_noise(text: str) -> bool:
    """过滤明显不适合作为结构化记忆槽位的目录树、表格和代码正文。"""
    normalized = str(text).strip()
    if not normalized:
        return True

    lower_line = normalized.lower()
    if any(token in lower_line for token in _TOOL_RESULT_SKIP_TOKENS):
        return True
    if lower_line.startswith(_NOISE_PREFIXES):
        return True
    if _MARKDOWN_TABLE_RE.match(normalized):
        return True
    if lower_line.startswith(_CODE_LIKE_PREFIXES):
        return True
    if normalized.count("```") >= 1:
        return True
    if normalized.count("|") >= 4:
        return True
    if normalized.count("/") >= 4 and " " not in normalized:
        return True
    return False

