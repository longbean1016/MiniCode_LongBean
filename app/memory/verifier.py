from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

from app.agent.circuit_breaker import CircuitBreaker
from app.logger import log_event
from app.memory.store import MemoryEntry
from app.memory.vector_index import MemoryVectorIndex
from app.agent.retry import RetryPolicy, run_with_retry, should_retry_model_error

# verifier 只负责判断候选记忆和已有记忆的关系，
# 不负责从对话里抽取候选记忆本身。


# verifier 的最终动作。
# `supersede_store` 表示：
# - 新记忆允许写入
# - 它不是普通新增，而是“新版本替代旧版本”
# - 上层在写入后，需要让旧版本逐步退出主检索面
VerificationAction = Literal[
    "store",
    "supersede_store",
    "duplicate",
    "conflict",
    "reject",
]


@dataclass(slots=True)
class MemoryVerificationDecision:
    """
    verifier 对一条候选长期记忆给出的最终结论。

    字段说明：
    - `action`: 最终动作
    - `reason`: 简短原因，便于日志排查
    - `matched_memory_id`:
      当动作是 `duplicate / conflict / supersede_store` 时，
      这里保存关联的旧记忆 id
    """

    action: VerificationAction
    reason: str = ""
    matched_memory_id: str = ""


VALID_VERIFICATION_ACTIONS: set[str] = {
    "store",
    "supersede_store",
    "duplicate",
    "conflict",
    "reject",
}

MATCH_REQUIRED_ACTIONS: set[str] = {
    "supersede_store",
    "duplicate",
    "conflict",
}

# 这些词通常意味着“当前规则正在更新旧规则”，
# 需要优先考虑是否应判为 supersede_store。
SUPERSEDE_MARKERS: tuple[str, ...] = (
    "统一",
    "改为",
    "切换到",
    "迁移到",
    "替换为",
    "不再",
    "只允许",
    "固定为",
    "以后",
    "必须",
    "采用",
    "保留",
    "唯一",
    "默认",
    "switch to",
    "change to",
    "migrate to",
    "replace with",
    "use only",
    "no longer",
)

# 这些词更像临时说明、未确认状态或低稳定度信息。
REJECT_MARKERS: tuple[str, ...] = (
    "临时",
    "暂时",
    "先这样",
    "试试",
    "待确认",
    "猜测",
    "可能",
    "也许",
    "todo",
    "fixme",
    "wip",
    "for now",
    "temporary",
    "maybe",
    "probably",
    "guess",
)

NON_PERSISTENT_PATTERNS: tuple[str, ...] = (
    r"\bthanks?\b",
    r"\bok\b",
    r"\bhello\b",
    r"\bhi\b",
    r"收到",
    r"好的",
    r"明白",
    r"已处理",
    r"稍后",
    r"回头",
    r"今天",
    r"明天",
    r"刚刚",
)

NEGATION_MARKERS: tuple[str, ...] = (
    "不要",
    "不能",
    "禁止",
    "避免",
    "不再",
    "avoid",
    "do not",
    "don't",
    "never",
    "must not",
    "no longer",
)


class MemoryVerifier:
    """
    长期记忆写入前的二次验证器。

    这层不负责抽取新记忆，只负责判断：
    1. 新候选与哪些旧记忆主题接近
    2. 它应是新增、重复、冲突，还是“新版本替代旧版本”

    当前策略：
    - 优先使用 Qdrant 做语义召回
    - Qdrant 不可用时退回到本地词面召回
    - 优先用模型做细粒度判断
    - 模型失败时再退回本地兜底规则
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        *,
        vector_index: MemoryVectorIndex | None = None,
        max_candidates: int = 6,
        min_similarity: float = 0.18,
        semantic_min_score: float = 0.55,
        retry_max_attempts: int = 3,
        retry_base_delay_seconds: float = 0.8,
        retry_backoff_multiplier: float = 2.0,
        retry_max_delay_seconds: float = 4.0,
        circuit_failure_threshold: int = 3,
        circuit_recovery_timeout_seconds: float = 45.0,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.vector_index = vector_index
        self.max_candidates = max_candidates
        self.min_similarity = min_similarity
        self.semantic_min_score = semantic_min_score
        self.retry_policy = RetryPolicy(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay_seconds,
            backoff_multiplier=retry_backoff_multiplier,
            max_delay_seconds=retry_max_delay_seconds,
        )
        self.circuit_breaker = CircuitBreaker(
            name="memory_verifier",
            failure_threshold=circuit_failure_threshold,
            recovery_timeout_seconds=circuit_recovery_timeout_seconds,
        )

    def find_similar_entries(
        self,
        candidate: MemoryEntry,
        existing_entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        # 先缩小候选比较集，后续 verifier 模型才不会浪费 token 在无关旧记忆上。
        """
        为候选记忆找出一小组值得验证的旧记忆。

        召回顺序：
        1. 有 Qdrant 时优先语义召回
        2. 语义召回失败或为空时，再退回词面召回
        """
        semantic_entries = self._find_similar_entries_semantically(
            candidate,
            existing_entries,
        )
        if semantic_entries:
            return semantic_entries

        return self._find_similar_entries_lexically(candidate, existing_entries)

    def verify(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
    ) -> MemoryVerificationDecision:
        # 先判“值不值得存”，再判“与谁重复/冲突/替代”，能减少无意义验证。
        """
        对候选记忆做二次验证。

        判定顺序：
        1. 先过滤本身就不值得持久化的候选
        2. 如果没有相似旧记忆，直接允许写入
        3. 有相似旧记忆时优先调 verifier 模型
        4. 模型不可用时退回本地规则
        """
        candidate_reject_reason = self._get_candidate_reject_reason(candidate)
        if candidate_reject_reason:
            return MemoryVerificationDecision(
                action="reject",
                reason=candidate_reject_reason,
            )

        if not similar_entries:
            return MemoryVerificationDecision(
                action="store",
                reason="没有找到足够接近的已有记忆",
            )

        model_decision = self._call_verifier_model(candidate, similar_entries)
        if model_decision is not None:
            return model_decision

        return self._fallback_verify(candidate, similar_entries)

    def _find_similar_entries_semantically(
        self,
        candidate: MemoryEntry,
        existing_entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        """使用 Qdrant 做语义召回。"""
        if self.vector_index is None:
            return []

        existing_by_id = {
            entry.id: entry
            for entry in existing_entries
            if entry.scope.strip().lower() == candidate.scope.strip().lower()
        }
        if not existing_by_id:
            return []

        try:
            hits = self.vector_index.search_similar_memories(
                query_text=self._build_candidate_query_text(candidate),
                top_k=self.max_candidates,
                scope=candidate.scope,
                include_archived=False,
                exclude_ids=[candidate.id] if candidate.id else [],
            )
        except Exception:
            return []

        result: list[MemoryEntry] = []
        seen_ids: set[str] = set()

        for hit in hits:
            if hit.score < self.semantic_min_score:
                continue

            # 语义召回只做粗筛，真正的 duplicate/conflict/supersede 判断在后面。
            entry = existing_by_id.get(hit.memory_id)
            if entry is None or entry.id in seen_ids:
                continue

            seen_ids.add(entry.id)
            result.append(entry)

        return result

    def _find_similar_entries_lexically(
        self,
        candidate: MemoryEntry,
        existing_entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        """Qdrant 不可用时的词面召回兜底。"""
        candidate_scope = candidate.scope.strip().lower()
        candidate_category = self._normalize_text(candidate.category)

        scored_entries: list[tuple[float, MemoryEntry]] = []
        for entry in existing_entries:
            existing_scope = entry.scope.strip().lower()
            if candidate_scope and existing_scope and existing_scope != candidate_scope:
                continue

            # 没有向量检索时，需要借助 category/tag/domain 这些结构信号兜底。
            similarity = self._jaccard_similarity(candidate.content, entry.content)
            same_topic_score = self._same_topic_score(candidate, entry)

            if self._normalize_text(entry.category) == candidate_category:
                similarity += 0.08

            shared_tags = set(self._normalize_tag_list(candidate.tags)) & set(
                self._normalize_tag_list(entry.tags)
            )
            if shared_tags:
                similarity += min(0.08, 0.02 * len(shared_tags))

            similarity += min(0.10, same_topic_score * 0.08)
            if similarity < self.min_similarity:
                continue

            scored_entries.append((similarity, entry))

        scored_entries.sort(
            key=lambda item: (item[0], item[1].updated_at),
            reverse=True,
        )
        return [entry for _, entry in scored_entries[: self.max_candidates]]

    def _call_verifier_model(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
    ) -> MemoryVerificationDecision | None:
        """
        调模型做二次验证。

        重点是把“可替代更新”与“普通冲突”分清：
        - `supersede_store`: 同主题，且新记忆明显是当前生效的新版本
        - `conflict`: 有冲突，但无法证明它是一次规则替代
        """
        system_prompt = """
你是代码 Agent 的长期记忆写入验证器。

你会收到：
1. 一条新的候选长期记忆
2. 若干条与它主题接近的已有长期记忆

你的任务是判断新候选相对已有记忆应执行哪一种动作。
只允许返回以下动作之一：
- store
- supersede_store
- duplicate
- conflict
- reject

动作定义：
- store: 允许写入，它提供了新的稳定价值
- supersede_store: 允许写入，它是“当前有效的新版本规则/结论”，应替代某条旧记忆
- duplicate: 不允许写入，因为与已有记忆本质重复
- conflict: 不允许写入，因为与已有记忆方向冲突，但不能证明它是一次新版本替代
- reject: 不允许写入，因为候选本身不稳定、过于临时，或不适合长期保存

先判断候选本身是否值得进入长期记忆。下面内容通常应判为 reject：
- 礼貌回复、确认回复、寒暄
- 一次性执行过程、临时安排、短时任务播报
- TODO、WIP、待确认、猜测、可能、也许这类未稳定信息
- 只描述本轮上下文，没有跨轮复用价值的内容

然后再判断它与旧记忆的关系。

什么时候优先判为 supersede_store：
- 新旧记忆约束的是同一个对象、同一个决策面、同一层级规则
- 新记忆明确带有“统一 / 改为 / 不再 / 只允许 / 默认 / switch to / replace with / no longer”等更新语气
- 新记忆像是在给出“现在起生效的新规范”
- 旧记忆不是错主题，而是被新规则替代

什么时候只能判为 conflict：
- 新旧记忆方向相反
- 但无法确认新记忆是在发布新版本规范
- 或两者来自不同假设、不同上下文、不同方案层级
- 或候选内容过于模糊，不能证明它是在替代旧记忆

什么时候判为 duplicate：
- 语义基本相同，只是换了说法、轻微展开、补了不重要细节

不要因为“只是相关”就判 duplicate。
不要因为“方向相反”就直接判 conflict，先检查是否属于 supersede_store。

规则约束：
- store / reject 时 matched_memory_id 必须留空
- duplicate / conflict / supersede_store 时必须命中提供列表中的一条旧记忆 id
- reason 要简短具体

只返回 JSON：
{
  "action": "store|supersede_store|duplicate|conflict|reject",
  "reason": "一句简短原因",
  "matched_memory_id": "需要时填写旧记忆 id，否则留空"
}
""".strip()

        user_prompt = self._build_verifier_user_prompt(candidate, similar_entries)
        if not self.circuit_breaker.allow_request():
            return None

        def _request_model() -> Any:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )

        try:
            response = run_with_retry(
                _request_model,
                policy=self.retry_policy,
                should_retry=should_retry_model_error,
                on_retry=lambda attempt, error, delay: log_event(
                    (
                        f"长期记忆 verifier 调用失败，准备第 {attempt + 1} 次尝试："
                        f"{type(error).__name__}: {error}，等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.circuit_breaker.record_failure(error)
            return None

        self.circuit_breaker.record_success()

        raw_content = response.choices[0].message.content or ""
        payload = self._parse_json_payload(raw_content)
        if not isinstance(payload, dict):
            return None

        return self._normalize_decision(
            candidate,
            similar_entries,
            action=payload.get("action", ""),
            reason=payload.get("reason", ""),
            matched_memory_id=payload.get("matched_memory_id", ""),
            reason_prefix="模型判定",
        )

    def _fallback_verify(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
    ) -> MemoryVerificationDecision:
        """
        模型不可用时的本地兜底判定。

        顺序：
        1. 明显重复 -> duplicate
        2. 明显同主题替代更新 -> supersede_store
        3. 明显同主题冲突 -> conflict
        4. 其余情况 -> store
        """
        normalized_candidate = self._normalize_text(candidate.content)
        candidate_category = self._normalize_text(candidate.category)
        best_topic_entry: MemoryEntry | None = None
        best_topic_score = 0.0

        for entry in similar_entries:
            normalized_existing = self._normalize_text(entry.content)
            similarity = self._jaccard_similarity(normalized_candidate, normalized_existing)
            same_topic_score = self._same_topic_score(candidate, entry)

            if same_topic_score > best_topic_score:
                best_topic_score = same_topic_score
                best_topic_entry = entry

            if (
                same_topic_score >= 0.82
                and not self._looks_conflicting(entry.content, candidate.content)
                and not self._looks_like_superseding_update(candidate, entry)
                and (similarity >= 0.92 or normalized_candidate == normalized_existing)
            ):
                return self._build_decision(
                    action="duplicate",
                    reason=f"本地兜底判定为同主题重复，相似度 {similarity:.2f}",
                    matched_memory_id=entry.id,
                )

            # 对“同主题 + 明确更新语气”的情况优先判为 supersede_store。
            # 这里不强依赖冲突检测，因为“新版本替代旧版本”不一定都呈现明显否定句式。
            if (
                similarity >= 0.30
                and self._normalize_text(entry.category) == candidate_category
                and same_topic_score >= 0.72
                and self._looks_like_superseding_update(candidate, entry)
            ):
                return self._build_decision(
                    action="supersede_store",
                    reason=f"本地兜底判定为同主题替代更新，相似度 {similarity:.2f}",
                    matched_memory_id=entry.id,
                )

            if (
                similarity >= 0.35
                and self._normalize_text(entry.category) == candidate_category
                and same_topic_score >= 0.72
                and self._looks_conflicting(entry.content, candidate.content)
            ):
                if self._looks_like_superseding_update(candidate, entry):
                    return self._build_decision(
                        action="supersede_store",
                        reason=f"本地兜底判定为同主题替代更新，相似度 {similarity:.2f}",
                        matched_memory_id=entry.id,
                    )

                return self._build_decision(
                    action="conflict",
                    reason=f"本地兜底判定为同主题冲突，相似度 {similarity:.2f}",
                    matched_memory_id=entry.id,
                )

        if best_topic_entry is not None and best_topic_score >= 0.72:
            return self._build_decision(
                action="store",
                reason="本地兜底判定为同主题补充信息，不构成重复或冲突",
            )

        return self._build_decision(
            action="store",
            reason="本地兜底未发现重复、替代更新或冲突",
        )

    def _looks_like_superseding_update(
        self,
        candidate: MemoryEntry,
        existing: MemoryEntry,
    ) -> bool:
        """
        判断候选是否像“同主题的新版本替代”。

        这是更接近 minicode 风格更新链路的关键：
        - 允许新版本先入库
        - 再让旧版本退出主检索面
        """
        if self._normalize_text(candidate.category) != self._normalize_text(existing.category):
            return False

        if self._same_topic_score(candidate, existing) < 0.72:
            return False

        candidate_text = self._normalize_text(candidate.content)
        if not any(marker in candidate_text for marker in SUPERSEDE_MARKERS):
            return False

        # 没有任何共享 tag/domain 也没有较高内容重合时，不认为是同主题替代。
        has_shared_tags = bool(
            set(self._normalize_tag_list(candidate.tags)) & set(self._normalize_tag_list(existing.tags))
        )
        has_shared_domains = bool(
            {self._normalize_text(domain) for domain in candidate.domains}
            & {self._normalize_text(domain) for domain in existing.domains}
        )
        has_content_overlap = self._jaccard_similarity(candidate.content, existing.content) >= 0.30
        if not (has_shared_tags or has_shared_domains or has_content_overlap):
            return False

        # 新结论的置信度不应明显弱于旧结论。
        if candidate.confidence + 0.05 < existing.confidence:
            return False

        return True

    def _get_candidate_reject_reason(self, candidate: MemoryEntry) -> str:
        """在调模型前先过滤掉明显不该持久化的候选。"""
        normalized_content = self._normalize_text(candidate.content)
        if not normalized_content:
            return "候选内容为空，不适合写入长期记忆"

        if len(normalized_content) < 6:
            return "候选内容过短，缺少稳定可复用信息"

        # 这里尽量把“临时性”和“寒暄性”挡在模型前面，
        # 让 verifier 的 token 预算留给真正值得比对的候选。
        if any(marker in normalized_content for marker in REJECT_MARKERS):
            return "候选内容含有临时或未确认表达，不适合写入长期记忆"

        if re.search("|".join(NON_PERSISTENT_PATTERNS), normalized_content):
            return "候选内容更像礼貌回复或短时会话，不适合写入长期记忆"

        return ""

    def _normalize_decision(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
        *,
        action: Any,
        reason: Any,
        matched_memory_id: Any,
        reason_prefix: str,
    ) -> MemoryVerificationDecision | None:
        """统一校验模型返回结构，避免脏结果进入主链路。"""
        normalized_action = self._normalize_text(action)
        if normalized_action not in VALID_VERIFICATION_ACTIONS:
            return None

        normalized_reason = " ".join(str(reason).strip().split())
        normalized_matched_id = str(matched_memory_id).strip()

        matched_entry = self._find_matched_entry(candidate, similar_entries, normalized_matched_id)
        if normalized_action in MATCH_REQUIRED_ACTIONS:
            if matched_entry is None:
                return None
            # 模型 matched id 写错时，允许这里按主题相近度做一次兜底纠正。
            normalized_matched_id = matched_entry.id
        else:
            normalized_matched_id = ""

        if not normalized_reason:
            normalized_reason = self._default_reason_for_action(
                normalized_action,
                matched_entry,
                prefix=reason_prefix,
            )

        if normalized_action == "reject":
            candidate_reject_reason = self._get_candidate_reject_reason(candidate)
            if candidate_reject_reason:
                normalized_reason = candidate_reject_reason

        return self._build_decision(
            action=normalized_action,
            reason=normalized_reason,
            matched_memory_id=normalized_matched_id,
        )

    def _build_decision(
        self,
        *,
        action: str,
        reason: str,
        matched_memory_id: str = "",
    ) -> MemoryVerificationDecision:
        """统一构造最终 decision。"""
        normalized_reason = " ".join(str(reason).strip().split()) or "未提供原因"
        normalized_matched_id = str(matched_memory_id).strip()

        if action in MATCH_REQUIRED_ACTIONS and not normalized_matched_id:
            action = "reject"
            normalized_reason = "缺少关联旧记忆 id，无法安全执行需要命中旧记忆的动作"

        if action not in MATCH_REQUIRED_ACTIONS:
            normalized_matched_id = ""

        return MemoryVerificationDecision(
            action=action,  # type: ignore[arg-type]
            reason=normalized_reason,
            matched_memory_id=normalized_matched_id,
        )

    def _find_matched_entry(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
        matched_memory_id: str,
    ) -> MemoryEntry | None:
        """
        优先使用模型返回的 matched_memory_id。
        如果没有或不可靠，再按同主题分数兜底挑选。
        """
        entry_by_id = {entry.id: entry for entry in similar_entries if entry.id}
        normalized_id = str(matched_memory_id).strip()
        if normalized_id and normalized_id in entry_by_id:
            return entry_by_id[normalized_id]

        # 如果模型没给出可用 id，就退回到“主题一致性 + 正文相似度”的混合打分，
        # 尽量把动作绑到最可能的那条旧记忆上。
        best_entry: MemoryEntry | None = None
        best_score = 0.0
        for entry in similar_entries:
            # 兜底匹配不只看正文相似度，还会叠加“是不是同一决策面”的结构分。
            score = self._same_topic_score(candidate, entry)
            score += self._jaccard_similarity(candidate.content, entry.content) * 0.35
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < 0.75:
            return None
        return best_entry

    def _default_reason_for_action(
        self,
        action: str,
        matched_entry: MemoryEntry | None,
        *,
        prefix: str,
    ) -> str:
        """给空 reason 提供最小可读兜底。"""
        relation_suffix = f"，命中旧记忆 {matched_entry.id}" if matched_entry else ""
        if action == "store":
            return f"{prefix}为新增稳定信息"
        if action == "supersede_store":
            return f"{prefix}为同主题新版本替代旧版本{relation_suffix}"
        if action == "duplicate":
            return f"{prefix}为同主题重复信息{relation_suffix}"
        if action == "conflict":
            return f"{prefix}为同主题冲突信息{relation_suffix}"
        return f"{prefix}为不适合持久化的内容"

    def _same_topic_score(self, candidate: MemoryEntry, existing: MemoryEntry) -> float:
        """
        判断两条记忆是否在说同一个“决策面”。

        这里不是简单相似度，而是看：
        - scope 是否一致
        - category 是否一致
        - tags / domains 是否重叠
        - 文本是否有一定内容交集
        """
        score = 0.0

        if self._normalize_text(candidate.scope) == self._normalize_text(existing.scope):
            score += 0.30
        if self._normalize_text(candidate.category) == self._normalize_text(existing.category):
            score += 0.20

        # tags / domains 在长期规则里通常比正文措辞更稳定。
        candidate_tags = set(self._normalize_tag_list(candidate.tags))
        existing_tags = set(self._normalize_tag_list(existing.tags))
        tag_overlap = len(candidate_tags & existing_tags)
        score += min(0.20, tag_overlap * 0.08)

        candidate_domains = {self._normalize_text(domain) for domain in candidate.domains}
        existing_domains = {self._normalize_text(domain) for domain in existing.domains}
        domain_overlap = len(candidate_domains & existing_domains)
        score += min(0.15, domain_overlap * 0.07)

        content_similarity = self._jaccard_similarity(candidate.content, existing.content)
        score += min(0.15, content_similarity * 0.30)
        return score

    def _build_candidate_query_text(self, candidate: MemoryEntry) -> str:
        """把候选记忆整理成语义召回查询文本。"""
        parts = [
            f"category: {candidate.category}",
            f"scope: {candidate.scope}",
            f"content: {candidate.content}",
        ]
        # tags / domains 即使没直接写在正文里，也会显著影响召回主题是否准确。
        if candidate.tags:
            parts.append(f"tags: {', '.join(candidate.tags)}")
        if candidate.domains:
            parts.append(f"domains: {', '.join(candidate.domains)}")
        return "\n".join(parts).strip()

    def _build_verifier_user_prompt(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
    ) -> str:
        """构造 verifier 的用户提示词。"""
        project_files_touched = candidate.extra.get("project_files_touched", [])
        if not isinstance(project_files_touched, list):
            project_files_touched = []

        lines: list[str] = [
            "## 新候选记忆",
            f"id: {candidate.id}",
            f"category: {candidate.category}",
            f"scope: {candidate.scope}",
            f"confidence: {candidate.confidence}",
            f"content: {candidate.content}",
            f"tags: {', '.join(candidate.tags) if candidate.tags else '(none)'}",
            f"domains: {', '.join(candidate.domains) if candidate.domains else '(none)'}",
            f"project_files_touched: {', '.join(project_files_touched) if project_files_touched else '(none)'}",
            "",
            "## 判定提醒",
            "- 先判断候选本身是否稳定，是否适合长期持久化",
            "- 如果只是礼貌回复、临时播报、一次性说明，直接 reject",
            "- 先看是否 duplicate",
            "- 若不是 duplicate，再优先判断是否属于 supersede_store",
            "- 只有无法证明是替代更新时，才判 conflict",
            "- 只有同一对象、同一决策面、同一层级规则的更新，才可能是 supersede_store",
            "- 如果只是相关但不是同一条规则，通常应判 store",
            "",
            "## 相似旧记忆",
        ]

        for entry in similar_entries:
            lines.extend(
                [
                    f"- id: {entry.id}",
                    f"  category: {entry.category}",
                    f"  scope: {entry.scope}",
                    f"  confidence: {entry.confidence}",
                    f"  tags: {', '.join(entry.tags) if entry.tags else '(none)'}",
                    f"  domains: {', '.join(entry.domains) if entry.domains else '(none)'}",
                    f"  content: {entry.content}",
                ]
            )

        return "\n".join(lines).strip()

    def _parse_json_payload(self, text: str) -> Any:
        """解析模型返回的 JSON，并兼容 ```json 代码块。"""
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _normalize_text(self, text: str) -> str:
        """标准化文本，便于本地相似度计算。"""
        return " ".join(str(text).strip().lower().split())

    def _normalize_tag_list(self, tags: list[str]) -> list[str]:
        """标准化标签列表。"""
        return [self._normalize_text(tag) for tag in tags if self._normalize_text(tag)]

    def _tokenize(self, text: str) -> set[str]:
        """
        做轻量切词。

        当前规则：
        - 英文按单词切
        - 中文按单字切
        """
        normalized = self._normalize_text(text)
        if not normalized:
            return set()
        return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized))

    def _jaccard_similarity(self, left: str, right: str) -> float:
        """计算两段文本的 Jaccard 相似度。"""
        left_tokens = self._tokenize(left)
        right_tokens = self._tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0

        union = left_tokens | right_tokens
        if not union:
            return 0.0

        intersection = left_tokens & right_tokens
        return len(intersection) / len(union)

    def _looks_conflicting(self, left: str, right: str) -> bool:
        """
        用于本地兜底的轻量冲突判断。

        这里只识别最明显的否定方向冲突，
        更细的语义判断仍然交给 verifier 模型。
        """
        left_text = self._normalize_text(left)
        right_text = self._normalize_text(right)

        left_has_neg = any(marker in left_text for marker in NEGATION_MARKERS)
        right_has_neg = any(marker in right_text for marker in NEGATION_MARKERS)
        return left_has_neg != right_has_neg
