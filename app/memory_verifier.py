from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

from app.memory_store import MemoryEntry
from app.memory_vector_index import MemoryVectorIndex


# verifier 的最终动作。
# `supersede_store` 表示：
# - 新记忆允许写入
# - 它不是普通新增，而是“新版本替代旧版本”
# - 上层在写入后，需要把旧版本降级出主检索面
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
    - `reason`: 简短原因，便于日志和排查
    - `matched_memory_id`:
        当动作是 duplicate / conflict / supersede_store 时，
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

# 这些词通常意味着“当前规则更新了旧规则”，应优先考虑 supersede_store。
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

# 这些词往往是过程性、临时性或低稳定度信息，不适合直接写入长期记忆。
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
    r"\b收到\b",
    r"\b好的\b",
    r"\b明白\b",
    r"\b已处理\b",
    r"\b稍后\b",
    r"\b回头\b",
    r"\b今天\b",
    r"\b明天\b",
    r"\b刚刚\b",
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

    这一层不负责“抽取新记忆”，只负责判断：
    1. 新候选与哪些旧记忆主题接近
    2. 它应该是新增、重复、冲突，还是“新版本替代旧版本”

    当前策略：
    - 优先使用 Qdrant 做语义召回
    - 如果语义召回不可用，再退回到本地词面召回
    - 如果模型判定失败，再退回到本地启发式规则
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
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.vector_index = vector_index
        self.max_candidates = max_candidates

        # 本地词面召回的最小 Jaccard 门槛。
        self.min_similarity = min_similarity

        # Qdrant 命中的最低语义分数。
        # 低于这个值，通常说明主题已经不够接近，不值得再做 duplicate/conflict 判断。
        self.semantic_min_score = semantic_min_score

    def find_similar_entries(
        self,
        candidate: MemoryEntry,
        existing_entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
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
        """
        对候选记忆做二次验证。

        关键点：
        - 没有相似旧记忆时，直接允许写入
        - 有相似旧记忆时，优先调用 verifier 模型
        - 模型失败时，使用本地兜底规则

        这里新增了 `supersede_store`：
        如果新记忆属于“同主题、更稳定的新版本结论”，
        就不要被 conflict 直接拦住，而是允许它入库，
        再由 curator 让旧版本退位。
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
        """
        使用 Qdrant 做语义召回。

        注意：
        - Qdrant 只负责召回 id
        - 本地 JSON 仍然是权威数据源
        """
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
        """
        本地词面召回兜底。

        当 Qdrant 不可用时，仍然要尽量把“可能同主题的旧记忆”召回出来，
        否则 verifier 无法判断 duplicate / conflict / supersede_store。
        """
        candidate_scope = candidate.scope.strip().lower()
        candidate_category = self._normalize_text(candidate.category)

        scored_entries: list[tuple[float, MemoryEntry]] = []
        for entry in existing_entries:
            existing_scope = entry.scope.strip().lower()
            if candidate_scope and existing_scope and existing_scope != candidate_scope:
                continue

            similarity = self._jaccard_similarity(candidate.content, entry.content)
            if self._normalize_text(entry.category) == candidate_category:
                similarity += 0.08

            shared_tags = set(self._normalize_tag_list(candidate.tags)) & set(
                self._normalize_tag_list(entry.tags)
            )
            if shared_tags:
                similarity += min(0.08, 0.02 * len(shared_tags))

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

        这里重点强化 prompt，让模型更稳定地区分：
        - 真冲突：方向相反，但不能证明是新版本替代旧版本
        - 可替代更新：同主题上出现了现在生效的新规范，应放行为 supersede_store
        """
        system_prompt = """
你是代码 Agent 的长期记忆写入验证器。

你会收到：
1. 一条新的候选长期记忆
2. 若干条与它主题接近的已有长期记忆

你的唯一任务是判断：这条新记忆相对已有记忆，应该执行哪一种动作。

可选动作只有：
- store: 允许写入，它提供了新的稳定价值
- supersede_store: 允许写入，并且它是“新版本”，应该替代某条同主题旧记忆
- duplicate: 不允许写入，因为与已有记忆本质重复
- conflict: 不允许写入，因为与已有记忆方向相反，但无法判断为“新版本替代旧版本”
- reject: 不允许写入，因为内容本身不够稳定、不够长期、或过于临时

先判断候选本身是否值得进入长期记忆。下面这些内容默认应判 reject：
- 礼貌回复、确认语、会话衔接句，例如“收到”“好的”“我来处理”
- 一次性执行过程、临时计划、今天/稍后再做之类的短时安排
- TODO、WIP、待确认、猜测、可能、也许、试试看之类的未稳定信息
- 只描述本轮上下文而没有可复用规则/事实/偏好的内容

你必须优先区分以下两类情况：

一、什么时候必须判为 supersede_store
满足下面大部分特征时，应优先判 supersede_store，而不是 conflict：
- 新旧记忆是同一主题、同一类别、同一层级约定
- 新记忆明确带有“更新/替代/切换”语气
- 新记忆像是在给出“现在起生效的新规范”
- 旧记忆不是错主题，而是被新规则替代

“同主题更新”指的是：它们在同一 scope 下约束同一对象、同一决策面、同一规则层级。
例如都在谈 embedding provider、默认数据库、接口约定、目录规范。

常见替代标记包括但不限于：
- 中文：统一、改为、不再、只允许、固定为、以后、必须、采用、保留、默认
- 英文：switch to, change to, replace with, migrate to, use only, no longer

典型例子：
- 旧：Embedding provider 统一使用 OpenAI 兼容接口
- 新：Embedding provider 统一改为 DashScope 兼容接口，不再使用 OpenAI 兼容接口
这应判为 supersede_store，不是 conflict。

二、什么时候只能判为 conflict
满足下面情况时，才应判 conflict：
- 新旧记忆方向相反
- 但看不出新记忆是在发布“新版本规范”
- 或者两条记忆来自不同假设、不同方案、不同上下文
- 或者候选内容太模糊，无法确认它是在替代旧记忆

“普通相似”不等于 duplicate，也不等于 conflict。
如果只是主题接近、共享一些词、补充了不同维度信息，但没有表达同一条规则重复，也没有方向冲突，应判 store。

典型例子：
- 旧：优先使用 REST API
- 新：GraphQL 更适合复杂查询
如果新记忆没有明确说“项目约定改为 GraphQL”，那更像 conflict 或 store，而不是 supersede_store。

三、其他动作的判断原则
- duplicate: 语义几乎相同，只是换了说法、轻微展开、补了不重要细节
- reject: 过程性内容、一次性结论、礼貌回复、临时操作、未稳定的猜测
- store: 与已有记忆相关，但不是重复，也不是替代更新，也不是明显冲突

硬性约束：
- store / reject 时 matched_memory_id 必须留空
- 不要因为只是“主题接近”就判 duplicate
- 不要因为“方向相反”就直接判 conflict，先检查是否属于 supersede_store
- 只要能合理判断为“新规则替代旧规则”，优先判 supersede_store
- 如果 matched_memory_id 留空，则 action 不能是 supersede_store / duplicate / conflict
- duplicate / conflict / supersede_store 必须命中一条最相关旧记忆，且 id 必须来自提供的相似旧记忆列表
- reason 必须简洁具体，直接说明“为何是重复 / 更新 / 冲突 / 拒绝”，不要写空话

只返回 JSON，不要返回解释性文本：
{
  "action": "store|supersede_store|duplicate|conflict|reject",
  "reason": "一句简短原因",
  "matched_memory_id": "如果 action 是 supersede_store/duplicate/conflict，则填写命中的旧记忆 id；否则留空"
}
""".strip()

        user_prompt = self._build_verifier_user_prompt(candidate, similar_entries)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            return None

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

        兜底顺序：
        1. 明显重复 -> duplicate
        2. 明显同主题替代更新 -> supersede_store
        3. 明显冲突 -> conflict
        4. 其他情况 -> store
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
                same_topic_score >= 0.78
                and not self._looks_conflicting(entry.content, candidate.content)
                and not self._looks_like_superseding_update(candidate, entry)
                and (similarity >= 0.92 or normalized_candidate == normalized_existing)
            ):
                return self._build_decision(
                    action="duplicate",
                    reason=f"本地兜底判定为同主题重复，相似度 {similarity:.2f}",
                    matched_memory_id=entry.id,
                )

            # “新规范替代旧规范”这类更新，词面上往往会替换掉一部分关键 token，
            # 所以相似度通常明显低于纯 duplicate。
            # 这里把门槛放宽到 0.35，避免这类更新被误漏到普通 store。
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
        判断当前候选是否像“新版本替代旧版本”。

        这是 minicode 风格更新链路的关键放行点：
        - 不是简单覆盖旧记忆
        - 而是允许新版本先进入库
        - 再让旧版本退出主检索面
        """
        if self._normalize_text(candidate.category) != self._normalize_text(existing.category):
            return False

        candidate_text = self._normalize_text(candidate.content)
        override_markers = (
            "统一",
            "改为",
            "不再",
            "只允许",
            "固定为",
            "以后",
            "必须",
            "采用",
            "保留",
            "默认",
            "replace with",
            "switch to",
            "change to",
            "migrate to",
            "use only",
            "no longer",
        )
        if not any(marker in candidate_text for marker in override_markers):
            return False

        # 新结论不能比旧结论明显更弱。
        if candidate.confidence + 0.05 < existing.confidence:
            return False

        return True

    def _get_candidate_reject_reason(self, candidate: MemoryEntry) -> str:
        """
        在调模型前先做一层本地过滤。

        这层只拦截明显不适合进入长期记忆的内容，避免把
        “礼貌回复 / 临时安排 / 未稳定猜测”送进后续 duplicate/conflict 判定。
        """
        normalized_content = self._normalize_text(candidate.content)
        if not normalized_content:
            return "候选内容为空，不适合写入长期记忆"

        if len(normalized_content) < 6:
            return "候选内容过短，缺少稳定可复用信息"

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
        """
        统一兜底模型/规则产物，保证动作、命中 id 和 reason 都合法。

        这里宁可返回 None 走下游 fallback，也不接受结构脏数据。
        """
        normalized_action = self._normalize_text(action)
        if normalized_action not in VALID_VERIFICATION_ACTIONS:
            return None

        normalized_reason = " ".join(str(reason).strip().split())
        normalized_matched_id = str(matched_memory_id).strip()

        matched_entry = self._find_matched_entry(candidate, similar_entries, normalized_matched_id)
        if normalized_action in MATCH_REQUIRED_ACTIONS:
            if matched_entry is None:
                return None
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
        """
        统一构造最终 decision。

        关键约束：
        - 需要命中旧记忆的动作必须带 id
        - 不需要命中旧记忆的动作必须清空 id
        - reason 至少保留一句可读原因，避免日志出现空串
        """
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

        如果模型没给、给错或给了列表外 id，再按“同主题得分 + 文本相似度”
        重新挑一个最稳妥的候选，避免把动作绑到错误旧记忆上。
        """
        entry_by_id = {entry.id: entry for entry in similar_entries if entry.id}
        normalized_id = str(matched_memory_id).strip()
        if normalized_id and normalized_id in entry_by_id:
            return entry_by_id[normalized_id]

        best_entry: MemoryEntry | None = None
        best_score = 0.0
        for entry in similar_entries:
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
        """给空 reason 提供最小可读兜底，便于日志排查。"""
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

        这个分数专门用来区分：
        - 同主题更新 / 冲突 / 重复
        - 只是普通相似、共享少量词的相关内容
        """
        score = 0.0

        if self._normalize_text(candidate.scope) == self._normalize_text(existing.scope):
            score += 0.30
        if self._normalize_text(candidate.category) == self._normalize_text(existing.category):
            score += 0.20

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
        """
        构造 verifier 的用户提示词。

        这里额外强调“是否存在明确更新语气”和“候选是否更像当前生效版本”，
        让模型在看到相反结论时先判断是不是替代更新，而不是直接给 conflict。
        """
        lines: list[str] = [
            "## 新候选记忆",
            f"id: {candidate.id}",
            f"category: {candidate.category}",
            f"scope: {candidate.scope}",
            f"confidence: {candidate.confidence}",
            f"content: {candidate.content}",
            "",
            "## 验证提醒",
            "- 先判断候选本身是否适合进入长期记忆；如果是礼貌回复、临时安排、未确认猜测，直接判 reject",
            "- 先判断是否是 duplicate",
            "- 如果不是 duplicate，再优先判断是否属于 supersede_store",
            "- 只有在无法证明是替代更新时，才可以判 conflict",
            "- “同主题更新”要求新旧记忆约束同一对象、同一规则层级；普通相关或普通相似不算同主题更新",
            "- 如果只是补充了另一个维度的信息，或者只是在相关话题上相近，但不是同一条规则，请判 store",
            "- 尤其注意候选内容里是否包含“统一 / 改为 / 不再 / 只允许 / switch to / replace with / no longer”等更新语气",
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
        """按空白做轻量切词。"""
        normalized = self._normalize_text(text)
        if not normalized:
            return set()
        return set(normalized.split())

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

        这里只识别最显眼的否定方向冲突，
        更细的语义判断还是交给 verifier 模型。
        """
        left_text = self._normalize_text(left)
        right_text = self._normalize_text(right)

        neg_markers = {
            "不要",
            "不能",
            "禁止",
            "avoid",
            "do not",
            "never",
            "must not",
            "不再",
        }
        left_has_neg = any(marker in left_text for marker in neg_markers)
        right_has_neg = any(marker in right_text for marker in neg_markers)
        return left_has_neg != right_has_neg
