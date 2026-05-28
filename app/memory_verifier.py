from __future__ import annotations

import json
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

你必须优先区分以下两类情况：

一、什么时候必须判为 supersede_store
满足下面大部分特征时，应优先判 supersede_store，而不是 conflict：
- 新旧记忆是同一主题、同一类别、同一层级约定
- 新记忆明确带有“更新/替代/切换”语气
- 新记忆像是在给出“现在起生效的新规范”
- 旧记忆不是错主题，而是被新规则替代

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

典型例子：
- 旧：优先使用 REST API
- 新：GraphQL 更适合复杂查询
如果新记忆没有明确说“项目约定改为 GraphQL”，那更像 conflict 或 store，而不是 supersede_store。

三、其他动作的判断原则
- duplicate: 语义几乎相同，只是换了说法、轻微展开、补了不重要细节
- reject: 过程性内容、一次性结论、礼貌回复、临时操作、未稳定的猜测
- store: 与已有记忆相关，但不是重复，也不是替代更新，也不是明显冲突

硬性约束：
- 不要因为只是“主题接近”就判 duplicate
- 不要因为“方向相反”就直接判 conflict，先检查是否属于 supersede_store
- 只要能合理判断为“新规则替代旧规则”，优先判 supersede_store
- 如果 matched_memory_id 留空，则 action 不能是 supersede_store / duplicate / conflict

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

        action = str(payload.get("action", "")).strip().lower()
        if action not in {
            "store",
            "supersede_store",
            "duplicate",
            "conflict",
            "reject",
        }:
            return None

        reason = " ".join(str(payload.get("reason", "")).strip().split())
        matched_memory_id = str(payload.get("matched_memory_id", "")).strip()
        if action in {"supersede_store", "duplicate", "conflict"} and not matched_memory_id:
            return None
        return MemoryVerificationDecision(
            action=action,  # type: ignore[arg-type]
            reason=reason,
            matched_memory_id=matched_memory_id,
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

        for entry in similar_entries:
            normalized_existing = self._normalize_text(entry.content)
            similarity = self._jaccard_similarity(normalized_candidate, normalized_existing)

            if similarity >= 0.92 or normalized_candidate == normalized_existing:
                return MemoryVerificationDecision(
                    action="duplicate",
                    reason=f"本地兜底判定为重复，相似度 {similarity:.2f}",
                    matched_memory_id=entry.id,
                )

            # “新规范替代旧规范”这类更新，词面上往往会替换掉一部分关键 token，
            # 所以相似度通常明显低于纯 duplicate。
            # 这里把门槛放宽到 0.35，避免这类更新被误漏到普通 store。
            if (
                similarity >= 0.35
                and self._normalize_text(entry.category) == candidate_category
                and self._looks_conflicting(entry.content, candidate.content)
            ):
                if self._looks_like_superseding_update(candidate, entry):
                    return MemoryVerificationDecision(
                        action="supersede_store",
                        reason=f"本地兜底判定为同主题替代更新，相似度 {similarity:.2f}",
                        matched_memory_id=entry.id,
                    )

                return MemoryVerificationDecision(
                    action="conflict",
                    reason=f"本地兜底判定为冲突，相似度 {similarity:.2f}",
                    matched_memory_id=entry.id,
                )

        return MemoryVerificationDecision(
            action="store",
            reason="本地兜底未发现重复或冲突",
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
            "- 先判断是否是 duplicate",
            "- 如果不是 duplicate，再优先判断是否属于 supersede_store",
            "- 只有在无法证明是替代更新时，才可以判 conflict",
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
