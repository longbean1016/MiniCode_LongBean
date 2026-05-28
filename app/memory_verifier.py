from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

from app.memory_store import MemoryEntry
from app.memory_vector_index import MemoryVectorIndex


# verifier 最终只会给出这四种动作，供主链路决定是否真的写入长期记忆。
VerificationAction = Literal["store", "duplicate", "conflict", "reject"]


@dataclass(slots=True)
class MemoryVerificationDecision:
    """
    二次验证后的最终结论。

    字段说明：
    - action: 最终动作，决定是否允许写入
    - reason: 对本次判断的简短解释，便于日志和排查
    - matched_memory_id: 如果命中 duplicate / conflict，对应的旧记忆 id
    """

    action: VerificationAction
    reason: str = ""
    matched_memory_id: str = ""


class MemoryVerifier:
    """
    长期记忆写入前的二次验证器。

    这一层的职责不是“再次抽记忆”，而是：
    1. 先召回一小组与候选记忆语义接近的旧记忆
    2. 再判断当前候选应该 store / duplicate / conflict / reject

    当前阶段优先走 Qdrant 语义召回。
    如果没有启用向量库，或者向量召回失败，则退回到本地词面召回兜底。
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
        # 这里仍然使用主模型做 verifier 判定。
        # 向量库只负责“召回相似旧记忆”，不负责最终裁决。
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.vector_index = vector_index
        self.max_candidates = max_candidates

        # 本地词面召回的最低 Jaccard 相似度。
        self.min_similarity = min_similarity

        # 语义召回的最低分数门槛。
        # 低于这个值的命中通常主题已经不够接近，不值得拿去做 duplicate / conflict 判定。
        self.semantic_min_score = semantic_min_score

    def find_similar_entries(
        self,
        candidate: MemoryEntry,
        existing_entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        """
        为候选记忆找出一小组“值得验证”的旧记忆。

        召回顺序：
        1. 如果启用了向量库，优先做语义召回
        2. 如果语义召回失败或没命中，再退回词面召回
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

        规则：
        - 没有相似旧记忆时，直接允许写入
        - 有相似旧记忆时，优先调用 verifier 模型判断
        - 模型失败时，回退到本地启发式判断
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

        这里不会直接把 Qdrant 的 payload 当成最终记忆对象使用，
        而是只取回命中的 memory id，再映射回本地 JSON 里的 `MemoryEntry`。
        这样可以保证 `.memory/memory.json` 始终是权威数据源。
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
            if entry is None:
                continue
            if entry.id in seen_ids:
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
        本地词面召回兜底逻辑。

        当 Qdrant 没启用，或者服务暂时不可用时，
        仍然能用轻量 Jaccard + category/tag 加权的方式找出一小组候选旧记忆。
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

        这一层只判断“和已有记忆相比该怎么处理”，
        不再让模型重新产出新的长期记忆内容。
        """
        system_prompt = """
你是一个代码 Agent 的长期记忆写入验证器。

你会收到：
1. 一条新的候选长期记忆
2. 几条与它主题相近的已有长期记忆

你的任务是判断新记忆应该执行哪一个动作：
- store: 允许写入，它提供了新的稳定价值
- duplicate: 与已有记忆本质重复，不应再次写入
- conflict: 与已有记忆存在明显冲突，不应直接写入
- reject: 自身质量不足、范围太临时、或不值得长期保存

判定原则：
- duplicate: 语义基本相同，只是换了说法、补了少量不重要细节
- conflict: 同一主题上给出了方向相反、约束相反、结论相反的长期记忆
- reject: 即使没有旧记忆冲突，这条内容本身也不够稳定、不够复用、不够长期
- store: 内容稳定、可复用、项目级，并且与已有记忆相比有明显新增价值

注意：
- 不要因为只是“主题接近”就判 duplicate
- 不要因为内容更详细就自动判 store，先看是否只是重复展开
- 如果候选内容明显是过程性、一次性、临时性的，也应判 reject

只返回 JSON：
{
  "action": "store|duplicate|conflict|reject",
  "reason": "一句简短原因",
  "matched_memory_id": "如果命中 duplicate/conflict，对应旧记忆 id；否则留空"
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
        if action not in {"store", "duplicate", "conflict", "reject"}:
            return None

        reason = " ".join(str(payload.get("reason", "")).strip().split())
        matched_memory_id = str(payload.get("matched_memory_id", "")).strip()
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

        这不是最终理想方案，但能保证主链路在 verifier 模型失败时仍然可运行：
        - 明显重复 -> duplicate
        - 明显冲突 -> conflict
        - 其他情况 -> store
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

            if (
                similarity >= 0.72
                and self._normalize_text(entry.category) == candidate_category
                and self._looks_conflicting(entry.content, candidate.content)
            ):
                return MemoryVerificationDecision(
                    action="conflict",
                    reason=f"本地兜底判定为冲突，相似度 {similarity:.2f}",
                    matched_memory_id=entry.id,
                )

        return MemoryVerificationDecision(
            action="store",
            reason="本地兜底未发现重复或冲突",
        )

    def _build_candidate_query_text(self, candidate: MemoryEntry) -> str:
        """
        把候选记忆整理成语义召回查询文本。

        这里故意把 `category` / `tags` / `domains` 一起带上，
        让向量召回更容易找到“主题接近且用途接近”的旧记忆。
        """
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
        """构造 verifier 的用户提示词。"""
        lines: list[str] = [
            "## 新候选记忆",
            f"id: {candidate.id}",
            f"category: {candidate.category}",
            f"scope: {candidate.scope}",
            f"confidence: {candidate.confidence}",
            f"content: {candidate.content}",
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
        return " ".join(text.strip().lower().split())

    def _normalize_tag_list(self, tags: list[str]) -> list[str]:
        """标准化 tag 列表。"""
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

        这里只处理最显眼的“否定方向相反”冲突，
        更细的语义判断仍然交给 verifier 模型负责。
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
        }
        left_has_neg = any(marker in left_text for marker in neg_markers)
        right_has_neg = any(marker in right_text for marker in neg_markers)
        return left_has_neg != right_has_neg
