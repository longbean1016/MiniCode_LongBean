from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

from app.memory_store import MemoryEntry


VerificationAction = Literal["store", "duplicate", "conflict", "reject"]


@dataclass(slots=True)
class MemoryVerificationDecision:
    """
    verifier 的最终判断结果。

    - `action`: 最终动作
    - `reason`: 解释为什么做这个判断
    - `matched_memory_id`: 如果命中了 duplicate / conflict，可带回对应旧记忆 id
    """

    action: VerificationAction
    reason: str = ""
    matched_memory_id: str = ""


class MemoryVerifier:
    """
    长期记忆写入前的二次验证器。

    目标：
    1. 先从已有记忆中召回一小组“主题接近”的候选旧记忆
    2. 再让 verifier 判断新记忆应该：
       - `store`
       - `duplicate`
       - `conflict`
       - `reject`

    当前阶段还没有向量库，
    所以召回层先用轻量文本相似度做候选筛选；
    等后面接入 Qdrant，这一层可以直接替换成向量召回。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        *,
        max_candidates: int = 5,
        min_similarity: float = 0.18,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.min_similarity = min_similarity

    def find_similar_entries(
        self,
        candidate: MemoryEntry,
        existing_entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        """
        从已有长期记忆中找出一小组可能相关的候选旧记忆。

        当前筛选规则比较保守：
        - 只看 `project` scope
        - 优先保留同 category 的记忆
        - 用词面重叠做第一轮近似召回
        """
        candidate_scope = str(candidate.extra.get("scope", "")).strip().lower()
        candidate_category = self._normalize_text(candidate.category)

        scored_entries: list[tuple[float, MemoryEntry]] = []
        for entry in existing_entries:
            existing_scope = str(entry.extra.get("scope", "")).strip().lower()
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

    def verify(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
    ) -> MemoryVerificationDecision:
        """
        对候选记忆做二次验证。

        - 没有近似旧记忆时，直接允许写入
        - 有近似旧记忆时，调用 verifier 模型做二次判断
        - 模型失败时，退回到本地启发式兜底
        """
        if not similar_entries:
            return MemoryVerificationDecision(
                action="store",
                reason="没有找到足够相近的已有记忆",
            )

        model_decision = self._call_verifier_model(candidate, similar_entries)
        if model_decision is not None:
            return model_decision

        return self._fallback_verify(candidate, similar_entries)

    def _call_verifier_model(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
    ) -> MemoryVerificationDecision | None:
        """
        调模型做二次验证。

        当前目标不是让模型“重新抽记忆”，
        而是让它在少量相似旧记忆的上下文里判断：
        - 这条是不是重复
        - 是不是冲突
        - 是不是价值不足应拒绝
        - 否则才允许写入
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
- reject: 本身质量不足、范围太临时、或不值得长期保存

判定原则：
- duplicate: 语义基本相同，只是换了说法、改了措辞、补了少量不重要细节
- conflict: 同一主题上给出了方向相反、约束相反、结论相反的长期记忆
- reject: 就算没有旧记忆冲突，这条内容本身也不够稳定、不够复用、不够长期
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
        当 verifier 模型不可用时，使用本地启发式兜底。

        这不是最终方案，只是为了保证主链路稳定：
        - 明显重复时判 duplicate
        - 明显冲突时判 conflict
        - 否则默认 store
        """
        normalized_candidate = self._normalize_text(candidate.content)
        candidate_category = self._normalize_text(candidate.category)

        for entry in similar_entries:
            normalized_existing = self._normalize_text(entry.content)
            similarity = self._jaccard_similarity(normalized_candidate, normalized_existing)

            if similarity >= 0.92:
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
            f"confidence: {candidate.extra.get('confidence', '')}",
            f"content: {candidate.content}",
            "",
            "## 相似旧记忆",
        ]

        for entry in similar_entries:
            lines.extend(
                [
                    f"- id: {entry.id}",
                    f"  category: {entry.category}",
                    f"  content: {entry.content}",
                ]
            )

        return "\n".join(lines).strip()

    def _parse_json_payload(self, text: str) -> Any:
        """解析模型返回的 JSON，兼容 ```json 代码块。"""
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

        这里只处理最显眼的否定方向冲突，
        更细的语义判断仍由 verifier 模型负责。
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
