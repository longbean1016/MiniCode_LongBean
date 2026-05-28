

from dataclasses import dataclass

from app.memory_store import MemoryEntry


@dataclass(slots=True)
class MemoryWriteDecision:
    """
    一条候选记忆的写入判定结果。

    字段说明：
    - should_store: 是否允许写入
    - reason: 判定原因，便于后续日志和排查
    """

    should_store: bool
    reason: str = ""

class MemoryWriteGuard:
    """
    写入前保护层。

    目标：
    1. 低 confidence 不写
    2. 高相似度的不重复写
    3. 疑似冲突内容先拦住

    “confidence gate + 记忆治理”的核心约束。
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.78,
        dedupe_similarity_threshold: float = 0.90,
        conflict_similarity_threshold: float = 0.72,
    ) -> None:
        
        # min_confidence: 最低置信度门槛。
        # 低于这个值，直接不写入长期记忆。
        self.min_confidence = min_confidence

        # dedupe_similarity_threshold: 去重相似度阈值。
        # 高于这个值，认为和已有记忆太像，不重复写。
        self.dedupe_similarity_threshold = dedupe_similarity_threshold

        # conflict_similarity_threshold: 冲突检测相似度阈值。
        # 高于这个值且语义方向相反，就拦住。
        self.conflict_similarity_threshold = conflict_similarity_threshold

    def should_store(
            self,
            candidate: MemoryEntry,
            existing_entries: list[MemoryEntry]
    )-> MemoryWriteDecision:
        """
        判断候选记忆是否应写入。

        处理顺序：
        1. confidence gate
        2. scope gate
        3. duplicate / similarity gate
        4. conflict gate
        """
        confidence = self._get_confidence(candidate)
        if confidence < self.min_confidence:
            return MemoryWriteDecision(
                should_store=False,
                reason=f"confidence太低: {confidence:.2f}",
            )
        # normalized_candidate: 标准化后的候选记忆正文。
        normalized_candidate = self._normalize_text(candidate.content)
        if not normalized_candidate:
            return MemoryWriteDecision(
                should_store=False,
                reason="空内容",
            )
        # 自动 reflection 只允许写 project scope。
        if str(candidate.extra.get("scope", "")).strip().lower() != "project":
            return MemoryWriteDecision(
                should_store=False,
                reason="自动 reflection 只允许写 project scope",
            )
        
        for existing in existing_entries:
            similarity = self._jaccard_similarity(
                normalized_candidate,
                self._normalize_text(existing.content),
            )

            # 高相似内容直接视为重复。
            if similarity >= self.dedupe_similarity_threshold:
                return MemoryWriteDecision(
                    should_store=False,
                    reason=f"高相似度的记忆: {similarity:.2f}",
                )
            # 如果主题足够相近，且类别相同，再进一步做冲突检测。
            if (
                similarity >= self.conflict_similarity_threshold
                and existing.category == candidate.category
                and self._looks_conflicting(existing.content, candidate.content)
            ):
                return MemoryWriteDecision(
                    should_store=False,
                    reason=f"和存在的长时记忆可能有冲突: {similarity:.2f}",
                )

        return MemoryWriteDecision(should_store=True)
    
    def _get_confidence(self, entry: MemoryEntry) -> float:
        """
        从 entry.extra 中读取置信度。

        约定：
        - confidence 存在 entry.extra["confidence"]
        - 读不到或类型异常时，按 0.0 处理
        """
        raw_value = entry.extra.get("confidence", 0.0)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 0.0
        
    def _looks_conflicting(self, left: str, right: str) -> bool:
        """
        轻量冲突检测。

        第一版用启发式规则：
        如果主题非常相似，但否定倾向相反，就判成疑似冲突。
        """
        # left_text / right_text: 标准化后的两段文本。
        left_text = self._normalize_text(left)
        right_text = self._normalize_text(right)

        # neg_markers: 否定类标记词。
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

        # 一边是否定，一边不是，说明可能存在方向冲突。
        return left_has_neg != right_has_neg
    
    def _normalize_text(self, text: str) -> str:
        """
        标准化文本，便于做相似度比较。

        处理方式：
        - 去掉首尾空格
        - 转小写
        - 压缩多余空白
        """
        return " ".join(text.strip().lower().split())
    
    def _tokenize(self, text: str) -> set[str]:
        """
        按空白做轻量切词。

        第一版先不做复杂分词，
        只为相似度计算提供最基础 token 集。
        """
        normalized = self._normalize_text(text)
        if not normalized:
            return set()
        return set(normalized.split())
    
    def _jaccard_similarity(self, left: str, right: str) -> float:
        """
        计算两段文本的 Jaccard 相似度。

        公式：
        similarity = 交集 token 数 / 并集 token 数
        """
        left_tokens = self._tokenize(left)
        right_tokens = self._tokenize(right)

        if not left_tokens or not right_tokens:
            return 0.0

        union = left_tokens | right_tokens
        if not union:
            return 0.0

        intersection = left_tokens & right_tokens
        return len(intersection) / len(union)