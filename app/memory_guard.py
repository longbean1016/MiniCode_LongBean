from __future__ import annotations

from dataclasses import dataclass

from app.memory_store import MemoryEntry


@dataclass(slots=True)
class MemoryWriteDecision:
    """
    一条候选记忆的快速门禁结果。

    - `should_store`: 是否允许进入下一阶段
    - `reason`: 拦截原因，便于日志和排查
    """

    should_store: bool
    reason: str = ""


class MemoryWriteGuard:
    """
    写入前保护层。

    这一层只做“快速门禁”，不再负责复杂的 duplicate / conflict 判定。
    它的职责是尽快挡掉明显不该写入的候选，减少后续 verifier 的负担。

    当前保留的检查项：
    1. confidence 过低不写
    2. 空内容不写
    3. 非 project scope 不写
    4. 明显低价值、过程性、礼貌性内容不写
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.78,
    ) -> None:
        self.min_confidence = min_confidence

    def should_store(self, candidate: MemoryEntry) -> MemoryWriteDecision:
        """
        判断候选记忆是否应该进入 verifier 阶段。

        注意：
        - 这里只做快速拦截
        - duplicate / conflict 交给 `MemoryVerifier`
        """
        confidence = self._get_confidence(candidate)
        if confidence < self.min_confidence:
            return MemoryWriteDecision(
                should_store=False,
                reason=f"confidence 太低: {confidence:.2f}",
            )

        normalized_candidate = self._normalize_text(candidate.content)
        if not normalized_candidate:
            return MemoryWriteDecision(
                should_store=False,
                reason="空内容",
            )

        if str(candidate.extra.get("scope", "")).strip().lower() != "project":
            return MemoryWriteDecision(
                should_store=False,
                reason="自动 reflection 只允许写 project scope",
            )

        if self._looks_low_value(candidate):
            return MemoryWriteDecision(
                should_store=False,
                reason="内容更像过程性说明或低价值回复",
            )

        return MemoryWriteDecision(should_store=True)

    def _get_confidence(self, entry: MemoryEntry) -> float:
        """从 `entry.extra` 中读取 confidence。"""
        raw_value = entry.extra.get("confidence", 0.0)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 0.0

    def _looks_low_value(self, entry: MemoryEntry) -> bool:
        """
        用本地规则过滤明显低价值候选。

        即使 confidence 勉强达标，只要内容明显像过程播报、临时说明、
        或礼貌性回复，也不应该进入长期记忆。
        """
        content = self._normalize_text(entry.content)
        category = self._normalize_text(entry.category)

        low_value_markers = {
            "本轮",
            "这一轮",
            "刚刚",
            "临时",
            "暂时",
            "稍后",
            "马上",
            "好的",
            "收到",
            "没问题",
            "我来帮你",
            "just now",
            "temporarily",
            "for now",
        }
        if any(marker.lower() in content for marker in low_value_markers):
            return True

        # 过短且不像规则/结论/失败经验的内容，通常不值得长期保留。
        if len(content) < 18 and category not in {"failure", "conclusion"}:
            return True

        return False

    def _normalize_text(self, text: str) -> str:
        """标准化文本，便于本地规则判断。"""
        return " ".join(text.strip().lower().split())
