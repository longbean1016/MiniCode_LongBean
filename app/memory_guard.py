from __future__ import annotations

from dataclasses import dataclass

from app.explicit_memory import EXPLICIT_MEMORY_SOURCES
from app.memory_store import MemoryEntry


@dataclass(slots=True)
class MemoryWriteDecision:
    should_store: bool
    reason: str = ""


class MemoryWriteGuard:
    def __init__(self, *, min_confidence: float = 0.78) -> None:
        self.min_confidence = min_confidence

    def should_store(self, candidate: MemoryEntry) -> MemoryWriteDecision:
        confidence = self._get_confidence(candidate)
        if confidence < self.min_confidence:
            return MemoryWriteDecision(
                should_store=False,
                reason=f"confidence 过低: {confidence:.2f}",
            )

        normalized_content = self._normalize_text(candidate.content)
        if not normalized_content:
            return MemoryWriteDecision(
                should_store=False,
                reason="空内容",
            )

        if candidate.source.strip().lower() in EXPLICIT_MEMORY_SOURCES:
            return MemoryWriteDecision(should_store=True)

        if candidate.scope.strip().lower() != "project":
            return MemoryWriteDecision(
                should_store=False,
                reason="自动 reflection 只允许写入 project scope",
            )

        if candidate.source.strip().lower() == "task_reflection":
            if candidate.category.strip().lower() not in {"convention", "constraint", "failure"}:
                return MemoryWriteDecision(
                    should_store=False,
                    reason="自动 reflection 只允许写入 convention / constraint / failure",
                )

            if not self._has_reflection_admission(candidate):
                return MemoryWriteDecision(
                    should_store=False,
                    reason="缺少反思准入证据，不允许自动写入 project memory",
                )

        if self._looks_low_value(candidate):
            return MemoryWriteDecision(
                should_store=False,
                reason="内容更像临时过程说明、礼貌回复或一次性结果",
            )

        return MemoryWriteDecision(should_store=True)

    def _get_confidence(self, entry: MemoryEntry) -> float:
        try:
            return float(entry.confidence)
        except (TypeError, ValueError):
            return 0.0

    def _has_reflection_admission(self, entry: MemoryEntry) -> bool:
        admission_source = self._normalize_text(
            entry.extra.get("reflection_admission_source", "")
        )
        return admission_source in {"project_file_evidence", "execution_signal"}

    def _looks_low_value(self, entry: MemoryEntry) -> bool:
        content = self._normalize_text(entry.content)
        category = self._normalize_text(entry.category)

        temporary_markers = {
            "本轮",
            "这一轮",
            "刚刚",
            "临时",
            "暂时",
            "稍后",
            "马上",
            "待会",
            "这次先",
            "先这样",
            "this turn",
            "just now",
            "temporarily",
            "for now",
            "later",
        }
        if any(marker.lower() in content for marker in temporary_markers):
            return True

        low_value_phrases = {
            "好的",
            "收到",
            "明白",
            "没问题",
            "已处理",
            "我来帮你",
            "我可以继续",
            "thanks",
            "thank you",
            "got it",
            "sounds good",
        }
        if any(phrase.lower() in content for phrase in low_value_phrases):
            return True

        if "tmp\\" in content or "tmp/" in content:
            return True

        if len(content) < 18 and category != "failure":
            return True

        return False

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).strip().lower().split())
