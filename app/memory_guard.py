from __future__ import annotations

from dataclasses import dataclass

from app.memory_store import MemoryEntry


@dataclass(slots=True)
class MemoryWriteDecision:
    """
    一条候选记忆的快速门禁结果。

    字段说明：
    - `should_store`: 是否允许继续进入 verifier 阶段
    - `reason`: 拦截原因，便于日志排查
    """

    should_store: bool
    reason: str = ""


class MemoryWriteGuard:
    """
    长期记忆写入前的快速门禁层。

    这一层不负责复杂的 duplicate / conflict 语义判断，
    只负责尽快挡掉明显不该进入长期记忆的候选内容。

    当前约束重点：
    - 自动 reflection 只允许写入 `project` scope
    - 自动 reflection 只允许写入 `convention / failure`
    - 自动 reflection 要么带有真实项目文件证据，要么带有较强任务执行信号
    - 过程性、临时性、礼貌性表达不进入长期记忆
    """

    def __init__(self, *, min_confidence: float = 0.78) -> None:
        self.min_confidence = min_confidence

    def should_store(self, candidate: MemoryEntry) -> MemoryWriteDecision:
        """判断候选记忆是否允许继续进入 verifier 阶段。"""
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

        if candidate.scope.strip().lower() != "project":
            return MemoryWriteDecision(
                should_store=False,
                reason="自动 reflection 只允许写入 project scope",
            )

        if candidate.source.strip().lower() == "task_reflection":
            if candidate.category.strip().lower() not in {"convention", "failure"}:
                return MemoryWriteDecision(
                    should_store=False,
                    reason="自动 reflection 只允许写入 convention / failure",
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
        """读取候选记忆上的 confidence。"""
        try:
            return float(entry.confidence)
        except (TypeError, ValueError):
            return 0.0

    def _has_reflection_admission(self, entry: MemoryEntry) -> bool:
        """
        判断这条自动反思候选是否具备准入证据。

        当前允许两种来源：
        1. `project_file_evidence`
           这轮任务确实改到了真实项目文件
        2. `execution_signal`
           虽然没有文件触点，但准入层已经认定它像稳定项目执行经验
        """
        admission_source = self._normalize_text(
            entry.extra.get("reflection_admission_source", "")
        )
        return admission_source in {"project_file_evidence", "execution_signal"}

    def _looks_low_value(self, entry: MemoryEntry) -> bool:
        """
        用本地规则过滤明显低价值候选。

        这一层只保留结构化、稳定的规则：
        - 过程播报
        - 临时说明
        - 礼貌性确认
        - 太短且不像失败经验的片段
        """
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
        """标准化文本，便于规则判断。"""
        return " ".join(str(text).strip().lower().split())
