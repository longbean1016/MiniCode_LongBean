from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from app.memory_store import MemoryEntry, MemoryStore


@dataclass(slots=True)
class DecayChange:
    """
    一条长期记忆在 decay 阶段发生的变化。

    字段说明：
    - `memory_id`: 被处理的记忆 id
    - `old_decay_score`: 更新前的 decay 分数
    - `new_decay_score`: 更新后的 decay 分数
    - `archived`: 本次是否顺手归档
    - `reason`: 变化原因，便于日志和排查
    """

    memory_id: str
    old_decay_score: float
    new_decay_score: float
    archived: bool = False
    reason: str = ""


@dataclass(slots=True)
class DecayRunResult:
    """
    一次 decay 刷新的结果。

    - `scanned_count`: 扫描了多少条记忆
    - `changed_count`: 实际更新了多少条
    - `changes`: 具体变化明细
    """

    scanned_count: int = 0
    changed_count: int = 0
    changes: list[DecayChange] = field(default_factory=list)

    def archived_count(self) -> int:
        """返回本次 decay 里被归档的记忆数量。"""
        return sum(1 for item in self.changes if item.archived)


class MemoryDecay:
    """
    长期记忆衰减器。

    当前阶段只做两件事：
    1. 周期性刷新 active project 记忆的 `decay_score`
    2. 把明显过时、长期未用、且价值偏低的记忆归档出主检索面

    设计目标：
    - 不删除历史记忆，只做降权和归档
    - 高频使用、近期访问、置信度高的记忆衰减更慢
    - 已被 curator 判成 superseded / duplicate 的记忆不在这里重复处理
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        full_scan_trigger_count: int = 40,
        min_decay_score: float = 0.05,
        archive_decay_threshold: float = 0.12,
        archive_age_days: float = 45.0,
        archive_confidence_threshold: float = 0.72,
        archive_usage_threshold: int = 1,
    ) -> None:
        self.memory_store = memory_store
        self.full_scan_trigger_count = max(10, full_scan_trigger_count)
        self.min_decay_score = max(0.0, min(1.0, float(min_decay_score)))
        self.archive_decay_threshold = max(0.0, min(1.0, float(archive_decay_threshold)))
        self.archive_age_days = max(1.0, float(archive_age_days))
        self.archive_confidence_threshold = max(
            0.0,
            min(1.0, float(archive_confidence_threshold)),
        )
        self.archive_usage_threshold = max(0, int(archive_usage_threshold))

    def refresh_new_entries(self, new_entries: list[MemoryEntry]) -> DecayRunResult:
        """
        围绕本次刚写入或刚整理过的记忆，做一次轻量 decay 刷新。

        当前实现会把这些记忆所在的 active project 集合重新计算一次 decay，
        这样新写入、刚 supersede、刚被访问过的记忆可以尽快反映到排序分数里。
        """
        candidate_ids = {entry.id for entry in new_entries if entry.id.strip()}
        if not candidate_ids:
            return DecayRunResult()

        active_project_entries = self.memory_store.filter_memories(
            scope="project",
            include_archived=False,
        )
        target_entries = [
            entry
            for entry in active_project_entries
            if entry.id in candidate_ids or self._shares_topic_with_new_entries(entry, new_entries)
        ]
        return self._refresh_entries(target_entries)

    def refresh_project_memories(self) -> DecayRunResult:
        """
        对当前全部 active project 记忆做一次全量 decay 刷新。

        这个方法适合低频调用，例如：
        - active 记忆总数达到某个阈值时
        - curator 完成一轮增量整理之后
        """
        active_project_entries = self.memory_store.filter_memories(
            scope="project",
            include_archived=False,
        )
        return self._refresh_entries(active_project_entries)

    def should_run_full_refresh(self) -> bool:
        """
        判断是否适合触发一次低频全量 decay 刷新。

        当前策略与 curator 保持一致：
        - active project 记忆数量达到阈值才考虑
        - 只有在阈值整数倍时才触发，避免每次都扫全表
        """
        active_project_entries = self.memory_store.filter_memories(
            scope="project",
            include_archived=False,
        )
        active_count = len(active_project_entries)
        if active_count < self.full_scan_trigger_count:
            return False
        return active_count % self.full_scan_trigger_count == 0

    def _refresh_entries(self, entries: list[MemoryEntry]) -> DecayRunResult:
        """刷新一组记忆的 decay 分数，并按规则决定是否归档。"""
        if not entries:
            return DecayRunResult()

        all_entries = self.memory_store.load_memories()
        by_id = {entry.id: entry for entry in all_entries}
        now = time.time()
        changed_ids: set[str] = set()
        result = DecayRunResult(scanned_count=len(entries))

        for entry in entries:
            stored_entry = by_id.get(entry.id)
            if stored_entry is None or stored_entry.archived:
                continue

            old_decay_score = float(stored_entry.decay_score)
            new_decay_score = self._compute_decay_score(stored_entry, now=now)
            archived = False
            reason_parts: list[str] = []

            if abs(new_decay_score - old_decay_score) >= 0.01:
                stored_entry.decay_score = new_decay_score
                reason_parts.append("刷新 decay_score")
                changed_ids.add(stored_entry.id)

            if self._should_archive_as_stale(stored_entry, now=now, decay_score=new_decay_score):
                stored_entry.archived = True
                stored_entry.updated_at = now
                stored_entry.extra["archived_reason"] = "decay_stale"
                stored_entry.extra["archived_by"] = "memory_decay"
                archived = True
                reason_parts.append("长期未使用且价值偏低，归档出主检索面")
                changed_ids.add(stored_entry.id)

            if reason_parts:
                result.changes.append(
                    DecayChange(
                        memory_id=stored_entry.id,
                        old_decay_score=old_decay_score,
                        new_decay_score=float(stored_entry.decay_score),
                        archived=archived,
                        reason="；".join(reason_parts),
                    )
                )

        if changed_ids:
            self.memory_store.save_memories_and_sync(
                list(by_id.values()),
                changed_entry_ids=sorted(changed_ids),
            )
            result.changed_count = len(changed_ids)

        return result

    def _compute_decay_score(self, entry: MemoryEntry, *, now: float) -> float:
        """
        计算一条 active 记忆当前的 decay 分数。

        当前分数由四部分组成：
        1. `freshness_score`: 最近是否被更新过
        2. `activity_score`: 最近是否被真正访问过
        3. `usage_bonus`: 长期是否多次被命中使用
        4. `confidence_bonus`: 这条记忆本身是否稳定可信

        结果始终限制在 `[min_decay_score, 1.0]`，避免完全掉到 0。
        """
        age_days = max(0.0, (now - float(entry.updated_at)) / 86400.0)
        last_active_at = float(entry.last_accessed_at) if entry.last_accessed_at > 0 else float(entry.updated_at)
        idle_days = max(0.0, (now - last_active_at) / 86400.0)

        freshness_score = math.exp(-age_days / 45.0)
        activity_score = math.exp(-idle_days / 30.0)
        usage_bonus = min(0.24, math.log1p(max(0, int(entry.usage_count))) * 0.08)
        confidence_bonus = max(0.0, min(1.0, float(entry.confidence))) * 0.18

        raw_score = (
            freshness_score * 0.42
            + activity_score * 0.25
            + usage_bonus
            + confidence_bonus
        )
        return max(self.min_decay_score, min(1.0, raw_score))

    def _should_archive_as_stale(
        self,
        entry: MemoryEntry,
        *,
        now: float,
        decay_score: float,
    ) -> bool:
        """
        判断一条 active 记忆是否已经低价值到应该归档。

        这里保持保守：
        - decay 很低
        - 足够久没有更新
        - 使用次数少
        - confidence 不高
        - 没有被显式标记为当前有效版本
        """
        if entry.extra.get("write_action") == "supersede_store":
            return False
        if entry.extra.get("superseded_by"):
            return False

        age_days = max(0.0, (now - float(entry.updated_at)) / 86400.0)
        if age_days < self.archive_age_days:
            return False
        if decay_score > self.archive_decay_threshold:
            return False
        if int(entry.usage_count) > self.archive_usage_threshold:
            return False
        if float(entry.confidence) >= self.archive_confidence_threshold:
            return False

        return True

    def _shares_topic_with_new_entries(
        self,
        candidate: MemoryEntry,
        new_entries: list[MemoryEntry],
    ) -> bool:
        """
        判断一条旧记忆是否和本次新写入记忆属于相近主题。

        这里只做非常轻量的 topic 扩散：
        - category 相同
        - tags 有交集
        - domains 有交集
        这样 `refresh_new_entries()` 时不必每次全表刷新，也能覆盖相邻主题。
        """
        candidate_tags = {item.strip().lower() for item in candidate.tags if item.strip()}
        candidate_domains = {item.strip().lower() for item in candidate.domains if item.strip()}
        candidate_category = candidate.category.strip().lower()

        for entry in new_entries:
            if candidate.id == entry.id:
                return True

            entry_tags = {item.strip().lower() for item in entry.tags if item.strip()}
            entry_domains = {item.strip().lower() for item in entry.domains if item.strip()}
            entry_category = entry.category.strip().lower()

            if candidate_category and candidate_category == entry_category:
                return True
            if candidate_tags and entry_tags and candidate_tags & entry_tags:
                return True
            if candidate_domains and entry_domains and candidate_domains & entry_domains:
                return True

        return False
