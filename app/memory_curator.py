from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from app.memory_store import MemoryEntry, MemoryStore


CuratorRelation = Literal[
    "keep_separate",
    "duplicate",
    "supersede_existing",
    "superseded_by_existing",
]


@dataclass(slots=True)
class CuratorChange:
    """
    curator 对单条记忆做出的整理动作。

    字段说明：
    - `action`: 这次整理的动作类型
    - `target_memory_id`: 被处理的那条记忆 id
    - `related_memory_id`: 与之关联的另一条记忆 id
    - `reason`: 归档/合并原因，方便排查
    """

    action: str
    target_memory_id: str
    related_memory_id: str = ""
    reason: str = ""


@dataclass(slots=True)
class CuratorRunResult:
    """
    一次 curator 执行的结果。

    - `scanned_count`: 本次扫描过多少条候选关系
    - `changed_count`: 本次实际修改了多少条记忆
    - `changes`: 每一条改动的明细
    """

    scanned_count: int = 0
    changed_count: int = 0
    changes: list[CuratorChange] = field(default_factory=list)


class MemoryCurator:
    """
    长期记忆整理器。

    当前阶段只做 project scope 的轻量整理：
    1. 合并近重复记忆
    2. 归档被新结论替代的旧记忆
    3. 不删除记忆，只通过 `archived=True` 降出主检索面

    设计目标：
    - 不影响主链路回答
    - 不碰 user / local scope
    - 尽量复用已有 semantic search 能力，而不是重新造一套检索
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        related_top_k: int = 6,
        full_scan_trigger_count: int = 40,
    ) -> None:
        self.memory_store = memory_store
        self.related_top_k = max(2, related_top_k)
        self.full_scan_trigger_count = max(10, full_scan_trigger_count)

    def curate_new_entries(self, new_entries: list[MemoryEntry]) -> CuratorRunResult:
        """
        只围绕“本次刚写入的记忆”做增量整理。

        这是主链路默认调用的方法，因为：
        - 成本低
        - 能尽快收敛同主题重复记忆
        - 不需要每轮都全表扫描
        """
        all_entries = self.memory_store.load_memories()
        by_id = {entry.id: entry for entry in all_entries}
        changed_ids: set[str] = set()
        result = CuratorRunResult()

        for new_entry in new_entries:
            current_entry = by_id.get(new_entry.id)
            if current_entry is None or current_entry.archived:
                continue

            # 优先处理“写入阶段已经明确指出替代谁”的更新关系。
            # 这是更接近 minicode 的做法：
            # 新记忆先入库成为当前有效版本，
            # 然后旧记忆再被显式降级为 superseded。
            explicit_target_id = str(
                current_entry.extra.get("supersedes_memory_id", "")
            ).strip()
            if explicit_target_id:
                explicit_target = by_id.get(explicit_target_id)
                if (
                    explicit_target is not None
                    and not explicit_target.archived
                    and explicit_target.id != current_entry.id
                ):
                    result.scanned_count += 1
                    if self._archive_as_superseded(
                        explicit_target,
                        current_entry,
                        changed_ids,
                        result,
                    ):
                        by_id[explicit_target.id] = explicit_target

            neighbors = self._find_related_entries(current_entry)
            for other_entry in neighbors:
                if other_entry.id == current_entry.id:
                    continue

                live_current = by_id.get(current_entry.id)
                live_other = by_id.get(other_entry.id)
                if live_current is None or live_other is None:
                    continue
                if live_current.archived or live_other.archived:
                    continue

                result.scanned_count += 1
                relation = self._decide_relation(live_current, live_other)

                if relation == "duplicate":
                    winner, loser = self._pick_better_entry(live_current, live_other)
                    if self._archive_as_duplicate(loser, winner, changed_ids, result):
                        by_id[loser.id] = loser
                    if loser.id == current_entry.id:
                        break
                    continue

                if relation == "supersede_existing":
                    if self._archive_as_superseded(
                        live_other,
                        live_current,
                        changed_ids,
                        result,
                    ):
                        by_id[live_other.id] = live_other
                    continue

                if relation == "superseded_by_existing":
                    if self._archive_as_superseded(
                        live_current,
                        live_other,
                        changed_ids,
                        result,
                    ):
                        by_id[live_current.id] = live_current
                    break

        if changed_ids:
            self.memory_store.save_memories_and_sync(
                list(by_id.values()),
                changed_entry_ids=sorted(changed_ids),
            )
            result.changed_count = len(changed_ids)

        return result

    def curate_project_memories(self) -> CuratorRunResult:
        """
        对当前全部 active project 记忆做一次全量整理。

        当前主链路不强制每轮调用它，
        但后面可以在：
        - 定时任务
        - 命令入口
        - 写入数达到阈值
        时触发。
        """
        active_entries = self.memory_store.filter_memories(
            scope="project",
            include_archived=False,
        )
        recent_entries = sorted(
            active_entries,
            key=lambda entry: entry.updated_at,
            reverse=True,
        )
        seed_entries = recent_entries[: self.full_scan_trigger_count]
        return self.curate_new_entries(seed_entries)

    def should_run_full_scan(self) -> bool:
        """
        判断当前是否适合触发一次 project 级全量整理。

        当前策略比较保守：
        - 只有 active project 记忆达到阈值后才考虑
        - 只有在阈值的整倍数时才触发
        这样可以避免每次写入都扫全表。
        """
        active_project_entries = self.memory_store.filter_memories(
            scope="project",
            include_archived=False,
        )
        active_count = len(active_project_entries)
        if active_count < self.full_scan_trigger_count:
            return False
        return active_count % self.full_scan_trigger_count == 0

    def _find_related_entries(self, entry: MemoryEntry) -> list[MemoryEntry]:
        """
        为一条记忆找一小组可能相关的 active project 记忆。

        这里直接复用 `memory_store.search_memories()`：
        - 有 Qdrant 时优先语义召回
        - 没有 Qdrant 时回退到词面检索
        """
        query_text = self._build_query_text(entry)
        neighbors = self.memory_store.search_memories(
            query_text,
            top_k=self.related_top_k + 2,
            scope="project",
            category=entry.category or None,
            domains=entry.domains or None,
            include_archived=False,
            mark_access=False,
        )
        return [item for item in neighbors if item.id != entry.id]

    def _decide_relation(
        self,
        candidate: MemoryEntry,
        existing: MemoryEntry,
    ) -> CuratorRelation:
        """
        判断两条 active project 记忆之间的关系。

        当前实现采用“先便宜规则，再保守归档”的策略：
        - 只有高度接近时才判 duplicate
        - 只有看起来是同主题且存在“新规范替代旧规范”时才判 supersede
        - 其余情况全部 keep_separate
        """
        content_similarity = self._jaccard_similarity(candidate.content, existing.content)
        tag_overlap = self._overlap_ratio(candidate.tags, existing.tags)
        domain_overlap = self._overlap_ratio(candidate.domains, existing.domains)
        same_category = self._normalize_text(candidate.category) == self._normalize_text(
            existing.category
        )

        duplicate_score = content_similarity
        if same_category:
            duplicate_score += 0.08
        duplicate_score += min(0.08, tag_overlap * 0.08)
        duplicate_score += min(0.06, domain_overlap * 0.06)

        if duplicate_score >= 0.93:
            return "duplicate"

        if same_category and self._looks_superseding(candidate, existing):
            return "supersede_existing"

        if same_category and self._looks_superseding(existing, candidate):
            return "superseded_by_existing"

        return "keep_separate"

    def _looks_superseding(self, newer: MemoryEntry, older: MemoryEntry) -> bool:
        """
        判断 `newer` 是否像是在替代 `older`。

        这里只做保守判断，避免误归档：
        1. 两条内容要有一定主题接近度
        2. newer 要更“新”或更“稳”
        3. newer 文本里要带有明显的“统一改为 / 只允许 / 规范化”语气
        """
        similarity = self._jaccard_similarity(newer.content, older.content)
        # 同主题规范更新时，关键名词往往会替换掉一部分旧 token，
        # 所以这里比 duplicate 更保守地放宽一点阈值。
        if similarity < 0.35:
            return False

        newer_score = self._quality_score(newer)
        older_score = self._quality_score(older)
        if newer_score + 0.05 < older_score:
            return False

        newer_text = self._normalize_text(newer.content)
        override_markers = (
            "\u7edf\u4e00",
            "\u6539\u4e3a",
            "\u53ea\u5141\u8bb8",
            "\u56fa\u5b9a\u4e3a",
            "\u4ee5\u540e",
            "\u5fc5\u987b",
            "\u91c7\u7528",
            "\u4fdd\u7559",
            "\u552f\u4e00",
            "\u9ed8\u8ba4",
            "change to",
            "switch to",
            "migrate to",
            "replace with",
            "use only",
        )
        if not any(marker in newer_text for marker in override_markers):
            return False

        return newer.updated_at >= older.updated_at or newer.confidence >= older.confidence

    def _pick_better_entry(
        self,
        left: MemoryEntry,
        right: MemoryEntry,
    ) -> tuple[MemoryEntry, MemoryEntry]:
        """
        在两条近重复记忆里选出保留者和归档者。

        优先级：
        1. 更高 confidence
        2. 更高 usage_count
        3. 更近 updated_at
        4. 内容更长，通常信息更完整
        """
        left_score = self._quality_score(left)
        right_score = self._quality_score(right)
        if left_score >= right_score:
            return left, right
        return right, left

    def _archive_as_duplicate(
        self,
        loser: MemoryEntry,
        winner: MemoryEntry,
        changed_ids: set[str],
        result: CuratorRunResult,
    ) -> bool:
        """把重复记忆归档，并记录 merged_into 关系。"""
        if loser.archived:
            return False

        now = time.time()
        loser.archived = True
        loser.updated_at = now
        loser.extra["merged_into"] = winner.id
        loser.extra["archived_reason"] = "duplicate"
        changed_ids.add(loser.id)
        result.changes.append(
            CuratorChange(
                action="archive_duplicate",
                target_memory_id=loser.id,
                related_memory_id=winner.id,
                reason="与已有 project 记忆语义重复，归档较弱版本",
            )
        )
        return True

    def _archive_as_superseded(
        self,
        loser: MemoryEntry,
        winner: MemoryEntry,
        changed_ids: set[str],
        result: CuratorRunResult,
    ) -> bool:
        """把被新规范/新结论替代的旧记忆归档。"""
        if loser.archived:
            return False

        now = time.time()
        loser.archived = True
        loser.updated_at = now
        loser.extra["superseded_by"] = winner.id
        loser.extra["archived_reason"] = "superseded"
        changed_ids.add(loser.id)
        result.changes.append(
            CuratorChange(
                action="archive_superseded",
                target_memory_id=loser.id,
                related_memory_id=winner.id,
                reason="被更新、更稳定的同主题 project 记忆替代",
            )
        )
        return True

    def _build_query_text(self, entry: MemoryEntry) -> str:
        """把记忆条目整理成检索查询文本，供语义召回或词面召回复用。"""
        parts = [
            f"category: {entry.category}",
            f"scope: {entry.scope}",
            f"content: {entry.content}",
        ]
        if entry.tags:
            parts.append(f"tags: {', '.join(entry.tags)}")
        if entry.domains:
            parts.append(f"domains: {', '.join(entry.domains)}")
        return "\n".join(parts).strip()

    def _quality_score(self, entry: MemoryEntry) -> float:
        """给单条记忆打一个内部质量分，用来做 duplicate 保留决策。"""
        age_seconds = max(0.0, time.time() - float(entry.updated_at))
        recent_bonus = max(0.0, 1.0 - age_seconds / (30 * 86400)) * 0.15
        return (
            float(entry.confidence) * 1.2
            + min(0.4, float(entry.usage_count) * 0.05)
            + min(0.2, len(entry.content.strip()) / 300.0)
            + recent_bonus
        )

    def _overlap_ratio(self, left: list[str], right: list[str]) -> float:
        """计算两个标签/领域列表的重合比例。"""
        left_set = {self._normalize_text(item) for item in left if self._normalize_text(item)}
        right_set = {
            self._normalize_text(item) for item in right if self._normalize_text(item)
        }
        if not left_set or not right_set:
            return 0.0

        union = left_set | right_set
        if not union:
            return 0.0

        return len(left_set & right_set) / len(union)

    def _jaccard_similarity(self, left: str, right: str) -> float:
        """计算两段文本的轻量 Jaccard 相似度。"""
        left_tokens = self._tokenize(left)
        right_tokens = self._tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0

        union = left_tokens | right_tokens
        if not union:
            return 0.0

        return len(left_tokens & right_tokens) / len(union)

    def _tokenize(self, text: str) -> set[str]:
        """
        对文本做轻量切词。

        这里不能只按空格切，因为项目里的长期记忆很多是中文句子。
        当前规则：
        - 英文/数字按连续单词切分
        - 中文按单字切分
        这虽然不如真正分词精细，但足够支撑 curator 做保守的相似度判断。
        """
        normalized = self._normalize_text(text)
        if not normalized:
            return set()
        return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized))

    def _normalize_text(self, text: str) -> str:
        """标准化文本，便于做本地比较。"""
        return " ".join(str(text).strip().lower().split())
