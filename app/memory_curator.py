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
    - `reason`: 归档/合并原因，方便日志排查
    - `details`: 补充结构化细节，便于后续观察链路收口是否正确
    """

    action: str
    target_memory_id: str
    related_memory_id: str = ""
    reason: str = ""
    details: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CuratorRunResult:
    """
    一次 curator 执行的结果。

    - `scanned_count`: 本次扫描过多少条候选关系
    - `changed_count`: 本次实际修改了多少条记忆
    - `governed_count`: 本次直接治理了多少条目标记忆
    - `clustered_count`: 本次进入“同主题簇治理”的种子记忆数
    - `rewired_count`: 本次把多少条旧链路重定向到了当前有效版本
    - `stats`: 按动作类型汇总的计数，方便日志聚合
    - `changes`: 每一条改动的明细
    """

    scanned_count: int = 0
    changed_count: int = 0
    governed_count: int = 0
    clustered_count: int = 0
    rewired_count: int = 0
    stats: dict[str, int] = field(default_factory=dict)
    changes: list[CuratorChange] = field(default_factory=list)


class MemoryCurator:
    """
    长期记忆整理器。

    当前策略仍然保持轻量，但补了一层“长期记忆治理”语义：
    1. 不只归档一条旧记忆，还会尽量把 duplicate / supersede 链收口到当前有效版本
    2. 不删除记忆，只通过 `archived=True` 把旧版本降出主检索面
    3. 不改 user / local scope，只治理 project scope
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

        主链路仍然保持不变：
        - 先加载全量记忆
        - 再对新写入记忆做局部邻域扫描

        这里新增的治理能力主要有两点：
        - 扫描前先把候选记忆解析到“当前有效版本”
        - 发生 duplicate / supersede 后，顺手把旧链路重定向到新的有效版本
        """
        all_entries = self.memory_store.load_memories()
        by_id = {entry.id: entry for entry in all_entries}
        changed_ids: set[str] = set()
        processed_pairs: set[tuple[str, str]] = set()
        result = CuratorRunResult()

        for new_entry in new_entries:
            current_entry = by_id.get(new_entry.id)
            current_entry = self._resolve_effective_entry(current_entry, by_id)
            if current_entry is None or current_entry.archived:
                continue

            result.clustered_count += 1
            current_entry = self._apply_explicit_supersede(
                current_entry,
                by_id,
                processed_pairs,
                changed_ids,
                result,
            )
            if current_entry is None:
                continue

            self._govern_topic_cluster(
                current_entry,
                by_id,
                processed_pairs,
                changed_ids,
                result,
            )

        if changed_ids:
            self.memory_store.save_memories_and_sync(
                list(by_id.values()),
                changed_entry_ids=sorted(changed_ids),
            )
            result.changed_count = len(changed_ids)

        return result

    def curate_project_memories(self) -> CuratorRunResult:
        """
        对当前 active project 记忆做一次全量整理。

        当前仍然通过 recent seed 触发，而不是完全重写主流程，
        这样可以在成本可控的前提下把近期活跃主题优先治理干净。
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

        当前策略保持保守：
        - 只有 active project 记忆达到阈值后才考虑
        - 只有在阈值的整数倍时才触发
        """
        active_project_entries = self.memory_store.filter_memories(
            scope="project",
            include_archived=False,
        )
        active_count = len(active_project_entries)
        if active_count < self.full_scan_trigger_count:
            return False
        return active_count % self.full_scan_trigger_count == 0

    def _apply_explicit_supersede(
        self,
        current_entry: MemoryEntry,
        by_id: dict[str, MemoryEntry],
        processed_pairs: set[tuple[str, str]],
        changed_ids: set[str],
        result: CuratorRunResult,
    ) -> MemoryEntry | None:
        """
        优先处理写入阶段已经明确给出的 supersede 关系。

        这里仍然只治理一对关系，但会先把目标解析到当前有效版本，
        避免上游挂到“已经退位的中间版本”上。
        """
        explicit_target_id = str(current_entry.extra.get("supersedes_memory_id", "")).strip()
        if not explicit_target_id:
            return current_entry

        explicit_target = self._resolve_effective_entry(by_id.get(explicit_target_id), by_id)
        if (
            explicit_target is None
            or explicit_target.archived
            or explicit_target.id == current_entry.id
        ):
            return current_entry

        pair_key = self._make_pair_key(current_entry.id, explicit_target.id)
        if pair_key in processed_pairs:
            return self._resolve_effective_entry(by_id.get(current_entry.id), by_id)

        processed_pairs.add(pair_key)
        result.scanned_count += 1
        relation_details = self._build_relation_details(
            current_entry,
            explicit_target,
            cluster_anchor=current_entry,
        )
        if self._archive_as_superseded(
            explicit_target,
            current_entry,
            by_id,
            changed_ids,
            result,
            reason="写入阶段已显式声明旧版本被当前记忆替代",
            reason_code="explicit_supersede",
            extra_details=relation_details,
        ):
            by_id[explicit_target.id] = explicit_target

        return self._resolve_effective_entry(by_id.get(current_entry.id), by_id)

    def _govern_topic_cluster(
        self,
        seed_entry: MemoryEntry,
        by_id: dict[str, MemoryEntry],
        processed_pairs: set[tuple[str, str]],
        changed_ids: set[str],
        result: CuratorRunResult,
    ) -> MemoryEntry | None:
        """
        以一条种子记忆为中心，尽量把同主题邻域收敛到一个当前有效版本。

        这里不是全库聚类，只是在现有“邻域扫描”主链路上多做几轮：
        - 如果当前 seed 赢了某个 duplicate/supersede，对同主题其它邻居再扫一轮
        - 如果当前 seed 自己被更强版本覆盖，就切到新的有效版本继续收口
        """
        anchor_id = seed_entry.id

        while True:
            live_anchor = self._resolve_effective_entry(by_id.get(anchor_id), by_id)
            if live_anchor is None or live_anchor.archived:
                return None

            cluster_changed = False
            neighbors = self._find_related_entries(live_anchor)
            for other_entry in neighbors:
                live_anchor = self._resolve_effective_entry(by_id.get(anchor_id), by_id)
                live_other = self._resolve_effective_entry(by_id.get(other_entry.id), by_id)
                if live_anchor is None or live_other is None:
                    continue
                if live_anchor.archived or live_other.archived:
                    continue
                if live_anchor.id == live_other.id:
                    continue

                pair_key = self._make_pair_key(live_anchor.id, live_other.id)
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                result.scanned_count += 1
                relation = self._decide_relation(live_anchor, live_other)
                relation_details = self._build_relation_details(
                    live_anchor,
                    live_other,
                    cluster_anchor=live_anchor,
                )

                if relation == "duplicate":
                    # duplicate 的治理目标是让整个主题只保留信息更强的一条当前版本。
                    winner, loser = self._pick_better_entry(live_anchor, live_other)
                    if self._archive_as_duplicate(
                        loser,
                        winner,
                        by_id,
                        changed_ids,
                        result,
                        reason="同主题近重复记忆归档到更强版本",
                        reason_code="neighbor_duplicate",
                        extra_details=relation_details,
                    ):
                        by_id[loser.id] = loser
                        cluster_changed = True
                    anchor_id = winner.id
                    if loser.id == live_anchor.id:
                        break
                    continue

                if relation == "supersede_existing":
                    if self._archive_as_superseded(
                        live_other,
                        live_anchor,
                        by_id,
                        changed_ids,
                        result,
                        reason="当前记忆是同主题的更新规范或更新结论",
                        reason_code="neighbor_supersede_existing",
                        extra_details=relation_details,
                    ):
                        by_id[live_other.id] = live_other
                        cluster_changed = True
                    continue

                if relation == "superseded_by_existing":
                    if self._archive_as_superseded(
                        live_anchor,
                        live_other,
                        by_id,
                        changed_ids,
                        result,
                        reason="当前记忆已被现有更稳定版本覆盖",
                        reason_code="neighbor_superseded_by_existing",
                        extra_details=relation_details,
                    ):
                        by_id[live_anchor.id] = live_anchor
                        cluster_changed = True
                    anchor_id = live_other.id
                    break

            if not cluster_changed:
                return self._resolve_effective_entry(by_id.get(anchor_id), by_id)

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

        当前采用“先保守判重，再保守判替代”的策略：
        - 只有高度接近时才归为 duplicate
        - 只有看起来是同主题的新规范/新结论时才归为 supersede
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

        这里故意保持保守，避免误归档：
        1. 两条内容要有足够主题接近度
        2. newer 质量不能明显弱于 older
        3. newer 文本里要带明显的“统一改为 / 只允许 / 规范化”语气
        """
        similarity = self._jaccard_similarity(newer.content, older.content)
        if similarity < 0.35:
            return False

        newer_score = self._quality_score(newer)
        older_score = self._quality_score(older)
        if newer_score + 0.05 < older_score:
            return False

        newer_text = self._normalize_text(newer.content)
        override_markers = (
            "统一",
            "改为",
            "只允许",
            "固定为",
            "以后",
            "必须",
            "采用",
            "保留",
            "唯一",
            "默认",
            "change to",
            "switch to",
            "migrate to",
            "replace with",
            "use only",
            "no longer",
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
        by_id: dict[str, MemoryEntry],
        changed_ids: set[str],
        result: CuratorRunResult,
        *,
        reason: str,
        reason_code: str,
        extra_details: dict[str, str] | None = None,
    ) -> bool:
        """把重复记忆归档，并把链路收口到当前有效版本。"""
        winner = self._resolve_effective_entry(winner, by_id)
        if loser.archived or winner is None or winner.id == loser.id:
            return False

        now = time.time()
        loser.archived = True
        loser.updated_at = now
        loser.extra["merged_into"] = winner.id
        loser.extra["effective_memory_id"] = winner.id
        loser.extra["archived_reason"] = "duplicate"
        loser.extra["governed_by"] = "memory_curator"
        loser.extra["governance_reason_code"] = reason_code
        loser.extra["governance_topic_signature"] = self._topic_signature(winner)
        loser.extra["is_current_version"] = False
        changed_ids.add(loser.id)

        self._promote_as_current_version(winner, changed_ids)
        rewired_count = self._rewire_memory_links(
            source_id=loser.id,
            target_id=winner.id,
            by_id=by_id,
            changed_ids=changed_ids,
        )

        details = dict(extra_details or {})
        details["reason_code"] = reason_code
        details["effective_memory_id"] = winner.id
        details["rewired_count"] = str(rewired_count)
        self._record_change(
            result,
            CuratorChange(
                action="archive_duplicate",
                target_memory_id=loser.id,
                related_memory_id=winner.id,
                reason=reason,
                details=details,
            ),
        )
        result.governed_count += 1
        result.rewired_count += rewired_count
        return True

    def _archive_as_superseded(
        self,
        loser: MemoryEntry,
        winner: MemoryEntry,
        by_id: dict[str, MemoryEntry],
        changed_ids: set[str],
        result: CuratorRunResult,
        *,
        reason: str,
        reason_code: str,
        extra_details: dict[str, str] | None = None,
    ) -> bool:
        """把旧版本记忆归档，并把版本链收口到当前有效版本。"""
        winner = self._resolve_effective_entry(winner, by_id)
        if loser.archived or winner is None or winner.id == loser.id:
            return False

        now = time.time()
        loser.archived = True
        loser.updated_at = now
        loser.extra["superseded_by"] = winner.id
        loser.extra["effective_memory_id"] = winner.id
        loser.extra["archived_reason"] = "superseded"
        loser.extra["governed_by"] = "memory_curator"
        loser.extra["governance_reason_code"] = reason_code
        loser.extra["governance_topic_signature"] = self._topic_signature(winner)
        loser.extra["is_current_version"] = False
        changed_ids.add(loser.id)

        self._promote_as_current_version(winner, changed_ids)
        rewired_count = self._rewire_memory_links(
            source_id=loser.id,
            target_id=winner.id,
            by_id=by_id,
            changed_ids=changed_ids,
        )

        details = dict(extra_details or {})
        details["reason_code"] = reason_code
        details["effective_memory_id"] = winner.id
        details["rewired_count"] = str(rewired_count)
        self._record_change(
            result,
            CuratorChange(
                action="archive_superseded",
                target_memory_id=loser.id,
                related_memory_id=winner.id,
                reason=reason,
                details=details,
            ),
        )
        result.governed_count += 1
        result.rewired_count += rewired_count
        return True

    def _resolve_effective_entry(
        self,
        entry: MemoryEntry | None,
        by_id: dict[str, MemoryEntry],
    ) -> MemoryEntry | None:
        """
        沿着 merged_into / superseded_by 链追到当前有效版本。

        这样做的目的是避免以下情况：
        - 新记忆把关系挂到一个已经被归档的中间节点上
        - duplicate / supersede 链已经形成多跳，但整理时仍在拿旧节点做判断
        """
        if entry is None:
            return None

        current = entry
        seen_ids = {current.id}
        while True:
            next_id = self._next_effective_memory_id(current)
            if not next_id:
                return current

            next_entry = by_id.get(next_id)
            if next_entry is None or next_entry.id in seen_ids:
                return current

            current = next_entry
            seen_ids.add(current.id)

    def _next_effective_memory_id(self, entry: MemoryEntry) -> str:
        """读取一条记忆指向的下一跳有效版本。"""
        for key in ("merged_into", "superseded_by"):
            target_id = str(entry.extra.get(key, "")).strip()
            if target_id:
                return target_id
        return ""

    def _promote_as_current_version(
        self,
        entry: MemoryEntry,
        changed_ids: set[str],
    ) -> None:
        """
        给当前有效版本补稳定标记。

        这里不额外新增复杂状态机，只维护几个后续排查最有价值的字段：
        - `effective_memory_id`
        - `is_current_version`
        """
        changed = False
        if str(entry.extra.get("effective_memory_id", "")).strip() != entry.id:
            entry.extra["effective_memory_id"] = entry.id
            changed = True

        if entry.extra.get("is_current_version") is not True:
            entry.extra["is_current_version"] = True
            changed = True

        if str(entry.extra.get("governed_by", "")).strip() != "memory_curator":
            entry.extra["governed_by"] = "memory_curator"
            changed = True

        topic_signature = self._topic_signature(entry)
        if str(entry.extra.get("governance_topic_signature", "")).strip() != topic_signature:
            entry.extra["governance_topic_signature"] = topic_signature
            changed = True

        # 活跃版本不应该再挂旧的被替代关系；这里只清理最容易误导排查的字段。
        for stale_key in ("merged_into", "superseded_by"):
            if str(entry.extra.get(stale_key, "")).strip():
                entry.extra.pop(stale_key, None)
                changed = True

        if changed:
            entry.updated_at = time.time()
            changed_ids.add(entry.id)

    def _rewire_memory_links(
        self,
        *,
        source_id: str,
        target_id: str,
        by_id: dict[str, MemoryEntry],
        changed_ids: set[str],
    ) -> int:
        """
        把所有仍然指向旧节点 `source_id` 的链路改写到 `target_id`。

        这一步是这次改动的核心治理能力：
        - duplicate 链不会一直挂在被归档的中间节点上
        - supersede 链会自然收口到当前有效版本
        - 后续日志只看 `effective_memory_id` 就能更快定位真实生效版本
        """
        if not source_id or not target_id or source_id == target_id:
            return 0

        rewired_count = 0
        now = time.time()
        link_keys = (
            "merged_into",
            "superseded_by",
            "supersedes_memory_id",
            "effective_memory_id",
        )

        for entry in by_id.values():
            if entry.id in {source_id, target_id}:
                continue

            changed = False
            for key in link_keys:
                current_target = str(entry.extra.get(key, "")).strip()
                if current_target != source_id:
                    continue
                entry.extra[key] = target_id
                changed = True

            if not changed:
                continue

            entry.updated_at = now
            changed_ids.add(entry.id)
            rewired_count += 1

        return rewired_count

    def _record_change(self, result: CuratorRunResult, change: CuratorChange) -> None:
        """记录结构化变更，并同步更新动作计数。"""
        result.changes.append(change)
        result.stats[change.action] = result.stats.get(change.action, 0) + 1

    def _build_relation_details(
        self,
        left: MemoryEntry,
        right: MemoryEntry,
        *,
        cluster_anchor: MemoryEntry | None = None,
    ) -> dict[str, str]:
        """
        给 duplicate / supersede 判定补一份轻量结构化快照。

        这些细节不会影响主逻辑，但对日志排查很有帮助：
        - 两侧相似度
        - tag/domain 重叠情况
        - 质量分对比
        """
        details = {
            "content_similarity": f"{self._jaccard_similarity(left.content, right.content):.4f}",
            "tag_overlap": f"{self._overlap_ratio(left.tags, right.tags):.4f}",
            "domain_overlap": f"{self._overlap_ratio(left.domains, right.domains):.4f}",
            "left_quality": f"{self._quality_score(left):.4f}",
            "right_quality": f"{self._quality_score(right):.4f}",
            "left_usage_count": str(left.usage_count),
            "right_usage_count": str(right.usage_count),
            "left_current_version": str(left.extra.get("is_current_version") is True).lower(),
            "right_current_version": str(right.extra.get("is_current_version") is True).lower(),
            "left_write_action": str(left.extra.get("write_action", "")).strip(),
            "right_write_action": str(right.extra.get("write_action", "")).strip(),
            "left_topic_signature": self._topic_signature(left),
            "right_topic_signature": self._topic_signature(right),
            "same_category": str(
                self._normalize_text(left.category) == self._normalize_text(right.category)
            ).lower(),
        }
        if cluster_anchor is not None:
            # cluster_anchor 表示这一轮簇治理当前围绕的“有效中心版本”。
            details["cluster_anchor_id"] = cluster_anchor.id
            details["cluster_topic_signature"] = self._topic_signature(cluster_anchor)
        return details

    def _topic_signature(self, entry: MemoryEntry) -> str:
        """
        生成一个轻量主题签名，便于日志里快速看出“这次治理在收哪个主题簇”。

        这里不追求全局唯一，只追求排查时可读：
        - category 给出主题大类
        - domains/tags 给出业务上下文
        """
        category = self._normalize_text(entry.category) or "uncategorized"
        domains = sorted(
            {
                self._normalize_text(domain)
                for domain in entry.domains
                if self._normalize_text(domain)
            }
        )
        tags = sorted(
            {self._normalize_text(tag) for tag in entry.tags if self._normalize_text(tag)}
        )
        parts = [f"category={category}"]
        if domains:
            parts.append(f"domains={','.join(domains[:3])}")
        if tags:
            parts.append(f"tags={','.join(tags[:4])}")
        return "|".join(parts)

    def _make_pair_key(self, left_id: str, right_id: str) -> tuple[str, str]:
        """把双向关系压成稳定 pair key，避免同一对记忆被重复扫描。"""
        return tuple(sorted((left_id, right_id)))

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
            + float(entry.decay_score) * 0.6
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
        """
        normalized = self._normalize_text(text)
        if not normalized:
            return set()
        return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized))

    def _normalize_text(self, text: str) -> str:
        """标准化文本，便于做本地比较。"""
        return " ".join(str(text).strip().lower().split())
