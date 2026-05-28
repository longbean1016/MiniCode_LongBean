from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from app.memory_vector_index import MemoryVectorIndex


MEMORY_DIR_NAME = ".memory"
MEMORY_FILE_NAME = "memory.json"

MemorySortField = Literal["updated_at", "created_at", "last_accessed_at", "usage_count"]


def _normalize_text(text: str) -> str:
    """把文本做基础规范化，便于过滤和检索。"""
    return " ".join(str(text).strip().lower().split())


def _tokenize(text: str) -> list[str]:
    """按空白做轻量切词。"""
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


@dataclass(slots=True)
class MemoryEntry:
    """
    一条长期记忆。

    当前常用 metadata 已经提升为正式字段，
    这样后续做 verifier / curator / decay / retrieval 时，
    不需要把核心信息都塞进 extra 里。
    """

    id: str
    content: str
    category: str = "note"
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    session_id: str = ""

    # 正式 metadata 字段
    scope: str = "project"
    confidence: float = 0.0
    domains: list[str] = field(default_factory=list)
    source: str = ""
    usage_count: int = 0
    last_accessed_at: float = 0.0
    decay_score: float = 1.0
    archived: bool = False

    # 额外信息留在 extra 里，便于兼容和扩展。
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转成可写入 JSON 的普通字典。"""
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "session_id": self.session_id,
            "scope": self.scope,
            "confidence": self.confidence,
            "domains": list(self.domains),
            "source": self.source,
            "usage_count": self.usage_count,
            "last_accessed_at": self.last_accessed_at,
            "decay_score": self.decay_score,
            "archived": self.archived,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """
        从字典恢复一条 MemoryEntry。

        同时兼容：
        1. 新版正式 metadata 字段
        2. 旧版把 metadata 塞在 extra 里的格式
        """
        extra = (
            dict(data.get("extra", {}))
            if isinstance(data.get("extra", {}), dict)
            else {}
        )

        scope = str(data.get("scope", extra.get("scope", "project"))).strip() or "project"

        try:
            confidence = float(data.get("confidence", extra.get("confidence", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0

        raw_domains = data.get("domains", extra.get("domains", []))
        domains = (
            [str(item).strip() for item in raw_domains if str(item).strip()]
            if isinstance(raw_domains, list)
            else []
        )

        source = str(data.get("source", extra.get("source", ""))).strip()

        try:
            usage_count = int(data.get("usage_count", extra.get("usage_count", 0)))
        except (TypeError, ValueError):
            usage_count = 0

        try:
            last_accessed_at = float(
                data.get("last_accessed_at", extra.get("last_accessed_at", 0.0))
            )
        except (TypeError, ValueError):
            last_accessed_at = 0.0

        try:
            decay_score = float(data.get("decay_score", extra.get("decay_score", 1.0)))
        except (TypeError, ValueError):
            decay_score = 1.0

        archived = bool(data.get("archived", extra.get("archived", False)))

        return cls(
            id=str(data["id"]),
            content=str(data["content"]),
            category=str(data.get("category", "note")),
            tags=[
                str(item).strip()
                for item in data.get("tags", [])
                if str(item).strip()
            ],
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            session_id=str(data.get("session_id", "")),
            scope=scope,
            confidence=max(0.0, min(1.0, confidence)),
            domains=domains,
            source=source,
            usage_count=max(0, usage_count),
            last_accessed_at=max(0.0, last_accessed_at),
            decay_score=decay_score,
            archived=archived,
            extra=extra,
        )


class MemoryStore(Protocol):
    """长期记忆存储接口。"""

    def load_memories(self) -> list[MemoryEntry]:
        ...

    def save_memories(self, entries: list[MemoryEntry]) -> None:
        ...

    def save_memories_and_sync(
        self,
        entries: list[MemoryEntry],
        *,
        changed_entry_ids: list[str] | None = None,
    ) -> None:
        ...

    def add_memory(self, entry: MemoryEntry) -> MemoryEntry:
        ...

    def reconcile_vector_index(self, entries: list[MemoryEntry] | None = None) -> None:
        ...

    def get_memory_by_id(self, memory_id: str) -> MemoryEntry | None:
        ...

    def mark_memories_accessed(self, memory_ids: list[str]) -> None:
        ...

    def search_memories(
        self,
        query: str,
        top_k: int = 5,
        *,
        scope: str | None = None,
        category: str | None = None,
        domains: list[str] | None = None,
        include_archived: bool = False,
        mark_access: bool = True,
    ) -> list[MemoryEntry]:
        ...

    def filter_memories(
        self,
        *,
        scope: str | None = None,
        category: str | None = None,
        domains: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[MemoryEntry]:
        ...

    def get_recent_memories(
        self,
        *,
        limit: int = 10,
        scope: str | None = None,
        category: str | None = None,
        domains: list[str] | None = None,
        include_archived: bool = False,
        sort_by: MemorySortField = "updated_at",
    ) -> list[MemoryEntry]:
        ...


class JsonMemoryStore:
    """
    基于本地 JSON 文件的长期记忆存储实现。

    这里同时负责两件事：
    1. 把权威数据落到 `.memory/memory.json`
    2. 如果启用了 Qdrant，同步维护语义索引
    """

    def __init__(
        self,
        workspace: str,
        *,
        vector_index: MemoryVectorIndex | None = None,
    ) -> None:
        self.workspace = str(Path(workspace).resolve())
        self.memory_dir = Path(self.workspace) / MEMORY_DIR_NAME
        self.memory_file = self.memory_dir / MEMORY_FILE_NAME
        self.vector_index = vector_index

    def _ensure_dir(self) -> None:
        """确保记忆目录存在。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _read_raw_entries(self) -> list[dict[str, Any]]:
        """从 JSON 文件读取原始字典列表。"""
        if not self.memory_file.exists():
            return []

        try:
            raw_text = self.memory_file.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
            if not isinstance(raw_data, dict):
                return []

            entries = raw_data.get("entries", [])
            if not isinstance(entries, list):
                return []

            return [item for item in entries if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []

    def load_memories(self) -> list[MemoryEntry]:
        """加载全部长期记忆，并转成 MemoryEntry 列表。"""
        raw_entries = self._read_raw_entries()
        result: list[MemoryEntry] = []

        for item in raw_entries:
            try:
                result.append(MemoryEntry.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue

        return result

    def get_memory_by_id(self, memory_id: str) -> MemoryEntry | None:
        """按 memory id 查找一条长期记忆。"""
        normalized_memory_id = memory_id.strip()
        if not normalized_memory_id:
            return None

        for entry in self.load_memories():
            if entry.id == normalized_memory_id:
                return entry

        return None

    def mark_memories_accessed(self, memory_ids: list[str]) -> None:
        """
        批量更新指定记忆的访问统计。

        这个接口给 prompt 注入链路使用：
        - 先召回一批候选
        - rerank 后只给真正注入 prompt 的记忆记一次 access
        这样可以避免只是“被召回但没被使用”的记忆污染 usage_count。
        """
        normalized_ids = {item.strip() for item in memory_ids if item.strip()}
        if not normalized_ids:
            return

        by_id = {entry.id: entry for entry in self.load_memories()}
        now = time.time()
        changed = False

        for memory_id in normalized_ids:
            stored_entry = by_id.get(memory_id)
            if stored_entry is None:
                continue

            stored_entry.usage_count += 1
            stored_entry.last_accessed_at = now
            changed = True

        if changed:
            # 访问统计也属于正式 metadata，
            # 因此这里同步更新 JSON 和 Qdrant payload，避免两边长期漂移。
            self.save_memories_and_sync(
                list(by_id.values()),
                changed_entry_ids=list(normalized_ids),
            )

    def save_memories(self, entries: list[MemoryEntry]) -> None:
        """把全部长期记忆完整写回 JSON 文件。"""
        self._ensure_dir()
        payload = {"entries": [entry.to_dict() for entry in entries]}
        temp_file = self.memory_file.with_suffix(".json.tmp")
        temp_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_file.replace(self.memory_file)

    def save_memories_and_sync(
        self,
        entries: list[MemoryEntry],
        *,
        changed_entry_ids: list[str] | None = None,
    ) -> None:
        """
        保存长期记忆，并把变更同步到 Qdrant。

        这个方法主要给 curator 使用，因为 curator 会修改已有记忆的：
        - archived
        - extra.merged_into / extra.superseded_by
        - updated_at
        """
        previous_text = ""
        had_previous_file = self.memory_file.exists()
        previous_entries = self.load_memories() if had_previous_file else []
        if had_previous_file:
            previous_text = self.memory_file.read_text(encoding="utf-8")

        self.save_memories(entries)

        normalized_changed_ids = {
            item.strip()
            for item in (changed_entry_ids or [])
            if item.strip()
        }
        if self.vector_index is None:
            return
        previous_ids = {entry.id for entry in previous_entries if entry.id.strip()}
        current_ids = {entry.id for entry in entries if entry.id.strip()}
        removed_ids = sorted(previous_ids - current_ids)
        by_id = {entry.id: entry for entry in entries}
        changed_entries = sorted(normalized_changed_ids | set(removed_ids))

        try:
            if removed_ids:
                self.vector_index.delete_memories(removed_ids)

            if normalized_changed_ids:
                self.sync_entries_to_vector(
                    [
                        by_id[memory_id]
                        for memory_id in changed_entries
                        if memory_id in by_id
                    ]
                )
        except Exception:
            # 如果 Qdrant 同步失败，就恢复 JSON，避免本地与向量库状态不一致。
            if had_previous_file:
                self.memory_file.write_text(previous_text, encoding="utf-8")
            else:
                try:
                    self.memory_file.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def reconcile_vector_index(self, entries: list[MemoryEntry] | None = None) -> None:
        """
        把 Qdrant 收敛到与本地 JSON 一致。

        这个方法用于两类场景：
        1. 启动时做一次全量对齐，清理历史遗留孤儿点
        2. 需要人工修复时，拿本地权威数据重新覆盖向量索引
        """
        if self.vector_index is None:
            return

        current_entries = entries if entries is not None else self.load_memories()
        current_by_id = {
            entry.id: entry
            for entry in current_entries
            if entry.id.strip()
        }
        current_ids = set(current_by_id.keys())
        vector_ids = self.vector_index.list_memory_ids()
        orphan_ids = sorted(vector_ids - current_ids)

        if orphan_ids:
            self.vector_index.delete_memories(orphan_ids)

        if current_by_id:
            self.sync_entries_to_vector(list(current_by_id.values()))

    def sync_entries_to_vector(self, entries: list[MemoryEntry]) -> None:
        """
        把指定记忆集合批量同步到 Qdrant。

        这里统一使用 upsert：
        - 新记忆会被写入
        - 已归档记忆会用新的 payload 覆盖旧 payload
        """
        if self.vector_index is None:
            return

        for entry in entries:
            self.vector_index.upsert_memory(entry)

    def add_memory(self, entry: MemoryEntry) -> MemoryEntry:
        """
        新增一条长期记忆并同步到向量索引。

        顺序仍然保持：
        1. 先写 JSON
        2. 再写 Qdrant
        3. 如果 Qdrant 失败，回滚 JSON
        """
        entries = self.load_memories()

        if not entry.id.strip():
            entry.id = uuid.uuid4().hex[:12]

        now = time.time()
        if not entry.created_at:
            entry.created_at = now
        entry.updated_at = now

        entries.append(entry)
        self.save_memories(entries)

        if self.vector_index is not None:
            try:
                self.sync_entries_to_vector([entry])
            except Exception:
                rolled_back_entries = [item for item in entries if item.id != entry.id]
                self.save_memories(rolled_back_entries)
                raise

        return entry

    def filter_memories(
        self,
        *,
        scope: str | None = None,
        category: str | None = None,
        domains: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[MemoryEntry]:
        """按 metadata 过滤长期记忆。"""
        normalized_scope = _normalize_text(scope or "")
        normalized_category = _normalize_text(category or "")
        normalized_domains = {
            _normalize_text(domain)
            for domain in (domains or [])
            if _normalize_text(domain)
        }

        result: list[MemoryEntry] = []
        for entry in self.load_memories():
            if not include_archived and entry.archived:
                continue

            if normalized_scope and _normalize_text(entry.scope) != normalized_scope:
                continue

            if normalized_category and _normalize_text(entry.category) != normalized_category:
                continue

            if normalized_domains:
                entry_domains = {_normalize_text(domain) for domain in entry.domains}
                if not (entry_domains & normalized_domains):
                    continue

            result.append(entry)

        return result

    def get_recent_memories(
        self,
        *,
        limit: int = 10,
        scope: str | None = None,
        category: str | None = None,
        domains: list[str] | None = None,
        include_archived: bool = False,
        sort_by: MemorySortField = "updated_at",
    ) -> list[MemoryEntry]:
        """获取最近写入或最近访问的记忆集合。"""
        entries = self.filter_memories(
            scope=scope,
            category=category,
            domains=domains,
            include_archived=include_archived,
        )
        entries.sort(
            key=lambda entry: self._get_sort_value(entry, sort_by),
            reverse=True,
        )
        return entries[:limit]

    def search_memories(
        self,
        query: str,
        top_k: int = 5,
        *,
        scope: str | None = None,
        category: str | None = None,
        domains: list[str] | None = None,
        include_archived: bool = False,
        mark_access: bool = True,
    ) -> list[MemoryEntry]:
        """
        按查询语句检索最相关的长期记忆。

        优先级：
        1. 若配置了向量索引，优先走 Qdrant 语义召回
        2. 如果语义召回失败，再回退到本地词面检索
        """
        semantic_entries = self._search_memories_semantically(
            query=query,
            top_k=top_k,
            scope=scope,
            category=category,
            domains=domains,
            include_archived=include_archived,
            mark_access=mark_access,
        )
        if semantic_entries:
            return semantic_entries

        return self._search_memories_lexically(
            query=query,
            top_k=top_k,
            scope=scope,
            category=category,
            domains=domains,
            include_archived=include_archived,
            mark_access=mark_access,
        )

    def _search_memories_semantically(
        self,
        *,
        query: str,
        top_k: int,
        scope: str | None,
        category: str | None,
        domains: list[str] | None,
        include_archived: bool,
        mark_access: bool,
    ) -> list[MemoryEntry]:
        """使用 Qdrant 做语义检索。"""
        if self.vector_index is None:
            return []

        try:
            hits = self.vector_index.search_similar_memories(
                query_text=query,
                top_k=top_k,
                scope=scope,
                category=category,
                domains=domains,
                include_archived=include_archived,
            )
        except Exception:
            return []

        by_id = {entry.id: entry for entry in self.load_memories()}
        result: list[MemoryEntry] = []
        for hit in hits:
            entry = by_id.get(hit.memory_id)
            if entry is not None:
                result.append(entry)

        if mark_access and result:
            self._mark_entries_accessed(result)

        return result

    def _search_memories_lexically(
        self,
        *,
        query: str,
        top_k: int,
        scope: str | None,
        category: str | None,
        domains: list[str] | None,
        include_archived: bool,
        mark_access: bool,
    ) -> list[MemoryEntry]:
        """使用本地词面规则做检索。"""
        normalized_query = _normalize_text(query)
        if not normalized_query:
            return []

        query_tokens = set(_tokenize(normalized_query))
        if not query_tokens:
            return []

        candidate_entries = self.filter_memories(
            scope=scope,
            category=category,
            domains=domains,
            include_archived=include_archived,
        )

        scored_entries: list[tuple[float, MemoryEntry]] = []
        for entry in candidate_entries:
            content_tokens = set(_tokenize(entry.content))
            if not content_tokens:
                continue

            overlap = query_tokens & content_tokens
            score = float(len(overlap))

            if entry.category and _normalize_text(entry.category) in normalized_query:
                score += 0.5

            for tag in entry.tags:
                normalized_tag = _normalize_text(tag)
                if normalized_tag and normalized_tag in normalized_query:
                    score += 0.5

            for domain in entry.domains:
                normalized_domain = _normalize_text(domain)
                if normalized_domain and normalized_domain in normalized_query:
                    score += 0.35

            # 轻量把 confidence 和 decay 也纳入排序。
            score += min(0.2, max(0.0, entry.confidence) * 0.2)
            score += min(0.2, max(0.0, entry.decay_score) * 0.1)

            if score > 0:
                scored_entries.append((score, entry))

        scored_entries.sort(
            key=lambda item: (item[0], item[1].updated_at),
            reverse=True,
        )
        picked_entries = [entry for _, entry in scored_entries[:top_k]]

        if mark_access and picked_entries:
            self._mark_entries_accessed(picked_entries)

        return picked_entries

    def _mark_entries_accessed(self, picked_entries: list[MemoryEntry]) -> None:
        """
        批量更新记忆访问统计。

        访问统计主要给后面的 decay / rerank 使用。
        """
        memory_ids = [entry.id for entry in picked_entries if entry.id]
        if not memory_ids:
            return
        self.mark_memories_accessed(memory_ids)

    def _get_sort_value(self, entry: MemoryEntry, sort_by: MemorySortField) -> float:
        """读取排序字段对应的值。"""
        if sort_by == "created_at":
            return float(entry.created_at)
        if sort_by == "last_accessed_at":
            return float(entry.last_accessed_at)
        if sort_by == "usage_count":
            return float(entry.usage_count)
        return float(entry.updated_at)


def create_memory_entry(
    content: str,
    category: str = "note",
    tags: list[str] | None = None,
    session_id: str = "",
    *,
    scope: str = "project",
    confidence: float = 0.0,
    domains: list[str] | None = None,
    source: str = "",
    usage_count: int = 0,
    last_accessed_at: float = 0.0,
    decay_score: float = 1.0,
    archived: bool = False,
    extra: dict[str, Any] | None = None,
) -> MemoryEntry:
    """创建一条新的长期记忆。"""
    now = time.time()

    return MemoryEntry(
        id=uuid.uuid4().hex[:12],
        content=content.strip(),
        category=category.strip() or "note",
        tags=[tag.strip() for tag in (tags or []) if tag.strip()],
        created_at=now,
        updated_at=now,
        session_id=session_id.strip(),
        scope=scope.strip() or "project",
        confidence=max(0.0, min(1.0, float(confidence))),
        domains=[domain.strip() for domain in (domains or []) if domain.strip()],
        source=source.strip(),
        usage_count=max(0, int(usage_count)),
        last_accessed_at=max(0.0, float(last_accessed_at)),
        decay_score=float(decay_score),
        archived=bool(archived),
        extra=dict(extra or {}),
    )
