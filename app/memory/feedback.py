from __future__ import annotations

import time

from app.memory.store import MemoryEntry, MemoryStore


class MemoryFeedbackStore:
    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        success_increment: int = 2,
        failure_decrement: int = 1,
    ) -> None:
        self.memory_store = memory_store
        self.success_increment = max(0, int(success_increment))
        self.failure_decrement = max(0, int(failure_decrement))

    def record(self, memory_ids: list[str], *, success: bool) -> list[MemoryEntry]:
        normalized_ids: list[str] = []
        seen: set[str] = set()
        for memory_id in memory_ids:
            normalized_id = str(memory_id).strip()
            if not normalized_id or normalized_id in seen:
                continue
            seen.add(normalized_id)
            normalized_ids.append(normalized_id)

        if not normalized_ids:
            return []

        by_id = {entry.id: entry for entry in self.memory_store.load_memories()}
        now = time.time()
        changed_ids: list[str] = []
        changed_entries: list[MemoryEntry] = []

        for memory_id in normalized_ids:
            entry = by_id.get(memory_id)
            if entry is None:
                continue

            entry.last_accessed_at = now
            if success:
                entry.usage_count += self.success_increment
            else:
                entry.usage_count = max(0, entry.usage_count - self.failure_decrement)

            changed_ids.append(memory_id)
            changed_entries.append(entry)

        if changed_ids:
            self.memory_store.save_memories_and_sync(
                list(by_id.values()),
                changed_entry_ids=changed_ids,
            )

        return changed_entries
