from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.memory_decay import DecayRunResult
from app.memory_store import MemoryEntry
from app.types import AgentStep, ChatMessage


class MemoryExtractorLike(Protocol):
    def extract_from_task(
        self,
        *,
        task_description: str,
        final_step: AgentStep,
        turn_messages: list[ChatMessage],
        session_id: str,
        key_decisions: list[str] | None = None,
        failures: list[str] | None = None,
        files_touched: list[str] | None = None,
    ) -> list[MemoryEntry]:
        ...


class MemoryVerifierDecisionLike(Protocol):
    action: str
    matched_memory_id: str


class MemoryVerifierLike(Protocol):
    def find_similar_entries(
        self,
        candidate: MemoryEntry,
        existing_entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        ...

    def verify(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
    ) -> MemoryVerifierDecisionLike:
        ...


class MemoryCuratorLike(Protocol):
    def curate_new_entries(self, new_entries: list[MemoryEntry]) -> None:
        ...

    def should_run_full_scan(self) -> bool:
        ...

    def curate_project_memories(self) -> None:
        ...


class MemoryDecayLike(Protocol):
    def refresh_new_entries(self, new_entries: list[MemoryEntry]) -> DecayRunResult:
        ...

    def should_run_full_refresh(self) -> bool:
        ...

    def refresh_project_memories(self) -> DecayRunResult:
        ...


@dataclass(slots=True)
class MemoryContextResult:
    prompt_context: str = ""
    query_text: str = ""
    injected_entries: list[MemoryEntry] = field(default_factory=list)
    pinned_entries: list[MemoryEntry] = field(default_factory=list)
    retrieved_entries: list[MemoryEntry] = field(default_factory=list)
    debug_lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryWriteResult:
    stored_entries: list[MemoryEntry] = field(default_factory=list)
    extracted_entries: list[MemoryEntry] = field(default_factory=list)
    reflection_attempted: bool = False
    reflection_reason: str = ""


@dataclass(slots=True)
class ExplicitMemoryHandleResult:
    handled: bool
    history: list[ChatMessage] = field(default_factory=list)
    assistant_text: str = ""
    stored_entries: list[MemoryEntry] = field(default_factory=list)
