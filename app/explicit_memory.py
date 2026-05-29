from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.memory_store import MemoryEntry, create_memory_entry


EXPLICIT_MEMORY_SOURCES: set[str] = {
    "manual_memory_input",
}


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _should_pin_to_prompt(*, scope: str, category: str) -> bool:
    normalized_scope = _normalize_text(scope).lower()
    normalized_category = _normalize_text(category).lower()
    if normalized_scope == "user":
        return True
    return normalized_scope == "project" and normalized_category == "convention"


@dataclass(slots=True)
class ExplicitMemoryIntent:
    should_store: bool
    continue_to_agent: bool
    scope: str = "project"
    category: str = "convention"
    source: str = "manual_memory_input"
    content: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    ack_message: str = ""

    def to_memory_entry(self, *, session_id: str) -> MemoryEntry:
        return create_memory_entry(
            content=self.content,
            category=self.category,
            tags=self.tags,
            session_id=session_id,
            scope=self.scope,
            confidence=self.confidence,
            domains=["memory", "project_rules"],
            source=self.source,
            extra={
                "managed_channel": "explicit_memory",
                "pin_to_prompt": _should_pin_to_prompt(
                    scope=self.scope,
                    category=self.category,
                ),
                "recognition_reason": self.reason,
            },
        )


def is_pinned_memory_entry(entry: MemoryEntry) -> bool:
    return (
        entry.source.strip().lower() in EXPLICIT_MEMORY_SOURCES
        and bool(entry.extra.get("pin_to_prompt", False))
        and entry.scope.strip().lower() in {"project", "user"}
        and not entry.archived
    )


def is_explicit_convention_entry(entry: MemoryEntry) -> bool:
    return (
        is_pinned_memory_entry(entry)
        and entry.scope.strip().lower() == "project"
        and entry.category.strip().lower() == "convention"
    )


def parse_manual_memory_input(user_input: str) -> ExplicitMemoryIntent | None:
    raw = user_input.strip()
    if not raw:
        return None

    if raw.startswith("#"):
        content = raw[1:].strip()
        if not content:
            return None
        return ExplicitMemoryIntent(
            should_store=True,
            continue_to_agent=False,
            scope="project",
            category="note",
            source="manual_memory_input",
            content=content,
            tags=["manual", "chat"],
            confidence=1.0,
            reason="user_requested_manual_memory",
            ack_message="已记录到项目记忆。",
        )

    if not raw.lower().startswith("/memory add"):
        return None

    content = raw[len("/memory add") :].strip()
    if not content:
        return ExplicitMemoryIntent(
            should_store=False,
            continue_to_agent=False,
            ack_message="用法：/memory add [user|project:] <内容>",
        )

    scope = "project"
    scope_match = re.match(r"^(user|project)\s*:\s*(.+)$", content, flags=re.I)
    if scope_match:
        scope = scope_match.group(1).strip().lower()
        content = scope_match.group(2).strip()

    if not content:
        return ExplicitMemoryIntent(
            should_store=False,
            continue_to_agent=False,
            ack_message="用法：/memory add [user|project:] <内容>",
        )

    category = "convention" if scope == "project" else "preference"

    return ExplicitMemoryIntent(
        should_store=True,
        continue_to_agent=False,
        scope=scope,
        category=category,
        source="manual_memory_input",
        content=content,
        tags=["manual", "chat"],
        confidence=1.0,
        reason="user_requested_manual_memory",
        ack_message=f"已记录到 {scope} 记忆。",
    )
