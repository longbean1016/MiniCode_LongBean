from __future__ import annotations

from app.memory.read_pipeline import MemoryReadPipeline
from app.memory.store import MemoryStore
from app.state.session import SessionData
from app.state.working_memory import WorkingMemory


def build_memory_context(
    *,
    user_input: str,
    session: SessionData,
    working_memory: WorkingMemory | None,
    memory_store: MemoryStore | None,
    session_summary_override: str = "",
    top_k: int = 4,
    retrieval_top_k: int = 8,
    max_summary_chars: int = 400,
    max_memory_chars_per_item: int = 180,
) -> str:
    return MemoryReadPipeline(memory_store).build_context(
        user_input=user_input,
        session=session,
        working_memory=working_memory,
        session_summary_override=session_summary_override,
        top_k=top_k,
        retrieval_top_k=retrieval_top_k,
        max_summary_chars=max_summary_chars,
        max_memory_chars_per_item=max_memory_chars_per_item,
    ).prompt_context
