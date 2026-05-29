from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass

from app.explicit_memory import parse_manual_memory_input
from app.memory_decay import DecayRunResult
from app.memory_feedback import MemoryFeedbackStore
from app.memory_pipeline import MemoryPipeline
from app.memory_read_pipeline import MemoryReadPipeline
from app.memory_store import JsonMemoryStore, MemoryEntry, create_memory_entry
from app.memory_write_pipeline import MemoryWritePipeline
from app.session import create_new_session
from app.types import AgentStep
from app.working_memory import WorkingMemory


@dataclass
class _VerifierDecision:
    action: str
    matched_memory_id: str = ""


class _FakeExtractor:
    def __init__(self, entries: list[MemoryEntry]) -> None:
        self.entries = entries
        self.called = False

    def extract_from_task(self, **_: object) -> list[MemoryEntry]:
        self.called = True
        return list(self.entries)


class _FakeVerifier:
    def find_similar_entries(
        self,
        candidate: MemoryEntry,
        existing_entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        return []

    def verify(
        self,
        candidate: MemoryEntry,
        similar_entries: list[MemoryEntry],
    ) -> _VerifierDecision:
        return _VerifierDecision(action="store")


class _FakeCurator:
    def curate_new_entries(self, new_entries: list[MemoryEntry]) -> None:
        return None

    def should_run_full_scan(self) -> bool:
        return False

    def curate_project_memories(self) -> None:
        return None


class _FakeDecay:
    def refresh_new_entries(self, new_entries: list[MemoryEntry]) -> DecayRunResult:
        return DecayRunResult()

    def should_run_full_refresh(self) -> bool:
        return False

    def refresh_project_memories(self) -> DecayRunResult:
        return DecayRunResult()


class MemoryPipelineTests(unittest.TestCase):
    def test_parse_manual_user_memory_is_pinned_for_prompt(self) -> None:
        intent = parse_manual_memory_input("/memory add user: Prefer concise answers")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertTrue(intent.should_store)

        entry = intent.to_memory_entry(session_id="sess-1")
        self.assertEqual(entry.scope, "user")
        self.assertTrue(entry.extra["pin_to_prompt"])

    def test_read_pipeline_includes_project_and_user_memories_and_marks_only_injected_usage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)

            project_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Use repository services for session persistence changes.",
                    category="convention",
                    tags=["session", "architecture"],
                    scope="project",
                    confidence=0.92,
                    source="task_reflection",
                )
            )
            user_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Prefer concise summaries with file references.",
                    category="preference",
                    tags=["style"],
                    scope="user",
                    confidence=1.0,
                    source="manual_memory_input",
                    extra={"pin_to_prompt": True, "managed_channel": "explicit_memory"},
                )
            )
            unrelated_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Use snake_case database migrations for analytics tables.",
                    category="convention",
                    tags=["database"],
                    scope="project",
                    confidence=0.89,
                    source="task_reflection",
                )
            )

            session = create_new_session(tmpdir)
            working_memory = WorkingMemory()
            working_memory.protect(
                "Need to summarize session persistence changes.",
                entry_type="active_task",
            )

            result = MemoryReadPipeline(memory_store).build_context(
                user_input="Summarize the session persistence update with file references.",
                session=session,
                working_memory=working_memory,
                top_k=2,
                retrieval_top_k=4,
            )

            self.assertIn(
                "Prefer concise summaries with file references.",
                result.prompt_context,
            )
            self.assertIn(
                "Use repository services for session persistence changes.",
                result.prompt_context,
            )
            self.assertNotIn(unrelated_entry.content, result.prompt_context)
            self.assertEqual(
                {entry.id for entry in result.injected_entries},
                {project_entry.id, user_entry.id},
            )

            by_id = {entry.id: entry for entry in memory_store.load_memories()}
            self.assertEqual(by_id[project_entry.id].usage_count, 1)
            self.assertEqual(by_id[user_entry.id].usage_count, 1)
            self.assertEqual(by_id[unrelated_entry.id].usage_count, 0)

    def test_write_pipeline_reflects_without_stable_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            candidate_entry = create_memory_entry(
                content="Session persistence updates should stay inside repository services.",
                category="convention",
                tags=["session", "architecture"],
                scope="project",
                confidence=0.91,
                source="task_reflection",
            )
            extractor = _FakeExtractor([candidate_entry])

            working_memory = WorkingMemory()
            working_memory.protect("app/session.py", entry_type="reflection_file")

            pipeline = MemoryWritePipeline(
                memory_store=memory_store,
                memory_extractor=extractor,
                memory_verifier=_FakeVerifier(),
                memory_curator=_FakeCurator(),
                memory_decay=_FakeDecay(),
            )

            result = pipeline.reflect_and_persist(
                task_description="Refactor session persistence memory flow.",
                final_step=AgentStep(
                    type="assistant",
                    content="我把 session 持久化链路拆到了仓储层里。",
                    kind="final",
                ),
                turn_messages=[
                    {
                        "role": "assistant",
                        "content": "我把 session 持久化链路拆到了仓储层里。",
                    }
                ],
                session_id="sess-2",
                working_memory=working_memory,
                decay_log_enabled=False,
                decay_log_echo=False,
            )

            self.assertTrue(extractor.called)
            self.assertEqual(len(result.stored_entries), 1)
            self.assertEqual(memory_store.load_memories()[0].content, candidate_entry.content)

    def test_write_pipeline_handles_explicit_user_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            pipeline = MemoryWritePipeline(
                memory_store=memory_store,
                memory_extractor=_FakeExtractor([]),
                memory_verifier=_FakeVerifier(),
                memory_curator=_FakeCurator(),
                memory_decay=_FakeDecay(),
            )

            result = pipeline.handle_explicit_input(
                user_input="/memory add user: Prefer concise answers",
                session_id="sess-3",
                history=[],
                decay_log_enabled=False,
                decay_log_echo=False,
            )

            self.assertTrue(result.handled)
            self.assertIn("已记录", result.assistant_text)
            stored_entries = memory_store.load_memories()
            self.assertEqual(len(stored_entries), 1)
            self.assertEqual(stored_entries[0].scope, "user")
            self.assertTrue(stored_entries[0].extra["pin_to_prompt"])

    def test_feedback_store_rewards_success_and_penalizes_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            project_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Use repository services for persistence work.",
                    scope="project",
                    usage_count=1,
                    confidence=0.9,
                )
            )
            user_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Prefer concise answers.",
                    scope="user",
                    usage_count=1,
                    confidence=1.0,
                )
            )

            feedback_store = MemoryFeedbackStore(memory_store)
            feedback_store.record([project_entry.id, user_entry.id], success=True)

            after_success = {entry.id: entry for entry in memory_store.load_memories()}
            self.assertEqual(after_success[project_entry.id].usage_count, 3)
            self.assertEqual(after_success[user_entry.id].usage_count, 3)

            feedback_store.record([project_entry.id, user_entry.id], success=False)

            after_failure = {entry.id: entry for entry in memory_store.load_memories()}
            self.assertEqual(after_failure[project_entry.id].usage_count, 2)
            self.assertEqual(after_failure[user_entry.id].usage_count, 2)
            self.assertGreater(after_failure[project_entry.id].last_accessed_at, 0)
            self.assertGreater(after_failure[user_entry.id].last_accessed_at, 0)

    def test_memory_pipeline_smoke_covers_injection_feedback_and_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            project_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Use repository services for session persistence changes.",
                    category="convention",
                    tags=["session", "architecture"],
                    scope="project",
                    confidence=0.92,
                    source="task_reflection",
                )
            )
            user_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Prefer concise summaries with file references.",
                    category="preference",
                    tags=["style"],
                    scope="user",
                    confidence=1.0,
                    source="manual_memory_input",
                    extra={"pin_to_prompt": True, "managed_channel": "explicit_memory"},
                )
            )
            reflected_entry = create_memory_entry(
                content="Session persistence updates should stay inside repository services.",
                category="convention",
                tags=["session", "architecture"],
                scope="project",
                confidence=0.91,
                source="task_reflection",
            )

            pipeline = MemoryPipeline(
                read_pipeline=MemoryReadPipeline(memory_store),
                write_pipeline=MemoryWritePipeline(
                    memory_store=memory_store,
                    memory_extractor=_FakeExtractor([reflected_entry]),
                    memory_verifier=_FakeVerifier(),
                    memory_curator=_FakeCurator(),
                    memory_decay=_FakeDecay(),
                ),
                feedback_store=MemoryFeedbackStore(memory_store),
            )

            session = create_new_session(tmpdir)
            working_memory = WorkingMemory()
            pipeline.remember_user_intent(
                working_memory,
                "Summarize the session persistence update with file references.",
            )
            working_memory.protect(
                "Need to summarize session persistence changes.",
                entry_type="active_task",
            )
            working_memory.protect("app/session.py", entry_type="reflection_file")

            context_result = pipeline.build_prompt_context(
                user_input="Summarize the session persistence update with file references.",
                session=session,
                working_memory=working_memory,
                top_k=2,
                retrieval_top_k=4,
            )

            self.assertIn(project_entry.content, context_result.prompt_context)
            self.assertIn(user_entry.content, context_result.prompt_context)

            write_result = pipeline.finalize_turn(
                task_description="Refactor session persistence memory flow.",
                final_step=AgentStep(
                    type="assistant",
                    content="I moved session persistence into repository services.",
                    kind="final",
                ),
                turn_messages=[
                    {
                        "role": "assistant",
                        "content": "I moved session persistence into repository services.",
                    }
                ],
                session_id="sess-4",
                working_memory=working_memory,
                decay_log_enabled=False,
                decay_log_echo=False,
            )

            self.assertTrue(write_result.reflection_attempted)
            self.assertEqual(len(write_result.stored_entries), 1)

            by_id = {entry.id: entry for entry in memory_store.load_memories()}
            self.assertEqual(by_id[project_entry.id].usage_count, 3)
            self.assertEqual(by_id[user_entry.id].usage_count, 3)
            self.assertEqual(
                by_id[write_result.stored_entries[0].id].content,
                reflected_entry.content,
            )


if __name__ == "__main__":
    unittest.main()
