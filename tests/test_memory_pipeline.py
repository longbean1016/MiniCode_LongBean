from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from app.memory.explicit import parse_manual_memory_input
from app.memory.guard import MemoryWriteGuard
from app.memory.reflection import ReflectionMemoryCandidate, TaskMemoryReflectionEngine
from app.memory.decay import DecayRunResult
from app.memory.feedback import MemoryFeedbackStore
from app.memory.pipeline import MemoryPipeline
from app.memory.read_pipeline import MemoryReadPipeline
from app.memory.store import JsonMemoryStore, MemoryEntry, create_memory_entry
from app.memory.write_pipeline import MemoryWritePipeline
from app.state.session import create_new_session
from app.types import AgentStep, ToolResult
from app.state.working_memory import WorkingMemory


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
    def test_memory_guard_allows_constraint_task_reflection_with_admission_evidence(self) -> None:
        candidate = create_memory_entry(
            content="上下文管理相关约束尽量沉到独立模块，不要把主要逻辑堆进 main.py。",
            category="constraint",
            tags=["constraint", "context_management"],
            scope="project",
            confidence=0.9,
            source="task_reflection",
            extra={"reflection_admission_source": "project_file_evidence"},
        )

        decision = MemoryWriteGuard().should_store(candidate)

        self.assertTrue(decision.should_store)

    def test_task_reflection_engine_post_filter_keeps_constraint_candidate(self) -> None:
        engine = TaskMemoryReflectionEngine.__new__(TaskMemoryReflectionEngine)
        candidate = ReflectionMemoryCandidate(
            content="上下文压缩相关约束应尽量放到独立模块，不要把逻辑继续堆进 main.py。",
            category="constraint",
            tags=["constraint", "context_management"],
            confidence=0.9,
            domains=["memory", "context"],
        )

        result = TaskMemoryReflectionEngine._post_filter_candidates(engine, [candidate])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category, "constraint")

    def test_write_pipeline_extracts_user_preferences_and_project_constraints_from_user_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MemoryWritePipeline(
                memory_store=JsonMemoryStore(tmpdir),
                memory_extractor=_FakeExtractor([]),
                memory_verifier=_FakeVerifier(),
                memory_curator=_FakeCurator(),
                memory_decay=_FakeDecay(),
            )
            working_memory = WorkingMemory()

            pipeline.remember_user_intent(
                working_memory,
                "默认使用中文回答，修改代码时加上中文注释，并且不要把太多上下文管理逻辑放进 main.py 和 agent_loop.py。",
            )

            preference_entries = working_memory.get_entries_by_type("user_preference")
            constraint_entries = working_memory.get_entries_by_type("project_constraint")

            self.assertTrue(preference_entries)
            self.assertTrue(
                any("中文" in entry.content or "注释" in entry.content for entry in preference_entries)
            )
            self.assertTrue(constraint_entries)
            self.assertTrue(
                any("main.py" in entry.content or "agent_loop.py" in entry.content for entry in constraint_entries)
            )

    def test_write_pipeline_extracts_recent_risks_from_assistant_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MemoryWritePipeline(
                memory_store=JsonMemoryStore(tmpdir),
                memory_extractor=_FakeExtractor([]),
                memory_verifier=_FakeVerifier(),
                memory_curator=_FakeCurator(),
                memory_decay=_FakeDecay(),
            )
            working_memory = WorkingMemory()

            pipeline.record_assistant_reply(
                working_memory,
                content="当前风险是大 tool_result 会在 recent window 里堆积，如果上下文过长就需要优先压缩并避免信息冲刷。",
            )
            pipeline.record_tool_failure(
                working_memory,
                tool_name="run_command",
                result=ToolResult(
                    ok=False,
                    output="context length exceeds limit",
                    error="prompt too long: context length exceeds limit",
                ),
            )

            risk_entries = working_memory.get_entries_by_type("recent_risk")
            error_entries = working_memory.get_entries_by_type("error_context")

            self.assertTrue(risk_entries)
            self.assertTrue(
                any("tool_result" in entry.content or "context length exceeds limit" in entry.content for entry in risk_entries)
            )
            self.assertTrue(error_entries)

    def test_write_pipeline_uses_lightweight_model_for_assistant_reply_memory(self) -> None:
        class _FakeSummarizer:
            def __init__(self) -> None:
                self.received_content = ""

            def extract_assistant_reply_memory(self, *, content: str) -> dict[str, list[str]]:
                self.received_content = content
                return {
                    "key_decisions": ["关键决策：assistant 回复抽取交给轻量模型"],
                    "recent_risks": ["最近风险：模型失败时必须回退规则抽取"],
                    "preferences": ["用户偏好：默认中文回答"],
                    "constraints": ["项目约束：新增逻辑补中文注释"],
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            summarizer = _FakeSummarizer()
            pipeline = MemoryWritePipeline(
                memory_store=JsonMemoryStore(tmpdir),
                memory_extractor=_FakeExtractor([]),
                memory_verifier=_FakeVerifier(),
                memory_curator=_FakeCurator(),
                memory_decay=_FakeDecay(),
                history_summarizer=summarizer,  # type: ignore[arg-type]
            )
            working_memory = WorkingMemory()

            pipeline.record_assistant_reply(
                working_memory,
                content="这段回复用自然语言确认了实现方案，但没有包含旧规则关键词。",
            )

            self.assertIn("自然语言确认", summarizer.received_content)
            self.assertTrue(working_memory.get_entries_by_type("key_decision"))
            self.assertTrue(working_memory.get_entries_by_type("recent_risk"))
            self.assertTrue(working_memory.get_entries_by_type("user_preference"))
            self.assertTrue(working_memory.get_entries_by_type("project_constraint"))

    def test_parse_manual_user_memory_is_not_treated_as_explicit_long_term_memory(self) -> None:
        intent = parse_manual_memory_input("/memory add user: Prefer concise answers")

        self.assertIsNone(intent)

    def test_parse_manual_memory_usage_only_mentions_project_scope(self) -> None:
        intent = parse_manual_memory_input("/memory add")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertFalse(intent.should_store)
        self.assertEqual(intent.ack_message, "用法：/memory add [project:] <内容>")

    def test_parse_manual_memory_keeps_default_and_explicit_project_forms(self) -> None:
        default_intent = parse_manual_memory_input("/memory add 新增接口时优先补测试")
        scoped_intent = parse_manual_memory_input(
            "/memory add project: 修改 session 相关逻辑时优先走 repository/service 层"
        )

        self.assertIsNotNone(default_intent)
        self.assertIsNotNone(scoped_intent)
        assert default_intent is not None
        assert scoped_intent is not None
        self.assertTrue(default_intent.should_store)
        self.assertTrue(scoped_intent.should_store)
        self.assertEqual(default_intent.scope, "project")
        self.assertEqual(scoped_intent.scope, "project")

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

    def test_read_pipeline_uses_active_context_summary_as_primary_baseline_and_orders_user_first(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
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

            session = create_new_session(tmpdir)
            session.extra["active_context_summary"] = (
                "压缩基线：当前任务是总结 session persistence 变更，并保留文件引用。"
            )

            result = MemoryReadPipeline(memory_store).build_context(
                user_input="Summarize the session persistence update with file references.",
                session=session,
                working_memory=WorkingMemory(),
                top_k=2,
                retrieval_top_k=4,
            )

            self.assertIn("## 当前会话压缩基线", result.prompt_context)
            self.assertIn("压缩基线：当前任务是总结 session persistence 变更", result.prompt_context)
            self.assertIn("## 固定用户长期记忆", result.prompt_context)
            self.assertIn("## 项目长期记忆", result.prompt_context)
            self.assertLess(
                result.prompt_context.index("## 当前会话压缩基线"),
                result.prompt_context.index("## 固定用户长期记忆"),
            )
            self.assertLess(
                result.prompt_context.index("## 固定用户长期记忆"),
                result.prompt_context.index("## 项目长期记忆"),
            )
            self.assertLess(
                result.prompt_context.index(user_entry.content),
                result.prompt_context.index(project_entry.content),
            )

    def test_read_pipeline_skips_memories_already_covered_by_summary_or_working_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            duplicate_summary_entry = memory_store.add_memory(
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
            duplicate_working_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Need to summarize session persistence changes.",
                    category="task",
                    tags=["session"],
                    scope="project",
                    confidence=0.9,
                    source="task_reflection",
                )
            )
            remaining_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Keep repository-level session persistence changes inside repository services.",
                    category="convention",
                    tags=["session", "architecture"],
                    scope="project",
                    confidence=0.92,
                    source="task_reflection",
                )
            )

            session = create_new_session(tmpdir)
            session.extra["active_context_summary"] = (
                "压缩基线：Prefer concise summaries with file references."
            )
            working_memory = WorkingMemory()
            working_memory.protect(
                "Need to summarize session persistence changes.",
                entry_type="active_task",
            )

            result = MemoryReadPipeline(memory_store).build_context(
                user_input="Summarize the session persistence update with file references.",
                session=session,
                working_memory=working_memory,
                top_k=3,
                retrieval_top_k=6,
            )

            self.assertIn(remaining_entry.content, result.prompt_context)
            self.assertNotIn(
                "1. (user/preference) Prefer concise summaries with file references.",
                result.prompt_context,
            )
            self.assertNotIn(
                "1. (project/task) Need to summarize session persistence changes.",
                result.prompt_context,
            )
            self.assertEqual(
                {entry.id for entry in result.injected_entries},
                {remaining_entry.id},
            )

            by_id = {entry.id: entry for entry in memory_store.load_memories()}
            self.assertEqual(by_id[duplicate_summary_entry.id].usage_count, 0)
            self.assertEqual(by_id[duplicate_working_entry.id].usage_count, 0)
            self.assertEqual(by_id[remaining_entry.id].usage_count, 1)

    def test_read_pipeline_splits_pinned_and_retrieved_user_memories_before_project_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            pinned_user_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Pinned user preference: always answer in concise Chinese.",
                    category="preference",
                    tags=["style"],
                    scope="user",
                    confidence=1.0,
                    source="manual_memory_input",
                    extra={"pin_to_prompt": True, "managed_channel": "explicit_memory"},
                )
            )
            retrieved_user_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Retrieved user preference: include file references when summarizing code changes.",
                    category="preference",
                    tags=["summary", "style"],
                    scope="user",
                    confidence=0.95,
                    source="task_reflection",
                )
            )
            project_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Project convention: keep context-management wiring out of main.py.",
                    category="convention",
                    tags=["architecture", "context_management"],
                    scope="project",
                    confidence=0.94,
                    source="task_reflection",
                )
            )

            session = create_new_session(tmpdir)
            result = MemoryReadPipeline(memory_store).build_context(
                user_input="Summarize the context management changes with file references in Chinese.",
                session=session,
                working_memory=WorkingMemory(),
                top_k=3,
                retrieval_top_k=6,
            )

            self.assertIn("## 固定用户长期记忆", result.prompt_context)
            self.assertIn("## 相关用户长期记忆", result.prompt_context)
            self.assertIn("## 项目长期记忆", result.prompt_context)
            self.assertLess(
                result.prompt_context.index("## 固定用户长期记忆"),
                result.prompt_context.index("## 相关用户长期记忆"),
            )
            self.assertLess(
                result.prompt_context.index("## 相关用户长期记忆"),
                result.prompt_context.index("## 项目长期记忆"),
            )
            self.assertIn(pinned_user_entry.content, result.prompt_context)
            self.assertIn(retrieved_user_entry.content, result.prompt_context)
            self.assertIn(project_entry.content, result.prompt_context)

    def test_read_pipeline_keeps_user_memory_when_top_k_is_tight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            pinned_user_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Pinned user preference: default to Chinese responses.",
                    category="preference",
                    tags=["style"],
                    scope="user",
                    confidence=1.0,
                    source="manual_memory_input",
                    extra={"pin_to_prompt": True, "managed_channel": "explicit_memory"},
                )
            )
            retrieved_user_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Retrieved user preference: preserve compressed context details when possible.",
                    category="preference",
                    tags=["compression", "style"],
                    scope="user",
                    confidence=0.96,
                    source="task_reflection",
                )
            )
            project_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Project convention: keep long-term memory prominent in compressed context.",
                    category="convention",
                    tags=["compression", "memory"],
                    scope="project",
                    confidence=0.95,
                    source="task_reflection",
                )
            )

            session = create_new_session(tmpdir)
            result = MemoryReadPipeline(memory_store).build_context(
                user_input="Adjust compression so long-term memory remains visible.",
                session=session,
                working_memory=WorkingMemory(),
                top_k=2,
                retrieval_top_k=6,
            )

            self.assertIn(pinned_user_entry.content, result.prompt_context)
            self.assertIn(retrieved_user_entry.content, result.prompt_context)
            self.assertNotIn(project_entry.content, result.prompt_context)

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

    def test_write_pipeline_does_not_handle_memory_add_user_alias(self) -> None:
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

            self.assertFalse(result.handled)
            self.assertEqual(len(memory_store.load_memories()), 0)
            self.assertFalse(result.assistant_text)
            self.assertFalse((Path(memory_store.workspace) / "USER.md").exists())

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
