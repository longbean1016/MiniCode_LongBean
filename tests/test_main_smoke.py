from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import app.main as main_module
from app.memory_decay import DecayRunResult
from app.memory_store import JsonMemoryStore, MemoryEntry, create_memory_entry
from app.memory_write_pipeline import MemoryWritePipeline
from app.types import AgentStep, AppConfig, ToolResult


class _FakeToolRegistry:
    def list_tool_name(self) -> list[str]:
        return []

    def execute_tool(self, tool_name: str, input_data: object, context: object) -> object:
        raise AssertionError(f"unexpected tool call: {tool_name}")


class _FakeModelAdapter:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, object]]] = []

    def next(self, messages: list[dict[str, object]], on_stream_chunk=None, store=None) -> AgentStep:
        self.calls.append(list(messages))
        return AgentStep(
            type="assistant",
            content=self.reply,
            kind="final",
        )


class _FakeSequentialModelAdapter:
    def __init__(self, steps: list[AgentStep]) -> None:
        self.steps = list(steps)
        self.calls: list[list[dict[str, object]]] = []
        self.index = 0

    def next(self, messages: list[dict[str, object]], on_stream_chunk=None, store=None) -> AgentStep:
        self.calls.append(list(messages))
        if self.index >= len(self.steps):
            raise AssertionError("unexpected extra model call")

        step = self.steps[self.index]
        self.index += 1
        return step


class _FakeSuccessfulToolRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def list_tool_name(self) -> list[str]:
        return ["read_file"]

    def execute_tool(self, tool_name: str, input_data: object, context: object) -> ToolResult:
        self.calls.append((tool_name, input_data))
        return ToolResult(
            ok=True,
            output="read app/session.py successfully",
        )


class _FakeFailingToolRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def list_tool_name(self) -> list[str]:
        return ["lookup_session_rules"]

    def execute_tool(self, tool_name: str, input_data: object, context: object) -> ToolResult:
        self.calls.append((tool_name, input_data))
        return ToolResult(
            ok=False,
            output="session lookup timed out",
            error="upstream timeout while looking up session rules",
        )


class _FakePermissionToolRegistry:
    _ACTION_KEY = "read_file:app/session.py"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, bool]] = []

    def list_tool_name(self) -> list[str]:
        return ["read_file"]

    def execute_tool(self, tool_name: str, input_data: object, context: object) -> ToolResult:
        approved_actions = set(getattr(context, "approved_actions", set()))
        is_approved = self._ACTION_KEY in approved_actions
        self.calls.append((tool_name, input_data, is_approved))

        if not is_approved:
            return ToolResult(
                ok=False,
                output="approval required for read_file",
                error="PERMISSION_REQUIRED",
                meta={
                    "command": "read_file app/session.py",
                    "reason": "reading app/session.py requires approval",
                    "action_key": self._ACTION_KEY,
                },
            )

        return ToolResult(
            ok=True,
            output="read app/session.py successfully",
        )


class _FakeExtractor:
    def __init__(self, entries: list[MemoryEntry]) -> None:
        self.entries = entries
        self.called = False

    def extract_from_task(self, **_: object) -> list[MemoryEntry]:
        self.called = True
        return list(self.entries)


class _FakeVerifierDecision:
    def __init__(self, action: str = "store", matched_memory_id: str = "") -> None:
        self.action = action
        self.matched_memory_id = matched_memory_id


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
    ) -> _FakeVerifierDecision:
        return _FakeVerifierDecision()


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


class _FakeSummarizer:
    def summarize(self, **_: object) -> str:
        return ""


def _build_test_config(workspace_root: str) -> AppConfig:
    return AppConfig(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="fake-model",
        embedding_model="fake-embedding",
        embedding_api_key="test-key",
        embedding_base_url="https://example.invalid/v1",
        embedding_dimensions=0,
        workspace_root=workspace_root,
        qdrant_enabled=False,
        qdrant_url="http://localhost:6333",
        qdrant_api_key="",
        qdrant_collection="project_memories",
        model_retry_max_attempts=1,
        model_retry_base_delay_seconds=0.0,
        model_retry_backoff_multiplier=1.0,
        model_retry_max_delay_seconds=0.0,
        model_circuit_failure_threshold=1,
        model_circuit_recovery_timeout_seconds=0.0,
        aux_model_retry_max_attempts=1,
        aux_model_retry_base_delay_seconds=0.0,
        aux_model_retry_backoff_multiplier=1.0,
        aux_model_retry_max_delay_seconds=0.0,
        aux_model_circuit_failure_threshold=1,
        aux_model_circuit_recovery_timeout_seconds=0.0,
        vector_retry_max_attempts=1,
        vector_retry_base_delay_seconds=0.0,
        vector_retry_backoff_multiplier=1.0,
        vector_retry_max_delay_seconds=0.0,
        vector_circuit_failure_threshold=1,
        vector_circuit_recovery_timeout_seconds=0.0,
        decay_full_scan_trigger_count=40,
        decay_min_score=0.05,
        decay_archive_threshold=0.12,
        decay_archive_age_days=45.0,
        decay_archive_confidence_threshold=0.72,
        decay_archive_usage_threshold=1,
        decay_log_enabled=False,
        decay_log_echo=False,
    )


class MainSmokeTests(unittest.TestCase):
    def test_main_entry_smoke_supports_user_add_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()

            with patch.object(sys, "argv", ["main.py"]), patch(
                "builtins.input",
                side_effect=[
                    "/user add 回答尽量直接，修改代码时加中文注释",
                    "/user",
                    "exit",
                ],
            ), patch.object(
                main_module,
                "load_config",
                return_value=_build_test_config(tmpdir),
            ), patch.object(
                main_module,
                "build_tool_registry",
                return_value=_FakeToolRegistry(),
            ), patch.object(
                main_module,
                "OpenAIModelAdapter",
                return_value=_FakeModelAdapter("unexpected"),
            ), patch.object(
                main_module,
                "LongTermMemoryExtractor",
                return_value=_FakeExtractor([]),
            ), patch.object(
                main_module,
                "MemoryVerifier",
                return_value=_FakeVerifier(),
            ), patch.object(
                main_module,
                "MemoryCurator",
                return_value=_FakeCurator(),
            ), patch.object(
                main_module,
                "MemoryDecay",
                return_value=_FakeDecay(),
            ), patch.object(
                main_module,
                "OlderHistorySummarizer",
                return_value=_FakeSummarizer(),
            ), redirect_stdout(stdout):
                main_module.main()

            output = stdout.getvalue()
            self.assertIn("已追加到 USER.md", output)
            self.assertIn("回答尽量直接", output)
            self.assertIn("修改代码时加中文注释", output)

            with open(f"{tmpdir}\\USER.md", "r", encoding="utf-8") as handle:
                user_md = handle.read()

            self.assertIn("## Custom Instructions", user_md)
            self.assertIn("- 回答尽量直接，修改代码时加中文注释", user_md)

    def test_main_entry_smoke_supports_user_profile_manual_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()

            with patch.object(sys, "argv", ["main.py"]), patch(
                "builtins.input",
                side_effect=[
                    "/user set preferences.language zh-CN",
                    "/user set coding_style.comments 修改代码时加中文注释",
                    "/user paths",
                    "/user",
                    "exit",
                ],
            ), patch.object(
                main_module,
                "load_config",
                return_value=_build_test_config(tmpdir),
            ), patch.object(
                main_module,
                "build_tool_registry",
                return_value=_FakeToolRegistry(),
            ), patch.object(
                main_module,
                "OpenAIModelAdapter",
                return_value=_FakeModelAdapter("unexpected"),
            ), patch.object(
                main_module,
                "LongTermMemoryExtractor",
                return_value=_FakeExtractor([]),
            ), patch.object(
                main_module,
                "MemoryVerifier",
                return_value=_FakeVerifier(),
            ), patch.object(
                main_module,
                "MemoryCurator",
                return_value=_FakeCurator(),
            ), patch.object(
                main_module,
                "MemoryDecay",
                return_value=_FakeDecay(),
            ), patch.object(
                main_module,
                "OlderHistorySummarizer",
                return_value=_FakeSummarizer(),
            ), redirect_stdout(stdout):
                main_module.main()

            output = stdout.getvalue()
            self.assertIn("已写入 USER.md", output)
            self.assertIn("USER.md 路径", output)
            self.assertIn("默认使用中文回答", output)
            self.assertIn("修改代码时加中文注释", output)

            with open(f"{tmpdir}\\USER.md", "r", encoding="utf-8") as handle:
                user_md = handle.read()

            self.assertIn("- **Language**: zh-CN", user_md)
            self.assertIn("- **Comments**: 修改代码时加中文注释", user_md)

    def test_main_entry_smoke_injects_memories_and_persists_reflection(self) -> None:
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

            final_reply = (
                "Session persistence should stay inside repository services, "
                "and future replies should keep file references concise."
            )
            reflected_content = (
                "Session persistence updates should stay inside repository services."
            )
            extractor = _FakeExtractor(
                [
                    create_memory_entry(
                        content=reflected_content,
                        category="convention",
                        tags=["session", "architecture"],
                        scope="project",
                        confidence=0.91,
                        source="task_reflection",
                    )
                ]
            )
            fake_model = _FakeModelAdapter(final_reply)
            stdout = io.StringIO()

            with patch.object(sys, "argv", ["main.py"]), patch(
                "builtins.input",
                side_effect=[
                    "Refactor session persistence to repository services.",
                    "exit",
                ],
            ), patch.object(
                main_module,
                "load_config",
                return_value=_build_test_config(tmpdir),
            ), patch.object(
                main_module,
                "build_tool_registry",
                return_value=_FakeToolRegistry(),
            ), patch.object(
                main_module,
                "OpenAIModelAdapter",
                return_value=fake_model,
            ), patch.object(
                main_module,
                "LongTermMemoryExtractor",
                return_value=extractor,
            ), patch.object(
                main_module,
                "MemoryVerifier",
                return_value=_FakeVerifier(),
            ), patch.object(
                main_module,
                "MemoryCurator",
                return_value=_FakeCurator(),
            ), patch.object(
                main_module,
                "MemoryDecay",
                return_value=_FakeDecay(),
            ), patch.object(
                main_module,
                "OlderHistorySummarizer",
                return_value=_FakeSummarizer(),
            ), redirect_stdout(stdout):
                main_module.main()

            self.assertTrue(extractor.called)
            self.assertEqual(len(fake_model.calls), 1)

            system_prompt = str(fake_model.calls[0][0]["content"])
            self.assertIn(project_entry.content, system_prompt)
            self.assertIn(user_entry.content, system_prompt)

            by_id = {entry.id: entry for entry in JsonMemoryStore(tmpdir).load_memories()}
            self.assertEqual(by_id[project_entry.id].usage_count, 3)
            self.assertEqual(by_id[user_entry.id].usage_count, 3)
            self.assertIn(reflected_content, [entry.content for entry in by_id.values()])

            output = stdout.getvalue()
            self.assertIn("LongBean MiniCode Agent", output)
            self.assertIn(final_reply, output)

    def test_main_entry_smoke_reflects_with_project_file_evidence(self) -> None:
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

            reflected_content = (
                "Session persistence updates should stay inside repository services."
            )
            extractor = _FakeExtractor(
                [
                    create_memory_entry(
                        content=reflected_content,
                        category="convention",
                        tags=["session", "architecture"],
                        scope="project",
                        confidence=0.91,
                        source="task_reflection",
                    )
                ]
            )
            fake_model = _FakeSequentialModelAdapter(
                [
                    AgentStep(
                        type="tool_calls",
                        calls=[
                            {
                                "id": "call-1",
                                "tool_name": "read_file",
                                "input": {"path": "app/session.py"},
                            }
                        ],
                    ),
                    AgentStep(
                        type="assistant",
                        content=(
                            "I checked app/session.py and kept session persistence inside "
                            "repository services."
                        ),
                        kind="final",
                    ),
                ]
            )
            tool_registry = _FakeSuccessfulToolRegistry()
            stdout = io.StringIO()

            with patch.object(sys, "argv", ["main.py"]), patch(
                "builtins.input",
                side_effect=[
                    "Refactor session persistence to repository services.",
                    "exit",
                ],
            ), patch.object(
                main_module,
                "load_config",
                return_value=_build_test_config(tmpdir),
            ), patch.object(
                main_module,
                "build_tool_registry",
                return_value=tool_registry,
            ), patch.object(
                main_module,
                "OpenAIModelAdapter",
                return_value=fake_model,
            ), patch.object(
                main_module,
                "LongTermMemoryExtractor",
                return_value=extractor,
            ), patch.object(
                main_module,
                "MemoryVerifier",
                return_value=_FakeVerifier(),
            ), patch.object(
                main_module,
                "MemoryCurator",
                return_value=_FakeCurator(),
            ), patch.object(
                main_module,
                "MemoryDecay",
                return_value=_FakeDecay(),
            ), patch.object(
                main_module,
                "OlderHistorySummarizer",
                return_value=_FakeSummarizer(),
            ), redirect_stdout(stdout):
                main_module.main()

            self.assertTrue(extractor.called)
            self.assertEqual(len(tool_registry.calls), 1)
            self.assertEqual(tool_registry.calls[0][0], "read_file")
            self.assertEqual(len(fake_model.calls), 2)

            first_prompt = str(fake_model.calls[0][0]["content"])
            second_prompt = str(fake_model.calls[1][0]["content"])
            self.assertIn(project_entry.content, first_prompt)
            self.assertIn(user_entry.content, first_prompt)
            self.assertIn(project_entry.content, second_prompt)
            self.assertIn(user_entry.content, second_prompt)

            stored_entries = JsonMemoryStore(tmpdir).load_memories()
            by_id = {entry.id: entry for entry in stored_entries}
            self.assertEqual(by_id[project_entry.id].usage_count, 4)
            self.assertEqual(by_id[user_entry.id].usage_count, 4)

            reflected_entries = [
                entry for entry in stored_entries if entry.content == reflected_content
            ]
            self.assertEqual(len(reflected_entries), 1)
            reflected_entry = reflected_entries[0]
            self.assertEqual(
                reflected_entry.extra["reflection_admission_source"],
                "project_file_evidence",
            )
            self.assertTrue(reflected_entry.extra["reflection_has_project_file_evidence"])
            self.assertEqual(
                reflected_entry.extra["project_files_touched"],
                ["app/session.py"],
            )

            output = stdout.getvalue()
            self.assertIn("LongBean MiniCode Agent", output)
            self.assertIn(
                "I checked app/session.py and kept session persistence inside repository services.",
                output,
            )

    def test_main_entry_smoke_penalizes_injected_memories_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            project_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Use repository services for session persistence changes.",
                    category="convention",
                    tags=["session", "architecture"],
                    scope="project",
                    confidence=0.92,
                    usage_count=2,
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
                    usage_count=2,
                    source="manual_memory_input",
                    extra={"pin_to_prompt": True, "managed_channel": "explicit_memory"},
                )
            )

            reflected_content = (
                "Session persistence updates should stay inside repository services."
            )
            extractor = _FakeExtractor(
                [
                    create_memory_entry(
                        content=reflected_content,
                        category="convention",
                        tags=["session", "architecture"],
                        scope="project",
                        confidence=0.91,
                        source="task_reflection",
                    )
                ]
            )
            final_reply = (
                f"{MemoryWritePipeline._BLOCKED_REFLECTION_PREFIXES[0]} upstream timeout"
            )
            fake_model = _FakeModelAdapter(final_reply)
            stdout = io.StringIO()

            with patch.object(sys, "argv", ["main.py"]), patch(
                "builtins.input",
                side_effect=[
                    "Refactor session persistence to repository services.",
                    "exit",
                ],
            ), patch.object(
                main_module,
                "load_config",
                return_value=_build_test_config(tmpdir),
            ), patch.object(
                main_module,
                "build_tool_registry",
                return_value=_FakeToolRegistry(),
            ), patch.object(
                main_module,
                "OpenAIModelAdapter",
                return_value=fake_model,
            ), patch.object(
                main_module,
                "LongTermMemoryExtractor",
                return_value=extractor,
            ), patch.object(
                main_module,
                "MemoryVerifier",
                return_value=_FakeVerifier(),
            ), patch.object(
                main_module,
                "MemoryCurator",
                return_value=_FakeCurator(),
            ), patch.object(
                main_module,
                "MemoryDecay",
                return_value=_FakeDecay(),
            ), patch.object(
                main_module,
                "OlderHistorySummarizer",
                return_value=_FakeSummarizer(),
            ), redirect_stdout(stdout):
                main_module.main()

            self.assertFalse(extractor.called)
            self.assertEqual(len(fake_model.calls), 1)

            system_prompt = str(fake_model.calls[0][0]["content"])
            self.assertIn(project_entry.content, system_prompt)
            self.assertIn(user_entry.content, system_prompt)

            stored_entries = JsonMemoryStore(tmpdir).load_memories()
            by_id = {entry.id: entry for entry in stored_entries}
            self.assertEqual(by_id[project_entry.id].usage_count, 2)
            self.assertEqual(by_id[user_entry.id].usage_count, 2)
            self.assertNotIn(reflected_content, [entry.content for entry in stored_entries])

            output = stdout.getvalue()
            self.assertIn("LongBean MiniCode Agent", output)
            self.assertIn(final_reply, output)

    def test_main_entry_smoke_reflects_failure_recovery_via_execution_signal(self) -> None:
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

            reflected_content = (
                "When session rule lookups fail, keep the fallback inside repository services."
            )
            extractor = _FakeExtractor(
                [
                    create_memory_entry(
                        content=reflected_content,
                        category="convention",
                        tags=["session", "failure"],
                        scope="project",
                        confidence=0.91,
                        source="task_reflection",
                    )
                ]
            )
            fake_model = _FakeSequentialModelAdapter(
                [
                    AgentStep(
                        type="tool_calls",
                        calls=[
                            {
                                "id": "call-1",
                                "tool_name": "lookup_session_rules",
                                "input": {"query": "session persistence fallback"},
                            }
                        ],
                    ),
                    AgentStep(
                        type="assistant",
                        content=(
                            "Session persistence should stay inside repository services, "
                            "and when rule lookups fail we should return a concise fallback."
                        ),
                        kind="final",
                    ),
                ]
            )
            tool_registry = _FakeFailingToolRegistry()
            stdout = io.StringIO()

            with patch.object(sys, "argv", ["main.py"]), patch(
                "builtins.input",
                side_effect=[
                    "Refactor session persistence to repository services.",
                    "exit",
                ],
            ), patch.object(
                main_module,
                "load_config",
                return_value=_build_test_config(tmpdir),
            ), patch.object(
                main_module,
                "build_tool_registry",
                return_value=tool_registry,
            ), patch.object(
                main_module,
                "OpenAIModelAdapter",
                return_value=fake_model,
            ), patch.object(
                main_module,
                "LongTermMemoryExtractor",
                return_value=extractor,
            ), patch.object(
                main_module,
                "MemoryVerifier",
                return_value=_FakeVerifier(),
            ), patch.object(
                main_module,
                "MemoryCurator",
                return_value=_FakeCurator(),
            ), patch.object(
                main_module,
                "MemoryDecay",
                return_value=_FakeDecay(),
            ), patch.object(
                main_module,
                "OlderHistorySummarizer",
                return_value=_FakeSummarizer(),
            ), redirect_stdout(stdout):
                main_module.main()

            self.assertTrue(extractor.called)
            self.assertEqual(len(tool_registry.calls), 1)
            self.assertEqual(tool_registry.calls[0][0], "lookup_session_rules")
            self.assertEqual(len(fake_model.calls), 2)

            first_prompt = str(fake_model.calls[0][0]["content"])
            second_prompt = str(fake_model.calls[1][0]["content"])
            self.assertIn(project_entry.content, first_prompt)
            self.assertIn(user_entry.content, first_prompt)
            self.assertIn(project_entry.content, second_prompt)
            self.assertIn(user_entry.content, second_prompt)

            second_messages = fake_model.calls[1]
            tool_results = [
                message
                for message in second_messages
                if str(message.get("role")) == "tool_result"
            ]
            self.assertEqual(len(tool_results), 1)
            self.assertTrue(bool(tool_results[0].get("is_error")))
            self.assertIn("session lookup timed out", str(tool_results[0].get("content")))

            stored_entries = JsonMemoryStore(tmpdir).load_memories()
            by_id = {entry.id: entry for entry in stored_entries}
            self.assertEqual(by_id[project_entry.id].usage_count, 4)
            self.assertEqual(by_id[user_entry.id].usage_count, 4)

            reflected_entries = [
                entry for entry in stored_entries if entry.content == reflected_content
            ]
            self.assertEqual(len(reflected_entries), 1)
            reflected_entry = reflected_entries[0]
            self.assertEqual(
                reflected_entry.extra["reflection_admission_source"],
                "execution_signal",
            )
            self.assertFalse(reflected_entry.extra["reflection_has_project_file_evidence"])
            self.assertEqual(reflected_entry.extra["project_files_touched"], [])

            output = stdout.getvalue()
            self.assertIn("LongBean MiniCode Agent", output)
            self.assertIn(
                "Session persistence should stay inside repository services, and when rule lookups fail we should return a concise fallback.",
                output,
            )

    def test_main_entry_smoke_approval_allows_tool_resume_and_reflection(self) -> None:
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

            reflected_content = (
                "Session persistence updates should stay inside repository services."
            )
            extractor = _FakeExtractor(
                [
                    create_memory_entry(
                        content=reflected_content,
                        category="convention",
                        tags=["session", "architecture"],
                        scope="project",
                        confidence=0.91,
                        source="task_reflection",
                    )
                ]
            )
            fake_model = _FakeSequentialModelAdapter(
                [
                    AgentStep(
                        type="tool_calls",
                        calls=[
                            {
                                "id": "call-1",
                                "tool_name": "read_file",
                                "input": {"path": "app/session.py"},
                            }
                        ],
                    ),
                    AgentStep(
                        type="assistant",
                        content=(
                            "Session persistence should stay inside repository services, "
                            "and future replies should keep file references concise."
                        ),
                        kind="final",
                    ),
                ]
            )
            tool_registry = _FakePermissionToolRegistry()
            stdout = io.StringIO()

            with patch.object(sys, "argv", ["main.py"]), patch(
                "builtins.input",
                side_effect=[
                    "Refactor session persistence to repository services.",
                    "y",
                    "exit",
                ],
            ), patch.object(
                main_module,
                "load_config",
                return_value=_build_test_config(tmpdir),
            ), patch.object(
                main_module,
                "build_tool_registry",
                return_value=tool_registry,
            ), patch.object(
                main_module,
                "OpenAIModelAdapter",
                return_value=fake_model,
            ), patch.object(
                main_module,
                "LongTermMemoryExtractor",
                return_value=extractor,
            ), patch.object(
                main_module,
                "MemoryVerifier",
                return_value=_FakeVerifier(),
            ), patch.object(
                main_module,
                "MemoryCurator",
                return_value=_FakeCurator(),
            ), patch.object(
                main_module,
                "MemoryDecay",
                return_value=_FakeDecay(),
            ), patch.object(
                main_module,
                "OlderHistorySummarizer",
                return_value=_FakeSummarizer(),
            ), redirect_stdout(stdout):
                main_module.main()

            self.assertTrue(extractor.called)
            self.assertEqual(len(tool_registry.calls), 2)
            self.assertEqual(tool_registry.calls[0][0], "read_file")
            self.assertFalse(tool_registry.calls[0][2])
            self.assertTrue(tool_registry.calls[1][2])
            self.assertEqual(len(fake_model.calls), 2)

            second_messages = fake_model.calls[1]
            tool_results = [
                message
                for message in second_messages
                if str(message.get("role")) == "tool_result"
            ]
            self.assertEqual(len(tool_results), 1)
            self.assertFalse(bool(tool_results[0].get("is_error")))
            self.assertIn(
                "read app/session.py successfully",
                str(tool_results[0].get("content")),
            )
            self.assertNotIn(
                "该操作需要用户授权，当前尚未执行。",
                str(tool_results[0].get("content")),
            )

            stored_entries = JsonMemoryStore(tmpdir).load_memories()
            by_id = {entry.id: entry for entry in stored_entries}
            self.assertEqual(by_id[project_entry.id].usage_count, 4)
            self.assertEqual(by_id[user_entry.id].usage_count, 4)

            reflected_entries = [
                entry for entry in stored_entries if entry.content == reflected_content
            ]
            self.assertEqual(len(reflected_entries), 1)
            reflected_entry = reflected_entries[0]
            self.assertEqual(
                reflected_entry.extra["reflection_admission_source"],
                "project_file_evidence",
            )
            self.assertTrue(reflected_entry.extra["reflection_has_project_file_evidence"])
            self.assertEqual(
                reflected_entry.extra["project_files_touched"],
                ["app/session.py"],
            )

            output = stdout.getvalue()
            self.assertIn("该操作需要用户授权", output)
            self.assertIn(
                "Session persistence should stay inside repository services, and future replies should keep file references concise.",
                output,
            )

    def test_main_entry_smoke_approval_denial_skips_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            project_entry = memory_store.add_memory(
                create_memory_entry(
                    content="Use repository services for session persistence changes.",
                    category="convention",
                    tags=["session", "architecture"],
                    scope="project",
                    confidence=0.92,
                    usage_count=2,
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
                    usage_count=2,
                    source="manual_memory_input",
                    extra={"pin_to_prompt": True, "managed_channel": "explicit_memory"},
                )
            )

            reflected_content = (
                "Session persistence updates should stay inside repository services."
            )
            extractor = _FakeExtractor(
                [
                    create_memory_entry(
                        content=reflected_content,
                        category="convention",
                        tags=["session", "architecture"],
                        scope="project",
                        confidence=0.91,
                        source="task_reflection",
                    )
                ]
            )
            fake_model = _FakeSequentialModelAdapter(
                [
                    AgentStep(
                        type="tool_calls",
                        calls=[
                            {
                                "id": "call-1",
                                "tool_name": "read_file",
                                "input": {"path": "app/session.py"},
                            }
                        ],
                    )
                ]
            )
            tool_registry = _FakePermissionToolRegistry()
            stdout = io.StringIO()

            with patch.object(sys, "argv", ["main.py"]), patch(
                "builtins.input",
                side_effect=[
                    "Refactor session persistence to repository services.",
                    "n",
                    "exit",
                ],
            ), patch.object(
                main_module,
                "load_config",
                return_value=_build_test_config(tmpdir),
            ), patch.object(
                main_module,
                "build_tool_registry",
                return_value=tool_registry,
            ), patch.object(
                main_module,
                "OpenAIModelAdapter",
                return_value=fake_model,
            ), patch.object(
                main_module,
                "LongTermMemoryExtractor",
                return_value=extractor,
            ), patch.object(
                main_module,
                "MemoryVerifier",
                return_value=_FakeVerifier(),
            ), patch.object(
                main_module,
                "MemoryCurator",
                return_value=_FakeCurator(),
            ), patch.object(
                main_module,
                "MemoryDecay",
                return_value=_FakeDecay(),
            ), patch.object(
                main_module,
                "OlderHistorySummarizer",
                return_value=_FakeSummarizer(),
            ), redirect_stdout(stdout):
                main_module.main()

            self.assertFalse(extractor.called)
            self.assertEqual(len(tool_registry.calls), 1)
            self.assertFalse(tool_registry.calls[0][2])
            self.assertEqual(len(fake_model.calls), 1)

            stored_entries = JsonMemoryStore(tmpdir).load_memories()
            by_id = {entry.id: entry for entry in stored_entries}
            self.assertEqual(by_id[project_entry.id].usage_count, 3)
            self.assertEqual(by_id[user_entry.id].usage_count, 3)
            self.assertNotIn(reflected_content, [entry.content for entry in stored_entries])

            output = stdout.getvalue()
            self.assertIn("该操作需要用户授权", output)
            self.assertIn("Agent> 用户已拒绝此次高风险操作。", output)


if __name__ == "__main__":
    unittest.main()
