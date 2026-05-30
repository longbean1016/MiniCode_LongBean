from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from app.agent_loop import run_agent_once
from app.memory_models import MemoryContextResult
from app.session import create_new_session
from app.types import AgentStep, ToolContext, ToolResult
from app.working_memory import WorkingMemory


class ContextManagerTests(unittest.TestCase):
    def test_default_usable_context_budget_matches_minicode_style_default(self) -> None:
        from app.context_manager import DEFAULT_USABLE_CONTEXT_BUDGET

        self.assertEqual(DEFAULT_USABLE_CONTEXT_BUDGET, 128_000)

    def test_estimate_tokens_uses_minicode_style_chinese_and_english_rules(self) -> None:
        from app.context_manager import estimate_tokens

        self.assertEqual(estimate_tokens("你好"), 1)
        self.assertEqual(estimate_tokens("hello"), 1)
        self.assertEqual(estimate_tokens("你好hello"), 2)

    def test_collect_context_stats_reports_total_and_usage_ratio(self) -> None:
        from app.context_manager import collect_context_stats

        stats = collect_context_stats(
            system_prompt="系统提示",
            recent_messages=[
                {"role": "user", "content": "hello world"},
                {"role": "tool_result", "content": "x" * 200},
            ],
            memory_context="长期记忆",
            usable_budget=100,
        )

        self.assertGreater(stats.total_tokens, 0)
        self.assertGreater(stats.usage_ratio, 0)
        self.assertGreater(stats.tool_result_tokens, 0)
        self.assertEqual(stats.message_count, 3)

    def test_decide_context_policy_shrinks_budget_when_usage_ratio_is_high(self) -> None:
        from app.context_manager import ContextStats, decide_context_policy

        stats = ContextStats(
            usable_budget=100,
            total_tokens=82,
            usage_ratio=0.82,
            system_tokens=6,
            recent_tokens=60,
            memory_tokens=16,
            tool_result_tokens=28,
            message_count=8,
            tool_result_count=3,
        )

        policy = decide_context_policy(stats)

        self.assertEqual(policy.level, 3)
        self.assertEqual(policy.keep_rounds, 3)
        self.assertEqual(policy.memory_top_k, 1)
        self.assertLessEqual(policy.memory_item_chars, 80)

    def test_user_profile_loader_parses_user_md_into_resolved_preferences(self) -> None:
        from app.user_profile import load_user_profile

        with tempfile.TemporaryDirectory() as tmpdir:
            user_md_path = f"{tmpdir}\\USER.md"
            with open(user_md_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "# User Profile\n\n"
                    "## Preferences\n"
                    "- **Language**: zh-CN\n"
                    "- **Verbosity**: concise\n"
                    "- **Response Style**: technical\n\n"
                    "## Coding Style\n"
                    "- **Comments**: 中文注释\n\n"
                    "## Custom Instructions\n"
                    "尽量最小改动。\n"
                )

            profile = load_user_profile(tmpdir)

            self.assertIsNotNone(profile)
            assert profile is not None
            self.assertIn("默认使用中文回答", profile.to_preference_lines())
            self.assertIn("回答尽量简洁", profile.to_preference_lines())
            self.assertIn("修改代码时加中文注释", profile.to_preference_lines())
            self.assertIn("尽量最小改动", profile.to_preference_lines())

    def test_user_profile_loader_accepts_loose_manual_text_in_user_md(self) -> None:
        from app.user_profile import load_user_profile

        with tempfile.TemporaryDirectory() as tmpdir:
            user_md_path = f"{tmpdir}\\USER.md"
            with open(user_md_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "# User Profile\n\n"
                    "这里是我手动写的一些偏好。\n"
                    "回答尽量直接。\n"
                    "修改代码时加中文注释。\n"
                )

            profile = load_user_profile(tmpdir)

            self.assertIsNotNone(profile)
            assert profile is not None
            self.assertIn("回答尽量直接", profile.to_preference_lines())
            self.assertIn("修改代码时加中文注释", profile.to_preference_lines())


class ContextCompactorTests(unittest.TestCase):
    def test_compactor_truncates_large_tool_results_and_clears_old_ones(self) -> None:
        from app.context_compactor import compact_recent_messages

        messages = [
            {"role": "user", "content": "开始任务"},
            {"role": "tool_result", "tool_name": "read_file", "content": "A" * 5000},
            {"role": "tool_result", "tool_name": "list_files", "content": "B" * 100},
            {"role": "tool_result", "tool_name": "grep_files", "content": "C" * 100},
            {"role": "tool_result", "tool_name": "read_file", "content": "D" * 100},
        ]

        result = compact_recent_messages(
            messages,
            max_recent_tool_results=2,
            truncate_tool_result_chars=800,
        )

        self.assertEqual(len(result.messages), len(messages))
        self.assertGreaterEqual(result.truncated_tool_results, 1)
        self.assertGreaterEqual(result.cleared_old_tool_results, 1)
        self.assertIn("已省略", result.messages[1]["content"])
        self.assertIn("已省略", result.messages[2]["content"])

    def test_compactor_persists_large_tool_result_and_replaces_with_line_preview(self) -> None:
        from app.context_compactor import compact_recent_messages

        with tempfile.TemporaryDirectory() as tmpdir:
            content_lines = [f"line-{index}" for index in range(1, 21)]
            tool_content = "\n".join(content_lines)
            messages = [
                {"role": "user", "content": "检查日志"},
                {"role": "tool_result", "tool_name": "read_file", "content": tool_content},
            ]

            result = compact_recent_messages(
                messages,
                max_recent_tool_results=2,
                truncate_tool_result_chars=40,
                workspace=tmpdir,
            )

            preview = result.messages[1]["content"]
            self.assertIn("[工具结果已落盘", preview)
            self.assertIn("工具: read_file", preview)
            self.assertIn(".cache", preview)
            self.assertIn("line-1", preview)
            self.assertIn("line-8", preview)
            self.assertIn("line-18", preview)
            self.assertIn("line-20", preview)
            self.assertIn("省略", preview)
            self.assertNotIn(tool_content, preview)

            persisted_path = result.messages[1]["_persisted_path"]
            self.assertTrue(str(persisted_path).startswith(tmpdir))
            self.assertIn("\\.cache\\", str(persisted_path))
            with open(persisted_path, "r", encoding="utf-8") as handle:
                persisted_text = handle.read()

            self.assertIn("tool_name", persisted_text)
            self.assertIn(tool_content, persisted_text)

    def test_compactor_prefers_raw_output_when_persisting_tool_result(self) -> None:
        from app.context_compactor import compact_recent_messages

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_output = "\n".join(f"raw-line-{index}" for index in range(1, 30))
            messages = [
                {"role": "user", "content": "检查原始输出"},
                {
                    "role": "tool_result",
                    "tool_name": "run_command",
                    "content": "summary only",
                    "meta": {"raw_output": raw_output, "truncated": True},
                },
            ]

            result = compact_recent_messages(
                messages,
                max_recent_tool_results=2,
                truncate_tool_result_chars=40,
                workspace=tmpdir,
            )

            persisted_path = result.messages[1]["_persisted_path"]
            with open(persisted_path, "r", encoding="utf-8") as handle:
                persisted_text = handle.read()
            self.assertIn(raw_output, persisted_text)
            self.assertNotIn("summary only", persisted_text.split("---CONTENT---", 1)[1])

    def test_compactor_dedups_same_read_file_when_path_and_content_hash_match(self) -> None:
        from app.context_compactor import compact_recent_messages

        shared_content = "\n".join(
            [
                "FILE: src/demo.py",
                "OFFSET: 0",
                "END: 12",
                "TOTAL_CHARS: 12",
                "TRUNCATED: no",
                "",
                "print('demo')",
            ]
        )
        messages = [
            {
                "role": "tool_result",
                "tool_name": "read_file",
                "content": shared_content,
                "meta": {"path": "src/demo.py"},
            },
            {
                "role": "tool_result",
                "tool_name": "read_file",
                "content": shared_content,
                "meta": {"path": "src/demo.py"},
            },
        ]

        result = compact_recent_messages(
            messages,
            max_recent_tool_results=4,
            truncate_tool_result_chars=10_000,
        )

        self.assertEqual(result.deduped_read_results, 1)
        self.assertIn("读取结果已去重", result.messages[0]["content"])
        self.assertEqual(result.messages[1]["content"], shared_content)

    def test_compactor_keeps_read_file_when_same_path_but_content_changed(self) -> None:
        from app.context_compactor import compact_recent_messages

        messages = [
            {
                "role": "tool_result",
                "tool_name": "read_file",
                "content": "FILE: src/demo.py\n\nvalue = 1\n",
                "meta": {"path": "src/demo.py"},
            },
            {
                "role": "tool_result",
                "tool_name": "read_file",
                "content": "FILE: src/demo.py\n\nvalue = 2\n",
                "meta": {"path": "src/demo.py"},
            },
        ]

        result = compact_recent_messages(
            messages,
            max_recent_tool_results=4,
            truncate_tool_result_chars=10_000,
        )

        self.assertEqual(result.deduped_read_results, 0)
        self.assertEqual(result.messages[0]["content"], messages[0]["content"])
        self.assertEqual(result.messages[1]["content"], messages[1]["content"])

    def test_compactor_dedups_semantically_equivalent_list_files_results(self) -> None:
        from app.context_compactor import compact_recent_messages

        list_output = (
            "ROOT: app\n"
            "TOTAL_ENTRIES: 4\n"
            "RETURNED_ENTRIES: 4\n"
            "TRUNCATED: no\n"
            "\n"
            "file main.py\n"
            "file context_auto_compact.py\n"
            "file context_compactor.py\n"
            "file context_runtime.py\n"
        )
        messages = [
            {"role": "tool_result", "tool_name": "list_files", "content": list_output},
            {"role": "tool_result", "tool_name": "list_files", "content": list_output},
        ]

        result = compact_recent_messages(
            messages,
            max_recent_tool_results=4,
            truncate_tool_result_chars=10_000,
        )

        self.assertEqual(result.deduped_read_results, 1)
        self.assertIn("扫描结果已去重", result.messages[0]["content"])
        self.assertEqual(result.messages[1]["content"], list_output)


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def next(self, messages, on_stream_chunk=None, store=None):  # type: ignore[no-untyped-def]
        self.calls.append(list(messages))
        return AgentStep(type="assistant", content="ok", kind="final")


class _OverflowThenSuccessModel:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def next(self, messages, on_stream_chunk=None, store=None):  # type: ignore[no-untyped-def]
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            raise RuntimeError("prompt too long: context length exceeds limit")
        return AgentStep(type="assistant", content="recovered", kind="final")


class _FakeToolRegistry:
    def list_tool_name(self) -> list[str]:
        return []

    def execute_tool(self, tool_name: str, input_data: object, context: object) -> object:
        raise AssertionError("unexpected tool call")


class _RecordingMemoryPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def build_prompt_context(self, **kwargs: object) -> MemoryContextResult:
        self.calls.append(dict(kwargs))
        return MemoryContextResult(prompt_context="记忆上下文")

    def record_assistant_reply(self, working_memory: WorkingMemory, *, content: str) -> None:
        return None


class _NoopExtractor:
    def extract_from_task(self, **_: object) -> list[object]:
        return []


@dataclass
class _NoopVerifierDecision:
    action: str = "store"
    matched_memory_id: str = ""


class _NoopVerifier:
    def find_similar_entries(self, candidate: object, existing_entries: list[object]) -> list[object]:
        return []

    def verify(self, candidate: object, similar_entries: list[object]) -> _NoopVerifierDecision:
        return _NoopVerifierDecision()


class _NoopCurator:
    def curate_new_entries(self, new_entries: list[object]) -> None:
        return None

    def should_run_full_scan(self) -> bool:
        return False

    def curate_project_memories(self) -> None:
        return None


class _NoopDecay:
    def refresh_new_entries(self, new_entries: list[object]) -> object:
        return object()

    def should_run_full_refresh(self) -> bool:
        return False

    def refresh_project_memories(self) -> object:
        return object()


class AgentLoopContextPolicyTests(unittest.TestCase):
    def test_agent_loop_logs_preview_token_stats_before_final_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _FakeModel()
            tool_registry = _FakeToolRegistry()
            memory_pipeline = _RecordingMemoryPipeline()
            session = create_new_session(tmpdir)
            working_memory = WorkingMemory()
            working_memory.protect("整理仓库上下文", entry_type="user_intent")

            long_tool_result = "日志" * 120_000
            history = [
                {"role": "user", "content": "请分析这个仓库"},
                {"role": "tool_result", "tool_name": "read_file", "content": long_tool_result},
            ]

            with patch("app.agent_loop.log_event") as mock_log_event:
                step, next_history = run_agent_once(
                    user_input="继续分析并总结",
                    model=model,
                    tool_registry=tool_registry,
                    tool_context=ToolContext(cwd=tmpdir),
                    session=session,
                    working_memory=working_memory,
                    memory_pipeline=memory_pipeline,
                    history=history,
                    max_steps=1,
                    session_id="sess-preview-log",
                )

            self.assertEqual(step.type, "assistant")
            self.assertTrue(next_history)
            log_messages = [str(call.args[0]) for call in mock_log_event.call_args_list if call.args]
            preview_logs = [message for message in log_messages if "preview_total=" in message]
            self.assertTrue(preview_logs)
            self.assertIn("preview_usage=", preview_logs[0])
            self.assertIn("preview_tool_results=", preview_logs[0])

    def test_agent_loop_triggers_compression_with_default_128k_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _FakeModel()
            tool_registry = _FakeToolRegistry()
            memory_pipeline = _RecordingMemoryPipeline()
            session = create_new_session(tmpdir)
            working_memory = WorkingMemory()
            working_memory.protect("整理仓库上下文", entry_type="user_intent")

            # 这里不覆盖 usable_context_budget，直接走默认 128k。
            # 用足够大的中文 tool_result 模拟真实压缩链路。
            long_tool_result = "日志" * 120_000
            history = [
                {"role": "user", "content": "请分析这个仓库"},
                {"role": "tool_result", "tool_name": "read_file", "content": long_tool_result},
            ]

            step, next_history = run_agent_once(
                user_input="继续分析并总结",
                model=model,
                tool_registry=tool_registry,
                tool_context=ToolContext(cwd=tmpdir),
                session=session,
                working_memory=working_memory,
                memory_pipeline=memory_pipeline,
                history=history,
                max_steps=1,
                session_id="sess-default-budget",
            )

            self.assertEqual(step.type, "assistant")
            self.assertTrue(next_history)
            self.assertEqual(len(memory_pipeline.calls), 1)
            self.assertEqual(memory_pipeline.calls[0]["top_k"], 1)
            self.assertLessEqual(memory_pipeline.calls[0]["max_memory_chars_per_item"], 80)
            self.assertEqual(session.extra["compaction_level"], 3)

    def test_prepare_agent_context_persists_active_context_state(self) -> None:
        from app.context_runtime import prepare_agent_context
        from app.context_state import load_context_state

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}\\USER.md", "w", encoding="utf-8") as handle:
                handle.write(
                    "# User Profile\n\n"
                    "## Preferences\n"
                    "- **Language**: zh-CN\n\n"
                    "## Coding Style\n"
                    "- **Comments**: 中文注释\n"
                )

            tool_registry = _FakeToolRegistry()
            memory_pipeline = _RecordingMemoryPipeline()
            session = create_new_session(tmpdir)
            working_memory = WorkingMemory()
            working_memory.protect("整理仓库上下文", entry_type="user_intent")

            full_history = [
                {"role": "user", "content": "分析这个仓库"},
                {"role": "tool_result", "tool_name": "read_file", "content": "日志" * 1200},
            ]

            prepared = prepare_agent_context(
                full_history=full_history,
                session=session,
                tool_registry=tool_registry,
                working_memory=working_memory,
                memory_pipeline=memory_pipeline,
                history_summarizer=None,
            )

            state = load_context_state(tmpdir, session.session_id)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.source_message_count, len(full_history))
            self.assertEqual(state.compaction_level, prepared.policy.level)
            self.assertTrue(state.compacted_messages)
            self.assertIn("preview_total", state.last_token_stats)
            self.assertTrue(state.compact_memory_context.strip())
            self.assertIn("默认使用中文回答", state.resolved_user_preferences)

    def test_prepare_agent_context_resolves_project_constraints_from_project_memory(self) -> None:
        from app.context_runtime import prepare_agent_context
        from app.context_state import load_context_state
        from app.memory_read_pipeline import MemoryReadPipeline
        from app.memory_store import JsonMemoryStore, create_memory_entry
        from app.memory_write_pipeline import MemoryWritePipeline
        from app.memory_pipeline import MemoryPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}\\USER.md", "w", encoding="utf-8") as handle:
                handle.write(
                    "# User Profile\n\n"
                    "## Preferences\n"
                    "- **Language**: zh-CN\n"
                )

            memory_store = JsonMemoryStore(tmpdir)
            memory_store.add_memory(
                create_memory_entry(
                    content="不要把太多上下文管理逻辑放进 main.py 和 agent_loop.py",
                    category="constraint",
                    tags=["constraint", "context_management"],
                    scope="project",
                    confidence=0.95,
                    source="task_reflection",
                )
            )
            memory_store.add_memory(
                create_memory_entry(
                    content="token 只按中文和英文估算",
                    category="convention",
                    tags=["constraint", "token_budget"],
                    scope="project",
                    confidence=0.93,
                    source="task_reflection",
                )
            )

            memory_pipeline = MemoryPipeline(
                read_pipeline=MemoryReadPipeline(memory_store),
                write_pipeline=MemoryWritePipeline(
                    memory_store=memory_store,
                    memory_extractor=_NoopExtractor(),
                    memory_verifier=_NoopVerifier(),
                    memory_curator=_NoopCurator(),
                    memory_decay=_NoopDecay(),
                ),
            )
            tool_registry = _FakeToolRegistry()
            session = create_new_session(tmpdir)
            working_memory = WorkingMemory()
            working_memory.protect("整理上下文约束", entry_type="user_intent")

            prepared = prepare_agent_context(
                full_history=[{"role": "user", "content": "继续整理上下文压缩逻辑"}],
                session=session,
                tool_registry=tool_registry,
                working_memory=working_memory,
                memory_pipeline=memory_pipeline,
                history_summarizer=None,
            )

            state = load_context_state(tmpdir, session.session_id)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertTrue(state.resolved_project_constraints)
            self.assertIn("默认使用中文回答", state.resolved_user_preferences)
            self.assertTrue(
                any("main.py" in item or "token" in item for item in state.resolved_project_constraints)
            )
            self.assertIn("项目约束", prepared.compact_memory_context)

    def test_prepare_agent_context_filters_transient_execution_constraints_from_state(self) -> None:
        from app.context_runtime import prepare_agent_context
        from app.context_state import load_context_state
        from app.memory_pipeline import MemoryPipeline
        from app.memory_read_pipeline import MemoryReadPipeline
        from app.memory_store import JsonMemoryStore, create_memory_entry
        from app.memory_write_pipeline import MemoryWritePipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = JsonMemoryStore(tmpdir)
            memory_store.add_memory(
                create_memory_entry(
                    content="修改上下文管理代码时，不要把太多逻辑继续堆进 main.py。",
                    category="constraint",
                    tags=["constraint", "context_management"],
                    scope="project",
                    confidence=0.95,
                    source="task_reflection",
                )
            )
            memory_store.add_memory(
                create_memory_entry(
                    content="执行验收测试时，要求最多允许 8 次工具调用，完成写文件后立即停止。",
                    category="constraint",
                    tags=["constraint", "token_budget"],
                    scope="project",
                    confidence=0.96,
                    source="task_reflection",
                )
            )

            memory_pipeline = MemoryPipeline(
                read_pipeline=MemoryReadPipeline(memory_store),
                write_pipeline=MemoryWritePipeline(
                    memory_store=memory_store,
                    memory_extractor=_NoopExtractor(),
                    memory_verifier=_NoopVerifier(),
                    memory_curator=_NoopCurator(),
                    memory_decay=_NoopDecay(),
                ),
            )
            tool_registry = _FakeToolRegistry()
            session = create_new_session(tmpdir)
            working_memory = WorkingMemory()
            working_memory.protect(
                "最多允许 6 次工具调用，完成后立即停止。",
                entry_type="project_constraint",
            )
            working_memory.protect(
                "修改代码时尽量保持接口兼容。",
                entry_type="project_constraint",
            )

            prepare_agent_context(
                full_history=[{"role": "user", "content": "继续整理上下文压缩逻辑"}],
                session=session,
                tool_registry=tool_registry,
                working_memory=working_memory,
                memory_pipeline=memory_pipeline,
                history_summarizer=None,
            )

            state = load_context_state(tmpdir, session.session_id)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertTrue(
                any("main.py" in item or "接口兼容" in item for item in state.resolved_project_constraints)
            )
            self.assertFalse(
                any("工具调用" in item or "立即停止" in item for item in state.resolved_project_constraints)
            )

    def test_build_compact_memory_context_merges_summary_and_working_memory(self) -> None:
        from app.context_compact_memory import build_compact_memory_context

        working_memory = WorkingMemory()
        working_memory.protect("用户偏好：回答时优先用中文", entry_type="user_preference")
        working_memory.protect("项目约束：尽量不要改动 main.py", entry_type="project_constraint")
        working_memory.protect("最近风险：上下文过长时需要优先压缩工具结果", entry_type="error_context")

        compact_memory_context = build_compact_memory_context(
            older_history_summary="旧对话摘要：最近主要在整理上下文压缩链路。",
            working_memory=working_memory,
        )

        self.assertIn("压缩记忆基线", compact_memory_context)
        self.assertIn("旧对话摘要", compact_memory_context)
        self.assertIn("用户偏好", compact_memory_context)
        self.assertIn("项目约束", compact_memory_context)
        self.assertIn("最近风险", compact_memory_context)

    def test_build_compact_memory_context_prioritizes_preferences_constraints_and_risks(self) -> None:
        from app.context_compact_memory import build_compact_memory_context

        working_memory = WorkingMemory()
        working_memory.protect("默认使用中文回答，并尽量把结论放在前面。", entry_type="user_preference", importance=1.0)
        working_memory.protect("不要把太多上下文管理逻辑放进 main.py。", entry_type="project_constraint", importance=1.0)
        working_memory.protect("最近风险：大 tool_result 可能堆积并挤占 recent window。", entry_type="recent_risk", importance=1.0)

        for index in range(8):
            working_memory.protect(
                f"普通活跃任务 {index}: 继续整理上下文链路的辅助细节。",
                entry_type="active_task",
                importance=0.4,
            )

        compact_memory_context = build_compact_memory_context(
            older_history_summary="旧对话摘要：" + ("这里是一些较长但优先级更低的上下文。 " * 20),
            working_memory=working_memory,
        )

        self.assertIn("用户偏好", compact_memory_context)
        self.assertIn("项目约束", compact_memory_context)
        self.assertIn("最近风险", compact_memory_context)
        self.assertIn("main.py", compact_memory_context)
        self.assertIn("tool_result", compact_memory_context)

    def test_build_compact_memory_context_skips_transient_carry_over_sections(self) -> None:
        from app.context_compact_memory import build_compact_memory_context

        compact_memory_context = build_compact_memory_context(
            older_history_summary="",
            working_memory=WorkingMemory(),
            previous_compact_memory_context=(
                "压缩记忆基线\n"
                "## 用户偏好\n"
                "- 默认使用中文回答\n"
                "## 项目约束\n"
                "- 最多允许 8 次工具调用，完成写文件后立即停止。\n"
                "## 当前任务\n"
                "- tmp/real_chain_outputs/session_context_audit.md\n"
                "## 稳定事实\n"
                "- 复用上一次压缩基线，而不是回退到旧摘要。\n"
            ),
        )

        self.assertIn("上次压缩延续", compact_memory_context)
        self.assertIn("默认使用中文回答", compact_memory_context)
        self.assertIn("复用上一次压缩基线", compact_memory_context)
        self.assertNotIn("最多允许 8 次工具调用", compact_memory_context)
        self.assertNotIn("tmp/real_chain_outputs", compact_memory_context)

    def test_build_compact_memory_snapshot_uses_structured_sections_instead_of_free_text_carry_over(self) -> None:
        from app.context_compact_memory import build_compact_memory_snapshot, render_compact_memory_context

        working_memory = WorkingMemory()
        working_memory.protect("当前活跃任务：重构上下文压缩链路", entry_type="active_task", importance=0.9)
        working_memory.protect("关键决策：压缩阶段优先保留结构化事实", entry_type="key_decision", importance=0.9)
        working_memory.protect("开放问题：需要避免重复目录扫描撑爆 recent window", entry_type="recent_risk", importance=0.9)

        previous_snapshot = {
            "preferences": ["默认使用中文回答"],
            "stable_constraints": ["所有新增代码尽量加中文注释"],
            "tool_findings": ["已验证 read_file 精确去重已经存在"],
            "active_tasks": ["旧任务：不应该从上一版快照直接续带"],
            "open_issues": ["旧问题：不应该覆盖本轮风险"],
        }

        snapshot = build_compact_memory_snapshot(
            older_history_summary="旧对话摘要：当前主要在治理上下文压缩污染。",
            working_memory=working_memory,
            previous_snapshot=previous_snapshot,
            resolved_user_preferences=["回答尽量直接"],
            resolved_project_constraints=["不要把太多上下文管理逻辑塞进 main.py"],
            recent_risks=["最近风险：目录树类噪音可能误入 recent_risk"],
        )
        rendered = render_compact_memory_context(snapshot)

        self.assertEqual(snapshot["preferences"][:2], ["回答尽量直接", "默认使用中文回答"])
        self.assertIn("不要把太多上下文管理逻辑塞进 main.py", snapshot["stable_constraints"])
        self.assertIn("当前活跃任务：重构上下文压缩链路", snapshot["active_tasks"])
        self.assertIn("关键决策：压缩阶段优先保留结构化事实", snapshot["decisions"])
        self.assertIn("最近风险：目录树类噪音可能误入 recent_risk", snapshot["open_issues"])
        self.assertNotIn("旧任务：不应该从上一版快照直接续带", snapshot["active_tasks"])
        self.assertIn("结构化压缩记忆", rendered)
        self.assertIn("关键工具发现", rendered)

    def test_working_memory_enforces_protected_slot_limits_and_builds_snapshot(self) -> None:
        working_memory = WorkingMemory(max_entries=20, max_tokens=400)
        working_memory.protect("偏好一：中文回答", entry_type="user_preference", importance=0.8)
        working_memory.protect("偏好二：回答直接", entry_type="user_preference", importance=0.9)
        working_memory.protect("偏好三：这条应该被挤掉", entry_type="user_preference", importance=0.2)
        working_memory.protect("约束一：不要把上下文治理逻辑都塞进 main.py", entry_type="project_constraint", importance=0.9)
        working_memory.protect("约束二：新增代码尽量加中文注释", entry_type="project_constraint", importance=0.8)
        working_memory.protect("任务一：重构 full compact", entry_type="active_task", importance=0.9)
        working_memory.protect("任务二：补 protected memory", entry_type="active_task", importance=0.8)
        working_memory.protect("决策一：压缩阶段优先保留结构化事实", entry_type="key_decision", importance=0.9)
        working_memory.protect("风险一：tool_result 会挤占 recent window", entry_type="recent_risk", importance=0.9)
        working_memory.protect("文件发现：已确认 context_auto_compact.py 负责 full compact", entry_type="reflection_file", importance=0.8)

        snapshot = working_memory.build_protected_snapshot()
        prompt_text = working_memory.format_for_prompt()

        self.assertEqual(len(snapshot["preferences"]), 2)
        self.assertNotIn("偏好三：这条应该被挤掉", snapshot["preferences"])
        self.assertIn("任务一：重构 full compact", snapshot["active_tasks"])
        self.assertIn("决策一：压缩阶段优先保留结构化事实", snapshot["decisions"])
        self.assertIn("风险一：tool_result 会挤占 recent window", snapshot["open_issues"])
        self.assertIn("文件发现：已确认 context_auto_compact.py 负责 full compact", snapshot["tool_findings"])
        self.assertIn("用户偏好：", prompt_text)
        self.assertIn("项目约束：", prompt_text)
        self.assertIn("关键工具发现：", prompt_text)

    def test_resolve_recent_risks_filters_directory_tree_markdown_and_file_body_noise(self) -> None:
        from app.context_signal_resolver import resolve_recent_risks

        working_memory = WorkingMemory()
        working_memory.protect("│ ├── context_compactor.py ← 压缩器主模块", entry_type="recent_risk")
        working_memory.protect("| 模块 | 作用 |", entry_type="recent_risk")
        working_memory.protect("def run_agent_once(): return 'not a risk body'", entry_type="error_context")
        working_memory.protect("真正风险：大 tool_result 会挤占 recent window", entry_type="recent_risk")

        risks = resolve_recent_risks(working_memory=working_memory)

        self.assertEqual(risks, ["真正风险：大 tool_result 会挤占 recent window"])

    def test_prepare_agent_context_records_auto_compact_history_when_pressure_remains_high(self) -> None:
        from app.context_runtime import prepare_agent_context
        from app.context_state import load_context_state

        with tempfile.TemporaryDirectory() as tmpdir:
            tool_registry = _FakeToolRegistry()
            memory_pipeline = _RecordingMemoryPipeline()
            session = create_new_session(tmpdir)
            session.extra["usable_context_budget"] = 1000
            working_memory = WorkingMemory()
            working_memory.protect("继续整理上下文压缩链路", entry_type="user_intent")

            full_history = [
                {"role": "user", "content": "分析上下文管理" * 40},
                {"role": "assistant", "content": "先检查最近窗口和工具结果预算" * 30},
                {"role": "user", "content": "继续看自动压缩触发条件" * 40},
                {"role": "assistant", "content": "需要补充 session memory compact 和 full compact" * 20},
                {"role": "user", "content": "再确认恢复路径" * 40},
                {"role": "assistant", "content": "恢复时应该重试而不是直接失败" * 25},
            ]

            prepared = prepare_agent_context(
                full_history=full_history,
                session=session,
                tool_registry=tool_registry,
                working_memory=working_memory,
                memory_pipeline=memory_pipeline,
                history_summarizer=None,
            )

            state = load_context_state(tmpdir, session.session_id)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertLess(prepared.stats.total_tokens, prepared.preview_stats.total_tokens)
            self.assertTrue(state.compaction_history)
            self.assertTrue(state.compaction_history[-1].get("auto_compact_applied"))
            self.assertEqual(state.compaction_history[-1].get("auto_compact_strategy"), "full")
            self.assertTrue(
                any(
                    message.get("role") == "system"
                    and "全量压缩" in str(message.get("content", ""))
                    for message in state.compacted_messages
                )
            )

    def test_run_auto_compact_uses_compact_memory_context_in_summary_base(self) -> None:
        from app.context_auto_compact import _run_session_memory_compact

        messages = [
            {"role": "user", "content": "第0条消息：当前要验证 session memory compact 优先使用压缩记忆基线。" * 8},
            {"role": "assistant", "content": "第1条消息：当前要验证 session memory compact 优先使用压缩记忆基线。" * 8},
            {"role": "user", "content": "第2条消息：当前要验证 session memory compact 优先使用压缩记忆基线。" * 8},
            {"role": "assistant", "content": "第3条消息：当前要验证 session memory compact 优先使用压缩记忆基线。" * 8},
            {"role": "user", "content": "第4条消息：当前要验证 session memory compact 优先使用压缩记忆基线。" * 8},
        ]

        result = _run_session_memory_compact(
            messages=messages,
            usable_budget=500,
            summary_base="MEMORY_BASELINE_TOKEN\n1. 当前目标是复用上一次压缩基线，而不是重新拼 older_history_summary。",
            summary_snapshot=None,
            fixed_overhead_tokens=0,
        )

        self.assertTrue(result.messages)
        self.assertEqual(result.messages[0]["role"], "system")
        self.assertIn("MEMORY_BASELINE_TOKEN", str(result.messages[0]["content"]))

    def test_run_auto_compact_full_summary_uses_event_sections(self) -> None:
        from app.context_auto_compact import run_auto_compact

        messages = [
            {"role": "user", "content": "目标：确认压缩前为什么 recent window 会被重复扫描撑爆。" * 6},
            {"role": "assistant_tool_call", "tool_name": "list_files", "content": "{\"path\": \".\"}"},
            {
                "role": "tool_result",
                "tool_name": "list_files",
                "content": "ROOT: .\nTOTAL_ENTRIES: 120\nRETURNED_ENTRIES: 120\nTRUNCATED: no\n\ndir app\ndir tests",
            },
            {"role": "assistant", "content": "中间结论：目录扫描结果重复注入，导致工具结果占比过高。" * 4},
            {"role": "user", "content": "再确认 grep 结果有没有重复结论。" * 6},
            {"role": "assistant_tool_call", "tool_name": "grep_files", "content": "{\"pattern\": \"compact\"}"},
            {
                "role": "tool_result",
                "tool_name": "grep_files",
                "is_error": True,
                "content": "SEARCH_ROOT: app\nPATTERN: compact\nERROR: output budget hit",
            },
            {"role": "assistant", "content": "待处理：需要避免把同类扫描结果反复带入后续轮次。" * 4},
            {"role": "user", "content": "最后整理成压缩策略。"},
        ]

        result = run_auto_compact(
            messages=messages,
            usable_budget=450,
            summary_base="结构化压缩记忆\n## 用户偏好\n- 回答尽量直接",
            fixed_overhead_tokens=0,
            force_full=True,
        )

        marker = str(result.messages[0]["content"])
        self.assertTrue(result.applied)
        self.assertIn("结构化压缩记忆", marker)
        self.assertNotIn("## 当前任务", marker)
        self.assertIn("## 关键决策", marker)
        self.assertIn("## 关键工具发现（上次压缩延续）", marker)

    def test_limit_summary_text_can_use_token_budget_for_full_compact(self) -> None:
        from app.context_auto_compact import _limit_summary_text

        summary_text = "\n".join(
            [
                "结构化压缩记忆",
                "压缩记忆基线",
                *[
                    f"- 第{index}条结论："
                    "context_auto_compact.py 需要按 token 预算保留结构化摘要，"
                    "避免因为字符硬截断而把最后一条关键决策截半。"
                    "同时要让 full compact 先保留 decisions，再考虑 tool_findings，"
                    "这样在高压上下文里也能把真正影响回答的语义核心保下来。"
                    for index in range(1, 9)
                ],
            ]
        )

        self.assertGreater(len(summary_text), 960)

        token_limited = _limit_summary_text(
            summary_text,
            2000,
            by_tokens=True,
        )
        char_limited = _limit_summary_text(summary_text, 960)

        self.assertEqual(token_limited, summary_text.strip())
        self.assertIn("[摘要已截断]", char_limited)

    def test_run_auto_compact_full_summary_can_fold_old_tool_rounds_into_summary(self) -> None:
        from app.context_auto_compact import run_auto_compact

        messages = [
            {"role": "user", "content": "先读取 alpha 并保留其核心结论。" * 8},
            {"role": "assistant_tool_call", "tool_name": "read_file", "content": "{\"path\": \"tmp/alpha.txt\"}"},
            {
                "role": "tool_result",
                "tool_name": "read_file",
                "content": (
                    "FILE: tmp/alpha.txt\n"
                    "ALPHA_FACT: recent window 压力主要来自重复扫描的 tool_result\n"
                    + ("001. Alpha filler: 继续放大上下文压力。\n" * 30)
                ),
            },
            {"role": "user", "content": "再读取 beta，并在后续高压时检查 alpha 结论是否仍能进入 summary。" * 8},
            {"role": "assistant_tool_call", "tool_name": "read_file", "content": "{\"path\": \"tmp/beta.txt\"}"},
            {
                "role": "tool_result",
                "tool_name": "read_file",
                "content": (
                    "FILE: tmp/beta.txt\n"
                    "BETA_FACT: protected working memory 应与 resolved_project_constraints 分离\n"
                    + ("001. Beta filler: 继续放大上下文压力。\n" * 30)
                ),
            },
            {"role": "assistant_tool_call", "tool_name": "read_file", "content": "{\"path\": \"tmp/filler.txt\"}"},
            {
                "role": "tool_result",
                "tool_name": "read_file",
                "content": (
                    "FILE: tmp/filler.txt\n"
                    "FILLER_HEADER: 这里只负责把 alpha 和 beta 一起逼进 full compact summary\n"
                    + ("001. filler: 继续拉高上下文压力。\n" * 40)
                ),
            },
            {"role": "user", "content": "最后再加入一条 filler，把旧工具轮次逼进 full compact summary。" * 8},
        ]

        result = run_auto_compact(
            messages=messages,
            usable_budget=420,
            summary_base="结构化压缩记忆\n## 用户偏好\n- 回答尽量直接",
            fixed_overhead_tokens=0,
            force_full=True,
        )

        self.assertTrue(result.messages)
        marker = str(result.messages[0]["content"])
        self.assertIn("重复扫描的 tool_result", marker)
        self.assertIn("resolved_project_constraints 分离", marker)

    def test_build_event_compact_memory_snapshot_prefers_semantic_tool_findings(self) -> None:
        from app.context_compact_memory import build_event_compact_memory_snapshot

        snapshot = build_event_compact_memory_snapshot(
            removed_messages=[
                {"role": "user", "content": "请分析为什么上下文压缩后丢掉了核心结论。"},
                {"role": "assistant_tool_call", "tool_name": "read_file", "content": "{\"path\": \"tmp/demo.txt\"}"},
                {
                    "role": "tool_result",
                    "tool_name": "read_file",
                    "content": (
                        "FILE: tmp/demo.txt\n"
                        "OFFSET: 0\n"
                        "TOTAL_CHARS: 4096\n"
                        "ALPHA_FACT_1: recent window 压力主要来自重复扫描的 tool_result\n"
                        "ALPHA_FACT_2: 工具输出要前置裁剪，避免脏文本先进入长期上下文\n"
                        "001. Alpha filler: 这是一段没有语义价值的填充文本\n"
                    ),
                },
                {
                    "role": "tool_result",
                    "tool_name": "read_file",
                    "content": (
                        "FILE: tmp/second.txt\n"
                        "BETA_FACT_1: full compact 应优先保留 active_tasks decisions open_issues tool_findings\n"
                        "BETA_FACT_2: protected working memory 应与 resolved_project_constraints 分离\n"
                    ),
                },
            ]
        )

        self.assertIn("tool_findings", snapshot)
        self.assertTrue(
            any("重复扫描的 tool_result" in item for item in snapshot["tool_findings"])
        )
        self.assertTrue(
            any("前置裁剪" in item for item in snapshot["tool_findings"])
        )
        self.assertTrue(
            any("resolved_project_constraints 分离" in item for item in snapshot["tool_findings"])
        )
        self.assertFalse(
            any("tmp/demo.txt" == item.strip() for item in snapshot["tool_findings"])
        )
        self.assertGreaterEqual(len(snapshot["tool_findings"]), 4)

    def test_run_auto_compact_full_summary_exposes_summary_snapshot(self) -> None:
        from app.context_auto_compact import run_auto_compact

        messages = [
            {"role": "user", "content": "目标：确认压缩后仍能保留核心语义。" * 12},
            {"role": "assistant_tool_call", "tool_name": "read_file", "content": "{\"path\": \"tmp/demo.txt\"}"},
            {
                "role": "tool_result",
                "tool_name": "read_file",
                "content": (
                    "FILE: tmp/demo.txt\n"
                    "CORE_FACT: protected working memory 应与 resolved_project_constraints 分离\n"
                    + ("补充说明：这一段用于拉高上下文压力并逼出 full compact。\n" * 18)
                ),
            },
            {"role": "assistant", "content": "结论：需要把 full compact 产出的语义快照回写到下一轮基线。" * 10},
            {"role": "user", "content": "继续压缩，并确认最近窗口里还能保住语义结论。" * 10},
            {"role": "assistant", "content": "最近窗口只负责保留尾部，较早结论应该沉淀进 compact summary。" * 10},
            {"role": "user", "content": "最后再检查 compact summary 是否真的包含 earlier fact。" * 10},
        ]

        result = run_auto_compact(
            messages=messages,
            usable_budget=320,
            summary_base="结构化压缩记忆\n## 用户偏好\n- 回答尽量直接",
            fixed_overhead_tokens=0,
            force_full=True,
        )

        self.assertIsNotNone(result.summary_snapshot)
        assert result.summary_snapshot is not None
        self.assertTrue(
            any(
                "resolved_project_constraints 分离" in item
                for item in result.summary_snapshot.get("tool_findings", [])
            )
        )
        self.assertTrue(
            any(
                "回写到下一轮基线" in item
                for item in result.summary_snapshot.get("decisions", [])
            )
        )
        self.assertTrue(result.summary_text.strip())

    def test_run_auto_compact_strips_previous_internal_markers(self) -> None:
        from app.context_auto_compact import run_auto_compact

        messages = [
            {"role": "system", "content": "[全量压缩]\n旧 marker A"},
            {"role": "system", "content": "[恢复压缩]\n旧 marker B"},
            {"role": "user", "content": "继续整理这一轮真实上下文"},
            {"role": "assistant", "content": "需要确认旧压缩 marker 不会递归进入新摘要"},
        ]

        result = run_auto_compact(
            messages=messages,
            usable_budget=500,
            summary_base="结构化压缩记忆",
            fixed_overhead_tokens=0,
            force_full=True,
        )

        marker_count = sum(
            1
            for message in result.messages
            if message.get("role") == "system" and "[全量压缩]" in str(message.get("content", ""))
        )
        self.assertEqual(marker_count, 1)
        self.assertFalse(
            any(
                message.get("role") == "system"
                and "[恢复压缩]" in str(message.get("content", ""))
                for message in result.messages
            )
        )

    def test_should_trigger_auto_compact_can_fire_early_for_tool_result_pressure(self) -> None:
        from app.context_auto_compact import should_trigger_auto_compact

        triggered = should_trigger_auto_compact(
            total_tokens=720,
            usable_budget=1000,
            tool_result_tokens=460,
            repeated_scan_count=3,
        )

        self.assertTrue(triggered)

    def test_normalize_tool_call_pairs_drops_orphan_tool_messages(self) -> None:
        from app.context_message_safety import normalize_tool_call_pairs

        messages = [
            {"role": "assistant_tool_call", "tool_use_id": "call-1", "tool_name": "read_file", "input": {"path": "a.py"}},
            {"role": "tool_result", "tool_use_id": "call-1", "tool_name": "read_file", "content": "ok"},
            {"role": "tool_result", "tool_use_id": "call-orphan", "tool_name": "read_file", "content": "orphan"},
            {"role": "assistant_tool_call", "tool_use_id": "call-missing", "tool_name": "read_file", "input": {"path": "b.py"}},
            {"role": "user", "content": "继续下一步"},
        ]

        normalized = normalize_tool_call_pairs(messages)

        self.assertEqual(
            [message.get("role") for message in normalized],
            ["assistant_tool_call", "tool_result", "user"],
        )
        self.assertFalse(
            any(str(message.get("tool_use_id", "")) == "call-orphan" for message in normalized)
        )
        self.assertFalse(
            any(str(message.get("tool_use_id", "")) == "call-missing" for message in normalized)
        )

    def test_reactive_tail_recover_strips_old_compaction_markers_before_retry(self) -> None:
        from app.context_manager import estimate_messages_tokens
        from app.context_reactive_compact import _aggressive_tail_recover

        messages = [
            {"role": "system", "content": "基础 system prompt\n" + ("规则说明" * 80)},
            {
                "role": "system",
                "content": "[全量压缩]\n已折叠较早消息数：9\n\n## 对话摘要\n旧摘要\n\n--- 最近对话继续如下 ---",
            },
            {"role": "assistant", "content": "较长的中间分析结论" * 60},
            {"role": "user", "content": "继续保留最后一个问题"},
        ]

        tokens_before = estimate_messages_tokens(messages)
        result = _aggressive_tail_recover(messages=messages, usable_budget=700)
        tokens_after = estimate_messages_tokens(result)

        self.assertLess(tokens_after, tokens_before)
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "基础 system prompt" in str(message.get("content", ""))
                for message in result
            )
        )
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "恢复压缩" in str(message.get("content", ""))
                for message in result
            )
        )
        self.assertFalse(
            any(
                message.get("role") == "system"
                and "[全量压缩]" in str(message.get("content", ""))
                for message in result
            )
        )

    def test_prepare_agent_context_restores_compact_memory_context_from_cached_state(self) -> None:
        from app.context_runtime import prepare_agent_context
        from app.context_state import ContextStateData, build_history_fingerprint, load_context_state, save_context_state

        with tempfile.TemporaryDirectory() as tmpdir:
            tool_registry = _FakeToolRegistry()
            session = create_new_session(tmpdir)
            session.extra["usable_context_budget"] = 700
            working_memory = WorkingMemory()
            working_memory.protect("继续处理上下文压缩恢复", entry_type="user_intent")

            cached_compact_memory_context = (
                "压缩记忆基线\n"
                "## 稳定事实\n"
                "1. 当前目标是复用上一次压缩基线，而不是重新拼 older_history_summary。"
            )
            save_context_state(
                tmpdir,
                ContextStateData(
                    session_id=session.session_id,
                    source_message_count=0,
                    source_history_fingerprint=build_history_fingerprint([]),
                    compacted_messages=[],
                    older_history_summary="旧摘要：这段文字不应该优先进入会话压缩摘要。",
                    compact_memory_context=cached_compact_memory_context,
                ),
            )

            full_history = [
                {"role": "user", "content": "先整理旧对话摘要和压缩状态的关系，确认恢复阶段不能直接复制整段 older_history_summary。" * 10},
                {"role": "assistant", "content": "需要把压缩用基线和普通记忆注入拆开，否则摘要会重复膨胀。" * 9},
                {"role": "user", "content": "再检查命中 context_state 之后，为什么 session memory compact 应该优先复用已有压缩基线。" * 10},
                {"role": "assistant", "content": "因为这份基线已经是上次压缩阶段沉淀过的信息，比重新拼接更稳定。" * 9},
                {"role": "user", "content": "同时还要保证最近几轮完整消息继续保留，不然模型会丢掉紧邻当前问题的上下文。" * 10},
                {"role": "assistant", "content": "所以 session compact 只应该折叠较早消息，把尾部 recent window 继续保留下来。" * 9},
                {"role": "user", "content": "如果恢复后又遇到高压，就继续用 compact_memory_context 作为摘要基线，而不是回退到旧摘要。" * 10},
                {"role": "assistant", "content": "最后还要确认新的 context_state 会继续保存这份基线，供下一轮同 session 复用。" * 9},
            ]

            prepared = prepare_agent_context(
                full_history=full_history,
                session=session,
                tool_registry=tool_registry,
                working_memory=working_memory,
                memory_pipeline=None,
                history_summarizer=None,
            )

            state = load_context_state(tmpdir, session.session_id)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertTrue(
                any(
                    "复用上一次压缩基线" in item
                    for item in prepared.compact_memory_snapshot.get("tool_findings", [])
                )
            )
            self.assertTrue(
                any(
                    "复用上一次压缩基线" in item
                    for item in state.compact_memory_snapshot.get("tool_findings", [])
                )
            )
            self.assertTrue(state.compacted_messages)
            self.assertIn("preview_total", state.last_token_stats)

    def test_prepare_agent_context_persists_full_compact_semantic_snapshot(self) -> None:
        from app.context_runtime import prepare_agent_context
        from app.context_state import load_context_state

        with tempfile.TemporaryDirectory() as tmpdir:
            tool_registry = _FakeToolRegistry()
            session = create_new_session(tmpdir)
            session.extra["usable_context_budget"] = 1400
            working_memory = WorkingMemory()
            working_memory.protect("验证 full compact 后的新基线是否真的保留核心语义", entry_type="user_intent")

            full_history = [
                {"role": "user", "content": "请确认为什么现在压缩后只能保留路径，保不住结论。" * 40},
                {"role": "assistant_tool_call", "tool_name": "read_file", "content": "{\"path\": \"tmp/demo.txt\"}"},
                {
                    "role": "tool_result",
                    "tool_name": "read_file",
                    "content": (
                        "FILE: tmp/demo.txt\n"
                        "OFFSET: 0\n"
                        "CORE_FACT_1: recent window 压力主要来自重复扫描的 tool_result\n"
                        "CORE_FACT_2: protected working memory 应与 resolved_project_constraints 分离\n"
                        + ("补充说明：这一段用于显著抬高上下文压力，确保 prepare_agent_context 进入全量压缩。\n" * 60)
                    ),
                },
                {"role": "assistant", "content": "结论：需要把 full compact 产出的语义快照回写到 state，不能继续沿用旧 baseline。" * 36},
                {"role": "user", "content": "继续确认下一轮是否还能看到这两个核心事实。" * 40},
                {"role": "assistant", "content": "还要避免 tool_result 前缀元信息把真正结论挤掉。" * 36},
            ]

            prepared = prepare_agent_context(
                full_history=full_history,
                session=session,
                tool_registry=tool_registry,
                working_memory=working_memory,
                memory_pipeline=None,
                history_summarizer=None,
            )

            state = load_context_state(tmpdir, session.session_id)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.compaction_history[-1].get("auto_compact_strategy"), "full")
            self.assertIn("resolved_project_constraints 分离", state.compact_memory_context)
            self.assertIn("回写到 state", state.compact_memory_context)
            self.assertIn("resolved_project_constraints 分离", prepared.compact_memory_context)
            self.assertIn("重复扫描的 tool_result", prepared.compact_memory_context)

    def test_prepare_agent_context_auto_compact_handles_large_text_history(self) -> None:
        from app.context_runtime import prepare_agent_context
        from app.context_state import load_context_state

        with tempfile.TemporaryDirectory() as tmpdir:
            tool_registry = _FakeToolRegistry()
            session = create_new_session(tmpdir)
            session.extra["usable_context_budget"] = 1000
            working_memory = WorkingMemory()
            working_memory.protect("分析纯文本 recent window 的压缩效果", entry_type="user_intent")

            full_history: list[dict[str, object]] = []
            for index in range(12):
                full_history.append(
                    {
                        "role": "user",
                        "content": (f"第{index + 1}轮分析：继续总结上下文预算、memory 注入和压缩阈值的关系。") * 6,
                    }
                )
                full_history.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"第{index + 1}轮结论：当前主要压力来自 recent window、tool_result 和 older history summary，"
                            "需要继续观察 auto compact dispatcher 与 reactive recover。"
                        )
                        * 6,
                    }
                )

            prepared = prepare_agent_context(
                full_history=full_history,
                session=session,
                tool_registry=tool_registry,
                working_memory=working_memory,
                memory_pipeline=None,
                history_summarizer=None,
            )

            state = load_context_state(tmpdir, session.session_id)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertTrue(state.compaction_history[-1].get("auto_compact_applied"))
            self.assertLess(prepared.stats.usage_ratio, 0.85)
            self.assertTrue(
                any(
                    message.get("role") == "system"
                    and "压缩" in str(message.get("content", ""))
                    for message in state.compacted_messages
                )
            )

    def test_agent_loop_passes_shrunk_memory_budget_under_high_context_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _FakeModel()
            tool_registry = _FakeToolRegistry()
            memory_pipeline = _RecordingMemoryPipeline()
            session = create_new_session(tmpdir)
            session.extra["usable_context_budget"] = 3000
            working_memory = WorkingMemory()
            working_memory.protect("整理仓库上下文", entry_type="user_intent")

            long_tool_result = "日志" * 2500
            history = [
                {"role": "user", "content": "请分析这个仓库"},
                {"role": "tool_result", "tool_name": "read_file", "content": long_tool_result},
            ]

            step, next_history = run_agent_once(
                user_input="继续分析并总结",
                model=model,
                tool_registry=tool_registry,
                tool_context=ToolContext(cwd=tmpdir),
                session=session,
                working_memory=working_memory,
                memory_pipeline=memory_pipeline,
                history=history,
                max_steps=1,
                session_id="sess-test",
            )

            self.assertEqual(step.type, "assistant")
            self.assertTrue(next_history)
            self.assertEqual(len(memory_pipeline.calls), 1)
            self.assertEqual(memory_pipeline.calls[0]["top_k"], 1)
            self.assertLessEqual(memory_pipeline.calls[0]["max_memory_chars_per_item"], 80)

    def test_agent_loop_retries_after_reactive_compact_on_context_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _OverflowThenSuccessModel()
            tool_registry = _FakeToolRegistry()
            memory_pipeline = _RecordingMemoryPipeline()
            session = create_new_session(tmpdir)
            session.extra["usable_context_budget"] = 800
            working_memory = WorkingMemory()
            working_memory.protect("处理上下文超长恢复", entry_type="user_intent")

            history = [
                {"role": "user", "content": "分析最近的上下文压力" * 30},
                {"role": "assistant", "content": "最近窗口里存在很多工具结果和说明文本" * 25},
                {"role": "tool_result", "tool_name": "read_file", "content": "日志" * 2000},
            ]

            step, next_history = run_agent_once(
                user_input="继续回答并在必要时恢复压缩",
                model=model,
                tool_registry=tool_registry,
                tool_context=ToolContext(cwd=tmpdir),
                session=session,
                working_memory=working_memory,
                memory_pipeline=memory_pipeline,
                history=history,
                max_steps=1,
                session_id="sess-reactive-recover",
            )

            self.assertEqual(step.type, "assistant")
            self.assertEqual(step.content, "recovered")
            self.assertTrue(next_history)
            self.assertEqual(len(model.calls), 2)
            self.assertGreater(len(model.calls[0]), 0)
            self.assertGreater(len(model.calls[1]), 0)

    def test_agent_loop_writes_context_output_into_history_for_tool_results(self) -> None:
        class _SingleToolCallModel:
            def __init__(self) -> None:
                self.call_count = 0
                self.calls: list[list[dict[str, object]]] = []

            def next(self, messages, on_stream_chunk=None, store=None):  # type: ignore[no-untyped-def]
                self.calls.append(list(messages))
                self.call_count += 1
                if self.call_count == 1:
                    return AgentStep(
                        type="tool_calls",
                        calls=[
                            {
                                "id": "tool-1",
                                "tool_name": "grep_files",
                                "input": {"path": ".", "pattern": "compact"},
                            }
                        ],
                    )
                return AgentStep(type="assistant", content="done", kind="final")

        class _ContextOutputToolRegistry:
            def list_tool_name(self) -> list[str]:
                return ["grep_files"]

            def execute_tool(self, tool_name: str, input_data: object, context: object) -> object:
                return ToolResult(
                    ok=True,
                    output="较长工具输出正文" * 80,
                    meta={
                        "context_output": "结构化工具摘要：grep_files 命中过多，已保留首尾样本",
                        "raw_output": "较长工具输出正文" * 120,
                        "truncated": True,
                    },
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            session = create_new_session(tmpdir)
            working_memory = WorkingMemory()
            model = _SingleToolCallModel()
            step, next_history = run_agent_once(
                user_input="继续检查 grep 结果",
                model=model,
                tool_registry=_ContextOutputToolRegistry(),  # type: ignore[arg-type]
                tool_context=ToolContext(cwd=tmpdir),
                session=session,
                working_memory=working_memory,
                memory_pipeline=None,
                max_steps=2,
                session_id="sess-context-output",
            )

            self.assertEqual(step.type, "assistant")
            tool_results = [
                message for message in model.calls[1]
                if message.get("role") == "tool_result"
            ]
            self.assertEqual(len(tool_results), 1)
            self.assertEqual(
                tool_results[0]["content"],
                "结构化工具摘要：grep_files 命中过多，已保留首尾样本",
            )
            self.assertIn("raw_output", tool_results[0]["meta"])


if __name__ == "__main__":
    unittest.main()
