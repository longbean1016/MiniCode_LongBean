from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from app.agent_loop import run_agent_once
from app.memory_models import MemoryContextResult
from app.session import create_new_session
from app.types import AgentStep, ToolContext
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
            self.assertTrue(
                any(
                    message.get("role") == "system"
                    and "恢复压缩" in str(message.get("content", ""))
                    for message in model.calls[1]
                )
            )


if __name__ == "__main__":
    unittest.main()
