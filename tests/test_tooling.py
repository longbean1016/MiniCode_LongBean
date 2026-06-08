from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.tools import build_tool_registry
from app.agent.tooling import ToolDefinition, ToolRegistry
from app.types import ToolContext, ToolResult


class ToolBehaviorTests(unittest.TestCase):
    def test_list_files_reports_total_and_truncated_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for index in range(250):
                (Path(tmpdir) / f"file_{index:03d}.txt").write_text("x", encoding="utf-8")

            registry = build_tool_registry()
            result = registry.execute_tool(
                "list_files",
                {"path": "."},
                ToolContext(cwd=tmpdir),
            )

            self.assertTrue(result.ok)
            self.assertIn("ROOT: .", result.output)
            self.assertIn("TOTAL_ENTRIES: 250", result.output)
            self.assertIn("RETURNED_ENTRIES: 200", result.output)
            self.assertIn("TRUNCATED: yes", result.output)
            self.assertIn("file file_000.txt", result.output)

    def test_list_files_omits_internal_context_entries_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".cache").mkdir()
            (Path(tmpdir) / ".sessions").mkdir()
            (Path(tmpdir) / "app.py").write_text("print('x')", encoding="utf-8")

            registry = build_tool_registry()
            result = registry.execute_tool(
                "list_files",
                {"path": "."},
                ToolContext(cwd=tmpdir),
            )

            self.assertTrue(result.ok)
            self.assertIn("OMITTED_INTERNAL_ENTRIES: 2", result.output)
            self.assertIn("file app.py", result.output)
            self.assertNotIn(".cache", result.output)
            self.assertNotIn(".sessions", result.output)

    def test_list_files_blocks_internal_context_directories_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            internal_dir = Path(tmpdir) / ".context_state"
            internal_dir.mkdir()

            registry = build_tool_registry()
            result = registry.execute_tool(
                "list_files",
                {"path": ".context_state"},
                ToolContext(cwd=tmpdir),
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error, "LIST_POLICY_BLOCKED")
            self.assertIn("默认不允许列出内部上下文目录", result.output)

    def test_read_file_supports_offset_and_limit_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sample.txt"
            target.write_text("abcdefghij", encoding="utf-8")

            registry = build_tool_registry()
            result = registry.execute_tool(
                "read_file",
                {"path": "sample.txt", "offset": 2, "limit": 4},
                ToolContext(cwd=tmpdir),
            )

            self.assertTrue(result.ok)
            self.assertIn("FILE: sample.txt", result.output)
            self.assertIn("OFFSET: 2", result.output)
            self.assertIn("END: 6", result.output)
            self.assertIn("TOTAL_CHARS: 10", result.output)
            self.assertIn("TRUNCATED: yes", result.output)
            self.assertTrue(result.output.rstrip().endswith("cdef"))

    def test_grep_files_reports_total_and_truncated_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sample.txt"
            lines = [f"match line {index}" for index in range(250)]
            target.write_text("\n".join(lines), encoding="utf-8")

            registry = build_tool_registry()
            result = registry.execute_tool(
                "grep_files",
                {"path": ".", "pattern": "match"},
                ToolContext(cwd=tmpdir),
            )

            self.assertTrue(result.ok)
            self.assertIn("PATTERN: match", result.output)
            self.assertIn("ROOT: .", result.output)
            self.assertIn("TOTAL_MATCHES: 250", result.output)
            self.assertIn("RETURNED_MATCHES: 200", result.output)
            self.assertIn("TRUNCATED: yes", result.output)
            self.assertIn("sample.txt:1: match line 0", result.output)

    def test_grep_files_blocks_internal_context_directories_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            internal_dir = Path(tmpdir) / ".sessions"
            internal_dir.mkdir()
            (internal_dir / "demo.txt").write_text("match", encoding="utf-8")

            registry = build_tool_registry()
            result = registry.execute_tool(
                "grep_files",
                {"path": ".sessions", "pattern": "match"},
                ToolContext(cwd=tmpdir),
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error, "SEARCH_POLICY_BLOCKED")
            self.assertIn("默认不允许搜索内部上下文目录", result.output)

    def test_grep_files_clips_single_match_line_and_applies_output_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sample.txt"
            long_line = "match " + ("x" * 2000)
            target.write_text("\n".join(long_line for _ in range(80)), encoding="utf-8")

            registry = build_tool_registry()
            result = registry.execute_tool(
                "grep_files",
                {"path": ".", "pattern": "match", "max_matches": 80},
                ToolContext(cwd=tmpdir),
            )

            self.assertTrue(result.ok)
            self.assertIn("CONTENT_CLIPPED: yes", result.output)
            self.assertIn("OUTPUT_BUDGET_HIT: yes", result.output)
            self.assertIn("单行已截断", result.output)
            self.assertIn("TRUNCATED: yes", result.output)

    def test_read_file_falls_back_for_non_utf8_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sample_gbk.txt"
            target.write_bytes("中文内容\n第二行".encode("gbk"))

            registry = build_tool_registry()
            result = registry.execute_tool(
                "read_file",
                {"path": "sample_gbk.txt", "offset": 0, "limit": 200},
                ToolContext(cwd=tmpdir),
            )

            self.assertTrue(result.ok)
            self.assertIn("FILE: sample_gbk.txt", result.output)
            self.assertIn("中文内容", result.output)
            self.assertIn("第二行", result.output)

    def test_read_file_blocks_tool_result_cache_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache"
            cache_dir.mkdir()
            target = cache_dir / "tool_result_demo.txt"
            target.write_text("tool output", encoding="utf-8")

            registry = build_tool_registry()
            result = registry.execute_tool(
                "read_file",
                {"path": ".cache/tool_result_demo.txt"},
                ToolContext(cwd=tmpdir),
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error, "READ_POLICY_BLOCKED")
            self.assertIn("默认不允许读取", result.output)
            self.assertIn(".cache/tool_result_demo.txt", result.output)

    def test_read_file_blocks_session_history_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / ".sessions"
            sessions_dir.mkdir()
            target = sessions_dir / "demo.json"
            target.write_text("{}", encoding="utf-8")

            registry = build_tool_registry()
            result = registry.execute_tool(
                "read_file",
                {"path": ".sessions/demo.json"},
                ToolContext(cwd=tmpdir),
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error, "READ_POLICY_BLOCKED")
            self.assertIn("默认不允许读取", result.output)
            self.assertIn(".sessions/demo.json", result.output)

    def test_read_file_blocks_context_state_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / ".context_state"
            state_dir.mkdir()
            target = state_dir / "demo.json"
            target.write_text("{}", encoding="utf-8")

            registry = build_tool_registry()
            result = registry.execute_tool(
                "read_file",
                {"path": ".context_state/demo.json"},
                ToolContext(cwd=tmpdir),
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error, "READ_POLICY_BLOCKED")
            self.assertIn("默认不允许读取", result.output)
            self.assertIn(".context_state/demo.json", result.output)

    def test_read_file_breaks_same_range_repeat_reads_within_same_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sample.txt"
            target.write_text("abcdefghij", encoding="utf-8")

            registry = build_tool_registry()
            context = ToolContext(cwd=tmpdir)

            first = registry.execute_tool(
                "read_file",
                {"path": "sample.txt", "offset": 0, "limit": 4},
                context,
            )
            second = registry.execute_tool(
                "read_file",
                {"path": "sample.txt", "offset": 0, "limit": 4},
                context,
            )

            self.assertTrue(first.ok)
            self.assertFalse(second.ok)
            self.assertEqual(second.error, "READ_REPEAT_BLOCKED")
            self.assertIn("同一轮里已经读取过相同区间", second.output)
            self.assertIn("sample.txt", second.output)

    def test_read_file_allows_different_ranges_within_same_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sample.txt"
            target.write_text("abcdefghij", encoding="utf-8")

            registry = build_tool_registry()
            context = ToolContext(cwd=tmpdir)

            first = registry.execute_tool(
                "read_file",
                {"path": "sample.txt", "offset": 0, "limit": 4},
                context,
            )
            second = registry.execute_tool(
                "read_file",
                {"path": "sample.txt", "offset": 4, "limit": 4},
                context,
            )

            self.assertTrue(first.ok)
            self.assertTrue(second.ok)
            self.assertIn("efgh", second.output)

    def test_tool_registry_keeps_raw_output_when_smart_truncated(self) -> None:
        def _validate(input_data: object) -> dict[str, str]:
            return {}

        def _run(_: dict[str, str], __: ToolContext) -> ToolResult:
            output = "\n".join(f"ERROR line {index} {'x' * 40}" for index in range(1200))
            return ToolResult(ok=True, output=output)

        registry = ToolRegistry(
            tools=[
                ToolDefinition(
                    name="run_command",
                    description="test tool",
                    validator=_validate,
                    runner=_run,
                    input_schema={},
                )
            ]
        )

        result = registry.execute_tool("run_command", {}, ToolContext(cwd="."))

        self.assertTrue(result.ok)
        self.assertTrue(result.meta["truncated"])
        self.assertIn("raw_output", result.meta)
        self.assertGreater(len(str(result.meta["raw_output"])), len(result.output))
        self.assertIn("ERROR line 0", result.output)
        self.assertIn("ERROR line 1199", str(result.meta["raw_output"]))
        self.assertIn("context_output", result.meta)
        self.assertLessEqual(len(str(result.meta["context_output"])), len(str(result.meta["raw_output"])))

    def test_tool_registry_smart_truncates_grep_files_but_keeps_header_and_tail(self) -> None:
        def _validate(input_data: object) -> dict[str, str]:
            return {}

        def _run(_: dict[str, str], __: ToolContext) -> ToolResult:
            header = (
                "PATTERN: session\n"
                "ROOT: .\n"
                "TOTAL_MATCHES: 600\n"
                "RETURNED_MATCHES: 600\n"
                "TRUNCATED: no\n\n"
            )
            body = "\n".join(
                f"src/file_{index:03d}.py:{index + 1}: {'x' * 60} session marker {index}"
                for index in range(600)
            )
            return ToolResult(ok=True, output=header + body)

        registry = ToolRegistry(
            tools=[
                ToolDefinition(
                    name="grep_files",
                    description="test grep",
                    validator=_validate,
                    runner=_run,
                    input_schema={},
                )
            ]
        )

        result = registry.execute_tool("grep_files", {}, ToolContext(cwd="."))

        self.assertTrue(result.ok)
        self.assertTrue(result.meta["truncated"])
        self.assertIn("PATTERN: session", result.output)
        self.assertIn("TOTAL_MATCHES: 600", result.output)
        self.assertIn("src/file_000.py:1:", result.output)
        self.assertIn("src/file_599.py:600:", result.output)
        self.assertIn("省略", result.output)
        self.assertIn("raw_output", result.meta)
        self.assertIn("context_output", result.meta)
        self.assertIn("PATTERN: session", str(result.meta["context_output"]))
        self.assertLess(len(str(result.meta["context_output"])), len(str(result.meta["raw_output"])))

    def test_tool_registry_smart_truncates_list_files_but_keeps_header_and_tail(self) -> None:
        def _validate(input_data: object) -> dict[str, str]:
            return {}

        def _run(_: dict[str, str], __: ToolContext) -> ToolResult:
            header = (
                "ROOT: .\n"
                "TOTAL_ENTRIES: 500\n"
                "RETURNED_ENTRIES: 500\n"
                "TRUNCATED: no\n\n"
            )
            body = "\n".join(
                f"file nested_directory_{index:03d}_{'x' * 40}.txt"
                for index in range(500)
            )
            return ToolResult(ok=True, output=header + body)

        registry = ToolRegistry(
            tools=[
                ToolDefinition(
                    name="list_files",
                    description="test list",
                    validator=_validate,
                    runner=_run,
                    input_schema={},
                )
            ]
        )

        result = registry.execute_tool("list_files", {}, ToolContext(cwd="."))

        self.assertTrue(result.ok)
        self.assertTrue(result.meta["truncated"])
        self.assertIn("ROOT: .", result.output)
        self.assertIn("TOTAL_ENTRIES: 500", result.output)
        self.assertIn("file nested_directory_000_", result.output)
        self.assertIn("file nested_directory_499_", result.output)
        self.assertIn("省略", result.output)
        self.assertIn("raw_output", result.meta)
        self.assertIn("context_output", result.meta)
        self.assertIn("TOTAL_ENTRIES: 500", str(result.meta["context_output"]))
        self.assertLess(len(str(result.meta["context_output"])), len(str(result.meta["raw_output"])))


if __name__ == "__main__":
    unittest.main()
