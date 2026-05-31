from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from app.types import ToolContext, ToolResult


Validator = Callable[[Any], Any]
Runner = Callable[[Any, ToolContext], ToolResult]

# 不同工具类型适合不同的上下文保留策略。
_TOOL_OUTPUT_LIMITS: dict[str, int] = {
    "read_file": 40_000,
    "grep_files": 16_000,
    "list_files": 12_000,
    "file_overview": 10_000,
    "find_references": 8_000,
    "run_command": 30_000,
}
_DEFAULT_MAX_OUTPUT = 18_000
_TOOL_CONTEXT_OUTPUT_LIMITS: dict[str, int] = {
    "read_file": 6_000,
    "grep_files": 4_000,
    "list_files": 3_500,
    "file_overview": 2_500,
    "find_references": 2_000,
    "run_command": 5_000,
}
_DEFAULT_CONTEXT_OUTPUT_LIMIT = 6_000


@dataclass(slots=True)
class ToolDefinition:
    """表示一个工具的定义信息。"""

    name: str
    description: str
    validator: Validator
    runner: Runner
    input_schema: dict[str, Any]


class ToolRegistry:
    """工具注册表：统一管理工具，并在返回前做工具级输出治理。"""

    def __init__(self, tools: list[ToolDefinition], max_output_lines: int = 120) -> None:
        self._tools = tools
        self._tool_index: dict[str, ToolDefinition] = {tool.name: tool for tool in tools}
        self._max_output_lines = max_output_lines

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools)

    def list_tool_name(self) -> list[str]:
        return list(self._tool_index.keys())

    def find_tool(self, name: str) -> ToolDefinition | None:
        return self._tool_index.get(name)

    def _normalize_output(self, text: Any) -> str:
        """把工具输出统一转成字符串，并兜底空输出。"""
        if text is None:
            return "工具执行完成，但没有输出。"

        normalized = str(text)
        if not normalized.strip():
            return "工具执行完成，但没有输出。"
        return normalized

    def _smart_truncate_output(self, output: str, tool_name: str) -> str:
        """按工具类型压缩超大输出，尽量保留对推理最有价值的部分。"""
        if not output:
            return output

        limit = _TOOL_OUTPUT_LIMITS.get(tool_name, _DEFAULT_MAX_OUTPUT)
        if len(output) <= limit:
            return output

        lines = output.splitlines()
        total_lines = max(1, len(lines))
        total_chars = len(output)
        average_line_length = max(1, total_chars / total_lines)
        max_lines = max(8, int(limit / max(40, average_line_length)))

        if tool_name == "read_file":
            # read_file 已经天然支持分段，因此二次截断时优先保留头信息和正文头尾。
            return self._truncate_head_tail(lines, total_chars, max_lines, head_ratio=0.6)

        if tool_name in {"grep_files"}:
            # grep 结果通常有结构化头部，二次截断时要保留统计信息和首尾样本。
            return self._truncate_structured_collection_output(
                lines,
                total_chars,
                max_lines,
            )

        if tool_name in {"find_references"}:
            # 引用结果本质上也是命中列表，优先保留首尾样本而不是整段平铺。
            return self._truncate_structured_collection_output(
                lines,
                total_chars,
                max_lines,
            )

        if tool_name in {"list_files"}:
            # 目录列表和 grep 一样属于集合输出，适合保留头部统计和首尾目录项。
            return self._truncate_structured_collection_output(
                lines,
                total_chars,
                max_lines,
            )

        if tool_name in {"run_command"}:
            # 命令输出经常要保留报错行，因此在头尾之外尽量补一些 error/warning 行。
            error_lines = self._extract_error_lines(lines, max_keep=10)
            base = self._truncate_head_tail(lines, total_chars, max_lines, head_ratio=0.4, tail_ratio=0.4)
            if error_lines:
                error_block = "\n".join(error_lines)
                return f"{base}\n\n--- Error Highlights ---\n{error_block}"
            return base

        return self._truncate_head_tail(lines, total_chars, max_lines)

    def _build_context_output(self, raw_output: str, tool_name: str) -> str:
        """生成更适合写入对话历史的紧凑版工具结果。"""
        if not raw_output:
            return raw_output

        limit = _TOOL_CONTEXT_OUTPUT_LIMITS.get(tool_name, _DEFAULT_CONTEXT_OUTPUT_LIMIT)
        if len(raw_output) <= limit:
            return raw_output

        lines = raw_output.splitlines()
        total_lines = max(1, len(lines))
        average_line_length = max(1, len(raw_output) / total_lines)
        max_lines = max(6, int(limit / max(40, average_line_length)))

        if tool_name == "read_file":
            return self._truncate_head_tail(lines, len(raw_output), max_lines, head_ratio=0.55, tail_ratio=0.25)
        if tool_name in {"grep_files", "list_files", "find_references"}:
            return self._truncate_structured_collection_output(lines, len(raw_output), max_lines)
        if tool_name == "file_overview":
            return self._truncate_head_tail(lines, len(raw_output), max_lines, head_ratio=0.7, tail_ratio=0.15)
        if tool_name == "run_command":
            error_lines = self._extract_error_lines(lines, max_keep=6)
            base = self._truncate_head_tail(lines, len(raw_output), max_lines, head_ratio=0.35, tail_ratio=0.35)
            if error_lines:
                return f"{base}\n\n--- Error Highlights ---\n" + "\n".join(error_lines)
            return base
        return self._truncate_head_tail(lines, len(raw_output), max_lines)

    def _truncate_structured_collection_output(
        self,
        lines: list[str],
        total_chars: int,
        max_lines: int,
    ) -> str:
        """为 grep/list 这类集合输出保留统计头和首尾样本。"""
        total_lines = len(lines)
        if total_lines <= max_lines:
            return "\n".join(lines)

        header_lines, body_lines = self._split_structured_header(lines)
        if not body_lines:
            return self._truncate_head_tail(lines, total_chars, max_lines)

        header_budget = len(header_lines)
        remaining_budget = max_lines - header_budget - 1
        if remaining_budget < 4:
            return self._truncate_head_tail(lines, total_chars, max_lines)

        head_count = max(2, int(remaining_budget * 0.6))
        tail_count = max(2, remaining_budget - head_count)
        while head_count + tail_count > remaining_budget and tail_count > 1:
            tail_count -= 1

        head_body = body_lines[:head_count]
        tail_body = body_lines[-tail_count:] if tail_count > 0 else []
        omitted = max(0, len(body_lines) - len(head_body) - len(tail_body))

        parts: list[str] = []
        parts.extend(header_lines)
        parts.extend(head_body)
        if omitted > 0:
            parts.append(f"... [中间省略 {omitted} 行（输出过大：{total_chars} 字符）] ...")
        if tail_body:
            parts.extend(tail_body)
        return "\n".join(parts)

    def _truncate_head_tail(
        self,
        lines: list[str],
        total_chars: int,
        max_lines: int,
        *,
        head_ratio: float = 0.5,
        tail_ratio: float | None = None,
    ) -> str:
        """通用头尾截断，用于保留输出开头和结尾的关键信息。"""
        total_lines = len(lines)
        if total_lines <= max_lines:
            return "\n".join(lines)

        effective_tail_ratio = tail_ratio if tail_ratio is not None else (1.0 - head_ratio)
        head_lines = max(1, int(max_lines * head_ratio))
        tail_lines = max(1, int(max_lines * effective_tail_ratio))

        # 避免 head + tail 超出目标行数太多。
        while head_lines + tail_lines > max_lines and tail_lines > 1:
            tail_lines -= 1

        head = "\n".join(lines[:head_lines])
        tail = "\n".join(lines[-tail_lines:])
        omitted = max(0, total_lines - head_lines - tail_lines)
        return (
            f"{head}\n"
            f"\n... [中间省略 {omitted} 行（输出过大：{total_chars} 字符）] ...\n\n"
            f"{tail}"
        )

    def _split_structured_header(self, lines: list[str]) -> tuple[list[str], list[str]]:
        """拆分带空行分隔的结构化头部，便于集合结果做首尾保留。"""
        if "" not in lines:
            return [], lines

        separator_index = lines.index("")
        header_lines = lines[:separator_index + 1]
        body_lines = lines[separator_index + 1:]
        return header_lines, body_lines

    def _extract_error_lines(self, lines: list[str], *, max_keep: int) -> list[str]:
        """从命令输出里挑出显眼的错误/告警行，帮助模型快速定位失败原因。"""
        error_pattern = re.compile(r"(?i)(error|fail|exception|traceback|warning)")
        error_lines: list[str] = []
        for line in lines:
            if error_pattern.search(line):
                error_lines.append(line)
            if len(error_lines) >= max_keep:
                break
        return error_lines

    def _normalize_result(self, tool_name: str, result: ToolResult) -> ToolResult:
        """统一返回结构，并在必要时做按工具类型的 smart truncate。"""
        output = self._normalize_output(result.output)
        raw_output = output
        summarized_output = self._smart_truncate_output(output, tool_name)
        context_output = self._build_context_output(raw_output, tool_name)
        truncated = summarized_output != raw_output
        total_lines = len(raw_output.splitlines())

        error = result.error
        if not result.ok and not error:
            error = f"工具 {tool_name} 执行失败"

        meta = dict(result.meta)
        meta.setdefault("tool_name", tool_name)
        meta["truncated"] = truncated
        meta["total_lines"] = total_lines
        meta["max_output_lines"] = self._max_output_lines
        meta["output_limit_chars"] = _TOOL_OUTPUT_LIMITS.get(tool_name, _DEFAULT_MAX_OUTPUT)
        meta["raw_output_chars"] = len(raw_output)
        meta["context_output"] = context_output
        meta["context_output_chars"] = len(context_output)
        if truncated:
            # 保留完整原文，供后续 context compactor 落盘或调试使用。
            meta["raw_output"] = raw_output

        return ToolResult(
            ok=result.ok,
            output=summarized_output,
            error=error,
            meta=meta,
        )

    def execute_tool(self, tool_name: str, input_data: Any, context: ToolContext) -> ToolResult:
        """
        统一执行工具。

        流程：
        1. 找到工具
        2. 校验输入
        3. 执行工具
        4. 统一做输出治理
        """
        tool = self.find_tool(tool_name)
        if not tool:
            return self._normalize_result(
                tool_name,
                ToolResult(
                    ok=False,
                    output=f"不存在名称为 {tool_name} 的工具",
                    error="TOOL_NOT_FOUND",
                    meta={"tool_name": tool_name},
                ),
            )

        try:
            parsed_input = tool.validator(input_data)
        except (ValueError, TypeError, KeyError) as error:
            return self._normalize_result(
                tool_name,
                ToolResult(
                    ok=False,
                    output=f"工具参数错误：{error}",
                    error="INVALID_INPUT",
                    meta={"tool_name": tool_name},
                ),
            )

        try:
            raw_result = tool.runner(parsed_input, context)
        except Exception as error:
            return self._normalize_result(
                tool_name,
                ToolResult(
                    ok=False,
                    output=f"工具运行异常：{error}",
                    error="TOOL_RUNTIME_ERROR",
                    meta={"tool_name": tool_name},
                ),
            )

        if raw_result is None:
            raw_result = ToolResult(
                ok=False,
                output="工具未返回结果。",
                error="EMPTY_RESULT",
                meta={"tool_name": tool_name},
            )

        if not isinstance(raw_result, ToolResult):
            raw_result = ToolResult(
                ok=False,
                output=f"工具 {tool_name} 返回了非法结果类型: {type(raw_result).__name__}",
                error="INVALID_TOOL_RESULT",
                meta={"tool_name": tool_name},
            )

        return self._normalize_result(tool_name, raw_result)
