from __future__ import annotations

import re
import time

from app.context_reactive_compact import (
    is_context_overflow_error,
    recover_from_context_overflow,
)
from app.context_runtime import prepare_agent_context
from app.history_summarizer import OlderHistorySummarizer
from app.logger import log_event
from app.memory_pipeline import MemoryPipeline
from app.message_builder import MessageBuilder
from app.session import SessionData
from app.tooling import ToolRegistry
from app.types import AgentStep, ApprovalRequest, ChatMessage, ModelAdapter, ToolContext, ToolResult
from app.working_memory import WorkingMemory
from app.working_memory_updater import (
    extract_active_paths,
    extract_decision_from_assistant,
    extract_decisions_from_assistant,
    summarize_failure,
)

NUDGE_CONTINUE = (
    "继续推进：如果现有信息已经足够，请直接给出最终答案；"
    "否则只执行下一步最必要的动作，不要重复读取相同文件、符号或目录。"
)
NUDGE_AFTER_TOOL_RESULT = (
    "你已经拿到了工具结果。请先判断现有证据是否已经足够："
    "足够就直接给最终答案；不够才继续一次最必要的工具调用。不要重复刚看到的内容。"
)
NUDGE_ANALYSIS_CONVERGE = (
    "现有证据已经足够回答这次代码分析问题。"
    "禁止继续重复读取相同文件、相同符号或相同目录；"
    "请直接基于已有证据给出最终答案，并明确写出仍然不确定的点。"
)
NUDGE_ANALYSIS_TOOL_PRIORITY = (
    "这是代码链路分析任务。优先使用 get_ast_info 或 find_symbols 这类结构化工具确认真实函数表，"
    "再按需使用 read_file 看局部分块；不要一上来只靠连续分块阅读。"
)
NUDGE_ANALYSIS_STRUCTURE_FIRST = (
    "当前仍处于代码分析取证阶段。请先使用 get_ast_info、find_symbols、locate_symbol 或 file_overview "
    "确认真实顶层符号和文件结构，再决定是否需要 read_file 看局部源码。"
)
EXPLORATION_TOOL_NAMES = {
    "read_file",
    "file_overview",
    "find_symbols",
    "find_references",
    "locate_symbol",
    "get_ast_info",
    "list_files",
    "grep_files",
}
_ANALYSIS_KEYWORDS = (
    "分析",
    "链路",
    "调用链",
    "串联",
    "入口",
    "流程",
    "结构",
    "梳理",
    "trace",
    "analyze",
    "analysis",
    "call chain",
    "workflow",
    "walkthrough",
)
_PATH_PATTERN = re.compile(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FUNCTION_CALL_PATTERN = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_DOTTED_FUNCTION_CALL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\("
)
_STEP_TYPE_PATTERN = re.compile(r"step\.type\s*==\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']")
_LINE_COUNT_PATTERN = re.compile(r"(?:总行数|Lines)[:：]?\s*(\d+)")
_ANSWER_LINE_COUNT_PATTERN = re.compile(r"(?:文件共|文件总共|总行数[:：]?|共)\s*(\d+)\s*行")
_ANSWER_FUNCTION_COUNT_PATTERN = re.compile(r"(?:包含|共)\s*(\d+)\s*个函数")
_READ_FILE_HEADER_PATTERN = re.compile(r"^(OFFSET|END|TOTAL_CHARS|TRUNCATED):\s*(.+)$", re.MULTILINE)
_UNREAD_CLAIM_PATTERN = re.compile(r"((?:未读取到|没有读取到|未读到)[^。；\n]{0,80})")
_DIRECTORY_QUALIFIED_FILE_PATTERN = re.compile(
    r"([A-Za-z0-9_./\\-]+)\s*目录下的\s*([A-Za-z0-9_.-]+\.[A-Za-z0-9_]+)"
)
_CLI_ARG_PATTERN = re.compile(r"add_argument\(\s*[\"'](--[A-Za-z0-9_-]+)[\"']")
_ANSWER_CLI_ARG_PATTERN = re.compile(r"(--[A-Za-z0-9_-]+)")
_COMMON_FILE_EXTENSIONS = {"py", "js", "ts", "tsx", "jsx", "json", "yaml", "yml", "md", "txt"}
_ANALYSIS_STRUCTURE_FIRST_TOOLS = {"file_overview", "get_ast_info", "find_symbols", "locate_symbol"}


def _append_transient_user_nudge(
    messages: list[ChatMessage],
    content: str | None,
) -> list[ChatMessage]:
    """只在当前模型请求里追加临时引导语，不写入会话历史。"""
    if not content:
        return messages

    if messages:
        last_message = messages[-1]
        if last_message.get("role") == "user" and str(last_message.get("content", "")) == content:
            return messages

    result = list(messages)
    result.append(
        {
            "role": "user",
            "content": content,
        }
    )
    return result


def _extract_tool_target(tool_input: object) -> str:
    """尽量把工具输入归一成“当前在看什么”。"""
    if not isinstance(tool_input, dict):
        return ""

    for key in ("path", "file_path", "symbol_path", "directory", "root", "cwd"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("symbol", "name", "keyword", "pattern", "query"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _is_exploration_tool(tool_name: str) -> bool:
    return tool_name in EXPLORATION_TOOL_NAMES


def _normalize_target(value: str) -> str:
    """统一路径/符号文本，方便后续做重复探索判断。"""
    return value.replace("\\", "/").strip()


def _basename_for_path(path: str) -> str:
    """从规范化路径里取出文件名，便于把“main.py”解析到真实路径。"""
    normalized = _normalize_target(path)
    return normalized.rsplit("/", 1)[-1]


def _extract_latest_real_user_message(messages: list[ChatMessage]) -> str:
    """拿到当前任务的真实用户请求，跳过系统注入的临时提示。"""
    synthetic_messages = {
        NUDGE_CONTINUE,
        NUDGE_AFTER_TOOL_RESULT,
        NUDGE_ANALYSIS_CONVERGE,
        NUDGE_ANALYSIS_TOOL_PRIORITY,
        NUDGE_ANALYSIS_STRUCTURE_FIRST,
    }
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if not content or content in synthetic_messages:
            continue
        return content
    return ""


def _extract_requested_analysis_targets(text: str) -> tuple[set[str], set[str]]:
    """从用户问题里提取显式路径和裸文件名，减少“main.py”类问法丢目标。"""
    explicit_paths: set[str] = set()
    bare_filenames: set[str] = set()

    for directory_part, filename in _DIRECTORY_QUALIFIED_FILE_PATTERN.findall(text):
        directory = _normalize_target(directory_part).strip("/")
        file_name = _normalize_target(filename)
        if directory and file_name:
            explicit_paths.add(f"{directory}/{file_name}")

    for match in _PATH_PATTERN.findall(text):
        normalized = _normalize_target(match)
        if "/" in normalized or "\\" in match:
            explicit_paths.add(normalized)
        else:
            bare_filenames.add(normalized)

    return explicit_paths, bare_filenames


def _extract_candidate_paths(text: str) -> set[str]:
    """兼容旧调用，返回已解析出的显式路径。"""
    explicit_paths, _bare_filenames = _extract_requested_analysis_targets(text)
    return explicit_paths


def _classify_analysis_kind(text: str) -> str:
    """区分链路分析和普通文件总览，避免收敛条件一刀切。"""
    lowered_text = text.lower()
    call_chain_keywords = (
        "链路",
        "调用链",
        "串联",
        "入口",
        "流程",
        "trace",
        "call chain",
        "workflow",
        "walkthrough",
    )
    symbol_keywords = ("函数表", "函数列表", "symbol", "symbols", "ast", "接口")
    if any(keyword in text for keyword in call_chain_keywords) or any(
        keyword in lowered_text for keyword in call_chain_keywords
    ):
        return "call_chain"
    if any(keyword in text for keyword in symbol_keywords) or any(
        keyword in lowered_text for keyword in symbol_keywords
    ):
        return "symbol_inventory"
    return "file_summary"


def _is_code_analysis_request(text: str) -> bool:
    """粗粒度识别“理解代码/串联链路”这类任务。"""
    if not text:
        return False
    lowered_text = text.lower()
    # 这里同时兼容中英文提问。
    # 否则像 "analyze app/main.py call chain" 这类请求不会进入 analysis_tracker，
    # 后面的真实符号/事实校验链路就完全失效。
    if not any(keyword in text for keyword in _ANALYSIS_KEYWORDS):
        if not any(keyword in lowered_text for keyword in _ANALYSIS_KEYWORDS):
            return False
    code_signals = (".py", "目录", "项目", "文件", "函数", "模块", "app/", "类")
    english_code_signals = ("file", "files", "function", "functions", "module", "class")
    return any(signal in text for signal in code_signals) or any(
        signal in lowered_text for signal in english_code_signals
    )


def _create_analysis_tracker(task_text: str) -> dict[str, object]:
    """维护代码分析任务的结构化证据，而不是只盯着自由文本历史。"""
    candidate_paths, requested_basenames = _extract_requested_analysis_targets(task_text)
    return {
        "task_text": task_text,
        "analysis_kind": _classify_analysis_kind(task_text),
        "candidate_paths": candidate_paths,
        "requested_basenames": requested_basenames,
        "covered_paths": set(),
        "overview_paths": set(),
        "ast_paths": set(),
        "read_paths": set(),
        "reference_symbols": set(),
        "observed_symbols": set(),
        "observed_functions": set(),
        "observed_step_types": set(),
        "observed_file_line_counts": {},
        "observed_file_function_counts": {},
        "observed_file_cli_args": {},
        "fully_read_paths": set(),
        "truncated_read_paths": set(),
        "structured_hits": 0,
        "read_counts": {},
        "read_segments": {},
        "matched_target_paths": set(),
        "invalid_grep_file_paths": set(),
    }


def _bump_counter(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _normalize_symbol_name(raw_name: str) -> str:
    """把工具输出里的候选符号名裁成稳定的 Python 标识符。"""
    candidate = raw_name.strip()
    if not candidate:
        return ""
    candidate = candidate.split("@L", 1)[0].strip()
    candidate = candidate.split("(", 1)[0].strip()
    candidate = candidate.split(":", 1)[0].strip()
    candidate = candidate.rsplit(".", 1)[-1].strip()
    return candidate if _IDENTIFIER_PATTERN.fullmatch(candidate) else ""


def _extract_symbols_from_tool_result(tool_name: str, raw_output: str) -> tuple[set[str], set[str]]:
    """从结构化工具输出里提取真实符号名和顶层函数名。"""
    observed_symbols: set[str] = set()
    observed_functions: set[str] = set()

    if not raw_output.strip():
        return observed_symbols, observed_functions

    current_section = ""
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":"):
            current_section = line[:-1]
            continue

        if tool_name == "file_overview":
            if current_section == "导入":
                symbol_name = _normalize_symbol_name(line)
                if symbol_name:
                    observed_symbols.add(symbol_name)
                continue
            if current_section == "函数":
                match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
                if match:
                    function_name = match.group(1)
                    observed_symbols.add(function_name)
                    observed_functions.add(function_name)
                continue
            if current_section == "类":
                symbol_name = _normalize_symbol_name(line.split(" ", 1)[0])
                if symbol_name:
                    observed_symbols.add(symbol_name)
                continue

        if tool_name == "get_ast_info":
            if current_section == "导入":
                symbol_name = _normalize_symbol_name(line)
                if symbol_name:
                    observed_symbols.add(symbol_name)
                continue
            if line.startswith("function "):
                function_name = _normalize_symbol_name(line[len("function "):])
                if function_name:
                    observed_symbols.add(function_name)
                    observed_functions.add(function_name)
                continue
            if line.startswith("class "):
                symbol_name = _normalize_symbol_name(line[len("class "):])
                if symbol_name:
                    observed_symbols.add(symbol_name)
                continue
            if line.startswith("variable "):
                symbol_name = _normalize_symbol_name(line[len("variable "):])
                if symbol_name:
                    observed_symbols.add(symbol_name)
                continue
            if line.startswith("- method ") or line.startswith("  - method "):
                symbol_name = _normalize_symbol_name(line.split("method ", 1)[1])
                if symbol_name:
                    observed_symbols.add(symbol_name)
                continue

        if tool_name == "find_symbols":
            match = re.match(r"^[^:]+:\d+\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if match:
                symbol_kind = match.group(1)
                symbol_name = match.group(2)
                observed_symbols.add(symbol_name)
                if symbol_kind == "function":
                    observed_functions.add(symbol_name)
            continue

        if tool_name == "locate_symbol":
            if line.startswith("符号:"):
                symbol_name = _normalize_symbol_name(line.split(":", 1)[1])
                if symbol_name:
                    observed_symbols.add(symbol_name)
                continue
            match = re.match(
                r"^(?:[^:]+:\d+:\s+)?(class|function|method|variable)\s+([A-Za-z_][A-Za-z0-9_]*)",
                line,
            )
            if match:
                symbol_kind = match.group(1)
                symbol_name = match.group(2)
                observed_symbols.add(symbol_name)
                if symbol_kind == "function":
                    observed_functions.add(symbol_name)
            continue

    return observed_symbols, observed_functions


def _extract_analysis_facts_from_tool_result(
    tool_name: str,
    raw_output: str,
) -> tuple[set[str], int | None]:
    """从工具结果里提取 step.type 等可直接校验的事实。"""
    observed_step_types = set(_STEP_TYPE_PATTERN.findall(raw_output))
    line_count: int | None = None

    if tool_name in {"file_overview", "get_ast_info"}:
        match = _LINE_COUNT_PATTERN.search(raw_output)
        if match:
            line_count = int(match.group(1))

    return observed_step_types, line_count


def _extract_read_file_coverage(
    result: ToolResult,
    raw_output: str,
) -> tuple[int | None, int | None, int | None, bool | None]:
    """尽量从 read_file 结果里恢复 offset/end/total/truncated，用于判断是否完整读过文件。"""
    offset = result.meta.get("offset")
    end = result.meta.get("end")
    total_chars = result.meta.get("total_chars")
    truncated = result.meta.get("truncated")

    parsed_headers: dict[str, str] = {}
    for key, value in _READ_FILE_HEADER_PATTERN.findall(raw_output):
        parsed_headers[key] = value.strip()

    if not isinstance(offset, int):
        raw_offset = parsed_headers.get("OFFSET")
        if raw_offset and raw_offset.isdigit():
            offset = int(raw_offset)
        else:
            offset = None

    if not isinstance(end, int):
        raw_end = parsed_headers.get("END")
        if raw_end and raw_end.isdigit():
            end = int(raw_end)
        else:
            end = None

    if not isinstance(total_chars, int):
        raw_total = parsed_headers.get("TOTAL_CHARS")
        if raw_total and raw_total.isdigit():
            total_chars = int(raw_total)
        else:
            total_chars = None

    if not isinstance(truncated, bool):
        raw_truncated = parsed_headers.get("TRUNCATED", "").lower()
        if raw_truncated.startswith("yes"):
            truncated = True
        elif raw_truncated.startswith("no"):
            truncated = False
        else:
            truncated = None

    return offset, end, total_chars, truncated


def _extract_call_names_from_source_text(content: str) -> set[str]:
    """从 read_file 的源码片段里提取真实出现过的调用名。"""
    call_names: set[str] = set()

    for match in _DOTTED_FUNCTION_CALL_PATTERN.findall(content):
        last_part = match.rsplit(".", 1)[-1].strip()
        if _IDENTIFIER_PATTERN.fullmatch(last_part):
            call_names.add(last_part)

    for match in _FUNCTION_CALL_PATTERN.findall(content):
        if _IDENTIFIER_PATTERN.fullmatch(match):
            call_names.add(match)

    return call_names


def _extract_cli_args_from_source_text(content: str) -> set[str]:
    """从源码片段里提取 argparse 的真实参数名。"""
    return {match for match in _CLI_ARG_PATTERN.findall(content)}


def _matches_requested_analysis_target(tracker: dict[str, object], path: str) -> bool:
    """判断当前观察到的路径是否命中了用户真正询问的目标。"""
    candidate_paths = tracker["candidate_paths"]
    requested_basenames = tracker["requested_basenames"]
    if path in candidate_paths:
        return True
    return _basename_for_path(path) in requested_basenames


def _record_matched_analysis_target(tracker: dict[str, object], path: str) -> None:
    """把命中的真实路径沉淀下来，后面用于收敛和答案校验。"""
    if _matches_requested_analysis_target(tracker, path):
        tracker["matched_target_paths"].add(path)


def _build_analysis_read_signature(path: str, offset: int, limit: int) -> str:
    """为分析态的 read_file 去重生成稳定签名。"""
    return f"{path}::{offset}::{limit}"


def _should_redirect_analysis_to_structure_first(
    tracker: dict[str, object],
    calls: list[dict[str, object]],
) -> bool:
    """链路分析早期优先拉结构化证据，避免一开始就靠 read_file 猜流程。"""
    if str(tracker.get("analysis_kind", "")) != "call_chain":
        return False
    if int(tracker.get("structured_hits", 0)) > 0:
        return False
    if not calls:
        return False

    call_tool_names = {
        str(call.get("tool_name", "")).strip()
        for call in calls
        if isinstance(call, dict)
    }
    if not call_tool_names:
        return False
    if call_tool_names & _ANALYSIS_STRUCTURE_FIRST_TOOLS:
        return False

    # 这里不拦所有探索工具，只拦明显“还没看结构就直接切源码”的情况。
    return "read_file" in call_tool_names


def _extract_answer_function_names(content: str) -> set[str]:
    """从最终回答里提取被当作函数调用的标识符。"""
    names: set[str] = set()

    for match in _DOTTED_FUNCTION_CALL_PATTERN.findall(content):
        last_part = match.rsplit(".", 1)[-1].strip()
        if last_part in _COMMON_FILE_EXTENSIONS:
            continue
        if _IDENTIFIER_PATTERN.fullmatch(last_part):
            names.add(last_part)

    for match in _FUNCTION_CALL_PATTERN.findall(content):
        if _IDENTIFIER_PATTERN.fullmatch(match):
            names.add(match)

    return names


def _find_unobserved_answer_function_names(
    tracker: dict[str, object],
    content: str,
) -> list[str]:
    """找出回答里提到、但当前证据里从未观察到的函数名。"""
    observed_symbols = {
        symbol
        for symbol in tracker["observed_symbols"]
        if isinstance(symbol, str) and symbol
    }
    if not observed_symbols:
        return []

    mentioned_names = _extract_answer_function_names(content)
    if not mentioned_names:
        return []

    return sorted(name for name in mentioned_names if name not in observed_symbols)


def _extract_answer_step_types(content: str) -> set[str]:
    """从最终回答里提取 step.type 断言。"""
    return set(_STEP_TYPE_PATTERN.findall(content))


def _extract_answer_line_counts(content: str) -> set[int]:
    """从最终回答里提取文件总行数断言。"""
    return {int(match) for match in _ANSWER_LINE_COUNT_PATTERN.findall(content)}


def _extract_answer_function_counts(content: str) -> set[int]:
    """从最终回答里提取顶层函数数量断言。"""
    return {int(match) for match in _ANSWER_FUNCTION_COUNT_PATTERN.findall(content)}


def _select_primary_analysis_path(tracker: dict[str, object]) -> str:
    """优先选用户显式提到的目标文件，便于校验文件级事实。"""
    candidate_paths = sorted(
        path for path in tracker["candidate_paths"] if isinstance(path, str) and path
    )
    if candidate_paths:
        return candidate_paths[0]

    matched_target_paths = sorted(
        path for path in tracker["matched_target_paths"] if isinstance(path, str) and path
    )
    if len(matched_target_paths) == 1:
        return matched_target_paths[0]

    covered_paths = sorted(
        path for path in tracker["covered_paths"] if isinstance(path, str) and path
    )
    return covered_paths[0] if covered_paths else ""


def _find_unsupported_analysis_claims(
    tracker: dict[str, object],
    content: str,
) -> list[str]:
    """找出回答里缺少证据支撑的 step.type、参数名和统计数字断言。"""
    invalid_claims: list[str] = []

    observed_step_types = {
        step_type
        for step_type in tracker["observed_step_types"]
        if isinstance(step_type, str) and step_type
    }
    for step_type in sorted(_extract_answer_step_types(content)):
        if observed_step_types and step_type not in observed_step_types:
            invalid_claims.append(f'step.type == "{step_type}"')

    primary_path = _select_primary_analysis_path(tracker)
    if primary_path:
        fully_read_paths = tracker["fully_read_paths"]
        if primary_path in fully_read_paths:
            seen_unread_claims: set[str] = set()
            for match in _UNREAD_CLAIM_PATTERN.findall(content):
                claim = match.strip()
                if claim and claim not in seen_unread_claims:
                    invalid_claims.append(f"{primary_path} unread_claim={claim}")
                    seen_unread_claims.add(claim)

        observed_file_line_counts = tracker["observed_file_line_counts"]
        expected_line_count = observed_file_line_counts.get(primary_path)
        if isinstance(expected_line_count, int):
            for line_count in sorted(_extract_answer_line_counts(content)):
                if line_count != expected_line_count:
                    invalid_claims.append(
                        f"{primary_path} line_count={line_count} (expected {expected_line_count})"
                    )

        observed_file_function_counts = tracker["observed_file_function_counts"]
        expected_function_count = observed_file_function_counts.get(primary_path)
        if isinstance(expected_function_count, int):
            for function_count in sorted(_extract_answer_function_counts(content)):
                if function_count != expected_function_count:
                    invalid_claims.append(
                        f"{primary_path} function_count={function_count} (expected {expected_function_count})"
                    )

        observed_file_cli_args = tracker["observed_file_cli_args"]
        expected_cli_args = observed_file_cli_args.get(primary_path)
        if isinstance(expected_cli_args, set) and expected_cli_args:
            for cli_arg in sorted(set(_ANSWER_CLI_ARG_PATTERN.findall(content))):
                if cli_arg not in expected_cli_args:
                    invalid_claims.append(
                        f"{primary_path} cli_arg={cli_arg} (expected one of {sorted(expected_cli_args)})"
                    )

    return invalid_claims


def _build_analysis_symbol_correction_nudge(
    tracker: dict[str, object],
    invalid_names: list[str],
) -> str:
    """当最终回答出现未观察到的名字时，要求模型基于真符号自纠。"""
    observed_functions = sorted(
        symbol
        for symbol in tracker["observed_functions"]
        if isinstance(symbol, str) and symbol
    )
    confirmed_functions = "、".join(observed_functions[:8]) if observed_functions else "(暂未确认)"
    invalid_display = "、".join(invalid_names[:6])
    return (
        f"你刚才引用了未观察到的标识符: {invalid_display}。\n"
        f"已确认函数名: {confirmed_functions}。\n"
        "禁止引用未观察到的标识符；请仅基于已确认的真实函数名重写答案，"
        "如果仍不确定，就明确写“未确认”。"
    )


def _build_analysis_fact_correction_nudge(
    tracker: dict[str, object],
    invalid_names: list[str],
    invalid_claims: list[str],
) -> str:
    """当最终回答里的事实断言缺少证据时，要求模型基于已观察事实重写。"""
    lines: list[str] = []

    if invalid_names:
        lines.append(f"未观察到的标识符: {'、'.join(invalid_names[:6])}")
    if invalid_claims:
        lines.append(f"缺少证据的事实断言: {'；'.join(invalid_claims[:4])}")

    observed_functions = sorted(
        symbol
        for symbol in tracker["observed_functions"]
        if isinstance(symbol, str) and symbol
    )
    if observed_functions:
        lines.append(f"已确认函数名: {'、'.join(observed_functions[:8])}")

    observed_step_types = sorted(
        step_type
        for step_type in tracker["observed_step_types"]
        if isinstance(step_type, str) and step_type
    )
    if observed_step_types:
        lines.append(f"已确认 step.type: {'、'.join(observed_step_types)}")

    primary_path = _select_primary_analysis_path(tracker)
    if primary_path:
        line_count = tracker["observed_file_line_counts"].get(primary_path)
        if isinstance(line_count, int):
            lines.append(f"已确认 {primary_path} 总行数: {line_count}")
        function_count = tracker["observed_file_function_counts"].get(primary_path)
        if isinstance(function_count, int):
            lines.append(f"已确认 {primary_path} 顶层函数数: {function_count}")
        if primary_path in tracker["fully_read_paths"]:
            lines.append(f"已完整读取 {primary_path}；禁止再声称未读取到该文件的某段代码。")

    lines.append(
        "禁止引用未观察到的标识符；禁止补充未观察到的控制流、枚举值或统计数字；"
        "不确定就明确写“未确认”。"
    )
    return "\n".join(lines)


def _build_analysis_target_resolution_nudge(tracker: dict[str, object]) -> str:
    """用户只给文件名时，先提醒模型定位真实路径，避免分析错文件。"""
    candidate_paths = sorted(
        path for path in tracker["candidate_paths"] if isinstance(path, str) and path
    )
    requested_basenames = sorted(
        name for name in tracker["requested_basenames"] if isinstance(name, str) and name
    )
    if candidate_paths or not requested_basenames:
        return ""
    return (
        "用户目前只给了文件名，没有给完整路径。\n"
        f"待解析文件名: {'、'.join(requested_basenames[:3])}。\n"
        "请先定位真实路径；如果存在多个同名文件，就明确说明歧义，禁止自行假定目标文件。"
    )


def _normalize_analysis_answer_content(tracker: dict[str, object], content: str) -> str:
    """链路解释里提到私有 helper 时，按符号名展示即可，避免误导成外部入口函数。"""
    normalized = content
    for function_name in tracker["observed_functions"]:
        if not isinstance(function_name, str) or not function_name.startswith("_"):
            continue
        normalized = re.sub(
            rf"\b{re.escape(function_name)}\(\)",
            function_name,
            normalized,
        )
    return normalized


def _record_analysis_evidence(
    tracker: dict[str, object],
    *,
    tool_name: str,
    tool_input: object,
    result: ToolResult,
) -> None:
    """把高价值工具结果沉淀成结构化证据，后面据此决定是否该收敛。"""
    if not isinstance(tool_input, dict):
        return

    raw_target = tool_input.get("path")
    target = _normalize_target(raw_target) if isinstance(raw_target, str) and raw_target.strip() else ""
    covered_paths = tracker["covered_paths"]
    overview_paths = tracker["overview_paths"]
    ast_paths = tracker["ast_paths"]
    read_paths = tracker["read_paths"]
    reference_symbols = tracker["reference_symbols"]
    observed_symbols = tracker["observed_symbols"]
    observed_functions = tracker["observed_functions"]
    observed_step_types = tracker["observed_step_types"]
    observed_file_line_counts = tracker["observed_file_line_counts"]
    observed_file_function_counts = tracker["observed_file_function_counts"]
    observed_file_cli_args = tracker["observed_file_cli_args"]
    fully_read_paths = tracker["fully_read_paths"]
    truncated_read_paths = tracker["truncated_read_paths"]
    invalid_grep_file_paths = tracker["invalid_grep_file_paths"]
    read_counts = tracker["read_counts"]
    read_segments = tracker["read_segments"]
    raw_output = result.meta.get("raw_output", result.output)
    if not isinstance(raw_output, str):
        raw_output = result.output

    extracted_symbols, extracted_functions = _extract_symbols_from_tool_result(tool_name, raw_output)
    observed_symbols.update(extracted_symbols)
    observed_functions.update(extracted_functions)
    if target:
        _record_matched_analysis_target(tracker, target)
    if tool_name == "read_file" and result.ok:
        observed_symbols.update(_extract_call_names_from_source_text(raw_output))
        if target:
            cli_args = _extract_cli_args_from_source_text(raw_output)
            if cli_args:
                existing_cli_args = observed_file_cli_args.setdefault(target, set())
                if isinstance(existing_cli_args, set):
                    existing_cli_args.update(cli_args)
    extracted_step_types, line_count = _extract_analysis_facts_from_tool_result(tool_name, raw_output)
    observed_step_types.update(extracted_step_types)
    if target and line_count is not None:
        observed_file_line_counts[target] = line_count
    if target and tool_name in {"file_overview", "get_ast_info"} and extracted_functions:
        observed_file_function_counts[target] = len(extracted_functions)

    if tool_name == "file_overview" and result.ok and target:
        covered_paths.add(target)
        overview_paths.add(target)
        tracker["structured_hits"] = int(tracker["structured_hits"]) + 1
        return

    if tool_name == "read_file" and result.ok and target:
        covered_paths.add(target)
        read_paths.add(target)
        _bump_counter(read_counts, target)
        offset, _end, _total_chars, truncated = _extract_read_file_coverage(result, raw_output)
        raw_offset = tool_input.get("offset", 0)
        raw_limit = tool_input.get("limit", 0)
        try:
            signature_offset = int(raw_offset)
            signature_limit = int(raw_limit)
        except (TypeError, ValueError):
            signature_offset = None
            signature_limit = None
        if (
            signature_offset is not None
            and signature_limit is not None
            and signature_limit > 0
        ):
            signatures = read_segments.setdefault(target, set())
            if isinstance(signatures, set):
                signatures.add(_build_analysis_read_signature(target, signature_offset, signature_limit))
        if offset == 0 and truncated is False:
            fully_read_paths.add(target)
            truncated_read_paths.discard(target)
        elif truncated is True:
            truncated_read_paths.add(target)
        return

    if tool_name == "find_references" and result.ok:
        symbol = tool_input.get("symbol")
        if isinstance(symbol, str) and symbol.strip():
            reference_symbols.add(symbol.strip())
            tracker["structured_hits"] = int(tracker["structured_hits"]) + 1
        return

    if tool_name in {"find_symbols", "locate_symbol", "get_ast_info"} and result.ok:
        if target:
            covered_paths.add(target)
        if tool_name == "get_ast_info" and target:
            ast_paths.add(target)
        tracker["structured_hits"] = int(tracker["structured_hits"]) + 1
        return

    if (
        tool_name == "grep_files"
        and not result.ok
        and "目标不是目录" in result.output
        and target
    ):
        invalid_grep_file_paths.add(target)


def _has_sufficient_analysis_evidence(tracker: dict[str, object]) -> bool:
    """判断当前代码分析任务是否已经拿到了足够的结构化证据。"""
    analysis_kind = str(tracker.get("analysis_kind", "file_summary"))
    candidate_paths = tracker["candidate_paths"]
    requested_basenames = tracker["requested_basenames"]
    covered_paths = tracker["covered_paths"]
    overview_paths = tracker["overview_paths"]
    ast_paths = tracker["ast_paths"]
    read_paths = tracker["read_paths"]
    observed_functions = tracker["observed_functions"]
    fully_read_paths = tracker["fully_read_paths"]
    truncated_read_paths = tracker["truncated_read_paths"]
    matched_target_paths = tracker["matched_target_paths"]

    if not observed_functions:
        return False

    primary_path = _select_primary_analysis_path(tracker)

    # 用户只给了裸文件名时，至少要先把它解析成唯一真实路径，再谈“证据已足够”。
    if requested_basenames and not candidate_paths and len(matched_target_paths) != 1:
        return False

    if primary_path:
        if primary_path in truncated_read_paths and primary_path not in fully_read_paths:
            return False
        if primary_path not in covered_paths:
            return False

        if analysis_kind == "call_chain":
            # 链路分析必须至少拿到一份强结构化证据，不能只看自由文本概览。
            if primary_path not in overview_paths and primary_path not in ast_paths:
                return False
            # read_file 能补齐真实调用片段；如果已经拿到 AST 级结构，也允许直接收敛。
            if primary_path not in read_paths and primary_path not in ast_paths:
                return False
        return True

    if candidate_paths:
        return any(path in covered_paths for path in candidate_paths)

    return bool(covered_paths) and analysis_kind != "call_chain"


def _all_calls_are_exploration(calls: list[dict[str, object]]) -> bool:
    return bool(calls) and all(_is_exploration_tool(str(call.get("tool_name", ""))) for call in calls)


def _should_block_redundant_analysis_calls(
    tracker: dict[str, object],
    *,
    calls: list[dict[str, object]],
    step_index: int,
    max_steps: int,
) -> bool:
    """证据够了之后，拦掉明显重复的探索工具调用，逼模型进入答案收敛。"""
    if not _has_sufficient_analysis_evidence(tracker):
        return False
    if not _all_calls_are_exploration(calls):
        return False

    remaining_steps = max_steps - step_index - 1
    overview_paths = tracker["overview_paths"]
    read_segments = tracker["read_segments"]
    reference_symbols = tracker["reference_symbols"]
    invalid_grep_file_paths = tracker["invalid_grep_file_paths"]
    fully_read_paths = tracker["fully_read_paths"]

    redundant_calls = 0
    for call in calls:
        tool_name = str(call.get("tool_name", ""))
        tool_input = call.get("input")
        if not isinstance(tool_input, dict):
            continue

        raw_target = tool_input.get("path")
        target = _normalize_target(raw_target) if isinstance(raw_target, str) and raw_target.strip() else ""

        if tool_name == "file_overview" and target and target in overview_paths:
            redundant_calls += 1
            continue

        if tool_name == "read_file" and target:
            if target in fully_read_paths:
                redundant_calls += 1
                continue
            raw_offset = tool_input.get("offset", 0)
            raw_limit = tool_input.get("limit", 0)
            try:
                signature_offset = int(raw_offset)
                signature_limit = int(raw_limit)
            except (TypeError, ValueError):
                signature_offset = None
                signature_limit = None
            if (
                signature_offset is not None
                and signature_limit is not None
                and signature_limit > 0
            ):
                signatures = read_segments.get(target, set())
                signature = _build_analysis_read_signature(target, signature_offset, signature_limit)
                if isinstance(signatures, set) and signature in signatures:
                    redundant_calls += 1
                    continue

        if tool_name == "find_references":
            symbol = tool_input.get("symbol")
            if isinstance(symbol, str) and symbol.strip() in reference_symbols:
                redundant_calls += 1
                continue

        if tool_name == "grep_files" and target and target in invalid_grep_file_paths:
            redundant_calls += 1

    if remaining_steps <= 1:
        return True

    return redundant_calls == len(calls) and redundant_calls > 0


def _build_analysis_convergence_nudge(tracker: dict[str, object]) -> str:
    """把已掌握的证据压成极短提示，提醒模型直接作答。"""
    evidence_lines: list[str] = []
    analysis_kind = str(tracker.get("analysis_kind", "file_summary"))

    if analysis_kind == "call_chain":
        evidence_lines.append("当前任务类型: 链路分析")

    candidate_paths = list(tracker["candidate_paths"])
    if candidate_paths:
        evidence_lines.append(f"目标文件: {', '.join(sorted(candidate_paths)[:2])}")
    else:
        matched_target_paths = [
            path for path in sorted(tracker["matched_target_paths"]) if isinstance(path, str) and path
        ]
        if len(matched_target_paths) == 1:
            evidence_lines.append(f"已解析目标文件: {matched_target_paths[0]}")

    overview_paths = list(tracker["overview_paths"])
    if overview_paths:
        evidence_lines.append(f"已获取 file_overview: {', '.join(sorted(overview_paths)[:2])}")

    reference_symbols = list(tracker["reference_symbols"])
    if reference_symbols:
        evidence_lines.append(f"已获取 find_references: {', '.join(sorted(reference_symbols)[:3])}")

    observed_functions = [
        symbol
        for symbol in sorted(tracker["observed_functions"])
        if isinstance(symbol, str) and symbol
    ]
    if observed_functions:
        evidence_lines.append(f"已确认函数名: {', '.join(observed_functions[:6])}")

    observed_step_types = [
        step_type
        for step_type in sorted(tracker["observed_step_types"])
        if isinstance(step_type, str) and step_type
    ]
    if observed_step_types:
        evidence_lines.append(f"已确认 step.type: {', '.join(observed_step_types)}")

    primary_path = _select_primary_analysis_path(tracker)
    if primary_path:
        line_count = tracker["observed_file_line_counts"].get(primary_path)
        if isinstance(line_count, int):
            evidence_lines.append(f"已确认 {primary_path} 总行数: {line_count}")
        function_count = tracker["observed_file_function_counts"].get(primary_path)
        if isinstance(function_count, int):
            evidence_lines.append(f"已确认 {primary_path} 顶层函数数: {function_count}")
        cli_args = tracker["observed_file_cli_args"].get(primary_path)
        if isinstance(cli_args, set) and cli_args:
            evidence_lines.append(f"已确认 CLI 参数: {', '.join(sorted(cli_args))}")

    evidence_lines.append("禁止引用未观察到的标识符")

    if not evidence_lines:
        return NUDGE_ANALYSIS_CONVERGE
    return NUDGE_ANALYSIS_CONVERGE + "\n" + "\n".join(f"- {line}" for line in evidence_lines)


def _build_analysis_force_answer_nudge(tracker: dict[str, object]) -> str:
    """多次拦截重复探索后，升级为强制直接作答。"""
    return (
        _build_analysis_convergence_nudge(tracker)
        + "\n- 下一条消息必须直接输出最终答案"
        + "\n- 禁止再调用任何工具"
        + "\n- 如果还有不确定点，只能写“未确认”，不能继续 tool_calls"
    )


def _should_inject_convergence_nudge(
    *,
    exploration_history: list[tuple[str, str]],
    step_index: int,
    max_steps: int,
) -> bool:
    """在重复探索或接近上限时，提醒模型基于现有工具结果继续收敛。"""
    if not exploration_history:
        return False

    remaining_steps = max_steps - step_index - 1
    recent_explorations = exploration_history[-3:]
    recent_targets = [target for _, target in recent_explorations if target]

    repeated_target = (
        len(recent_targets) >= 2
        and len(set(recent_targets[-2:])) == 1
    )
    near_limit_after_multiple_explorations = remaining_steps <= 1 and len(recent_explorations) >= 2

    return repeated_target or near_limit_after_multiple_explorations


def run_agent_once(
    user_input: str,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    session: SessionData,
    working_memory: WorkingMemory,
    memory_pipeline: MemoryPipeline | None,
    history_summarizer: OlderHistorySummarizer | None = None,
    history: list[ChatMessage] | None = None,
    max_steps: int = 20,
    session_id: str = "",
) -> tuple[AgentStep, list[ChatMessage]]:
    """执行一轮 agent 主循环：模型 -> 工具 -> 再模型，直到完成或达到上限。"""
    # 没有历史时用空列表兜底
    if history is None:
        history = []

    # 用 MessageBuilder 统一管理本轮消息
    builder = MessageBuilder()
    builder.extend(history)
    builder.add_user(user_input)

    # 从“新用户输入”开始进入主循环
    return _run_agent_loop(
        builder=builder,
        model=model,
        tool_registry=tool_registry,
        tool_context=tool_context,
        session=session,
        max_steps=max_steps,
        working_memory=working_memory,
        memory_pipeline=memory_pipeline,
        history_summarizer=history_summarizer,
        session_id=session_id,
    )


def continue_agent_from_history(
    history: list[ChatMessage],
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    session: SessionData,
    working_memory: WorkingMemory,
    memory_pipeline: MemoryPipeline | None,
    history_summarizer: OlderHistorySummarizer | None = None,
    max_steps: int = 20,
    session_id: str = "",
) -> tuple[AgentStep, list[ChatMessage]]:
    """基于已有历史继续主循环，不再追加新的 user 消息。"""
    builder = MessageBuilder()
    builder.extend(history)
    return _run_agent_loop(
        builder=builder,
        model=model,
        tool_registry=tool_registry,
        tool_context=tool_context,
        session=session,
        max_steps=max_steps,
        working_memory=working_memory,
        memory_pipeline=memory_pipeline,
        history_summarizer=history_summarizer,
        session_id=session_id,
    )


def _run_agent_loop(
    builder: MessageBuilder,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    session: SessionData,
    max_steps: int,
    working_memory: WorkingMemory,
    memory_pipeline: MemoryPipeline | None,
    history_summarizer: OlderHistorySummarizer | None,
    session_id: str,
) -> tuple[AgentStep, list[ChatMessage]]:
    """执行真正的模型/工具循环，既可用于新请求，也可用于授权后的继续执行。"""
    # 记录整轮请求开始时间
    loop_started_at = time.perf_counter()
    exploration_history: list[tuple[str, str]] = []
    pending_user_nudge: str | None = None
    blocked_analysis_tool_call_count = 0
    task_text = _extract_latest_real_user_message(builder.build())
    analysis_tracker: dict[str, object] | None = None
    if _is_code_analysis_request(task_text):
        analysis_tracker = _create_analysis_tracker(task_text)
        target_resolution_nudge = _build_analysis_target_resolution_nudge(analysis_tracker)
        pending_user_nudge = NUDGE_ANALYSIS_TOOL_PRIORITY
        if target_resolution_nudge:
            pending_user_nudge = pending_user_nudge + "\n" + target_resolution_nudge

    # 记录本轮请求开始
    log_event(
        f"[session={session_id or '-'}] 开始一轮 Agent 请求"
    )

    # 限制循环步数，防止模型和工具来回打转
    for step_index in range(max_steps):
        # 记录当前 step 开始时间
        step_started_at = time.perf_counter()

        # 记录当前是第几轮循环
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮循环开始"
        )

        # 先取出当前完整历史。
        # 这份 full_history 只在本轮内部用于切窗口和生成摘要，
        # 不会整段原样发给模型。
        full_history = list(builder.build())

        # 把上下文准备工作委托给专门模块，避免主循环里塞入过多策略细节。
        prepared_context = prepare_agent_context(
            full_history=full_history,
            session=session,
            tool_registry=tool_registry,
            working_memory=working_memory,
            memory_pipeline=memory_pipeline,
            history_summarizer=history_summarizer,
        )

        # 记录本轮上下文裁剪结果和 token 占用情况，便于观察策略是否生效。
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮上下文窗口: "
            f"level={prepared_context.policy.level} keep_rounds={prepared_context.policy.keep_rounds} "
            f"older={len(prepared_context.history_window.older_messages)} recent={len(prepared_context.history_window.recent_messages)} "
            f"tool_truncated={prepared_context.compaction_result.truncated_tool_results} "
            f"tool_cleared={prepared_context.compaction_result.cleared_old_tool_results}"
        )
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮压缩前预估: "
            f"preview_total={prepared_context.preview_stats.total_tokens} "
            f"preview_usage={prepared_context.preview_stats.usage_ratio:.1%} "
            f"preview_budget={prepared_context.preview_stats.usable_budget} "
            f"preview_recent={prepared_context.preview_stats.recent_tokens} "
            f"preview_memory={prepared_context.preview_stats.memory_tokens} "
            f"preview_tool_results={prepared_context.preview_stats.tool_result_tokens}"
        )
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮 token统计: "
            f"total={prepared_context.stats.total_tokens} usage={prepared_context.stats.usage_ratio:.1%} "
            f"budget={prepared_context.stats.usable_budget} system={prepared_context.stats.system_tokens} "
            f"recent={prepared_context.stats.recent_tokens} memory={prepared_context.stats.memory_tokens} "
            f"tool_results={prepared_context.stats.tool_result_tokens}"
        )

        messages = _append_transient_user_nudge(
            prepared_context.messages,
            pending_user_nudge,
        )
        if pending_user_nudge:
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮注入临时引导提示"
            )
            pending_user_nudge = None

        try:
            # 记录即将请求模型
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮开始请求模型"
            )

            # 请求模型给出下一步动作
            step = model.next(messages=messages)

            # 记录模型返回类型
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮模型返回类型: {step.type}"
            )
        except Exception as error:
            if is_context_overflow_error(error):
                recovery_result = recover_from_context_overflow(
                    messages=messages,
                    usable_budget=prepared_context.stats.usable_budget,
                )
                if recovery_result.recovered:
                    log_event(
                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮触发 Reactive Compact Recover: "
                        f"strategy={recovery_result.strategy} "
                        f"before={recovery_result.tokens_before} after={recovery_result.tokens_after}"
                    )
                    try:
                        step = model.next(messages=recovery_result.messages)
                        log_event(
                            f"[session={session_id or '-'}] 第 {step_index + 1} 轮恢复后模型返回类型: {step.type}"
                        )
                    except Exception as retry_error:
                        error = retry_error
                    else:
                        if step.type == "assistant":
                            if step.kind == "progress":
                                builder.add_progress(step.content)
                                pending_user_nudge = NUDGE_CONTINUE
                                log_event(
                                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮恢复后收到 progress，继续下一轮"
                                )
                                continue

                            # 对代码分析类回答做最后一道符号校验，避免把猜测出的函数名直接放行。
                            if (
                                analysis_tracker is not None
                                and _has_sufficient_analysis_evidence(analysis_tracker)
                            ):
                                invalid_names = _find_unobserved_answer_function_names(
                                    analysis_tracker,
                                    step.content,
                                )
                                invalid_claims = _find_unsupported_analysis_claims(
                                    analysis_tracker,
                                    step.content,
                                )
                                if (invalid_names or invalid_claims) and step_index < max_steps - 1:
                                    pending_user_nudge = _build_analysis_fact_correction_nudge(
                                        analysis_tracker,
                                        invalid_names,
                                        invalid_claims,
                                    )
                                    log_event(
                                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮恢复后检测到缺少证据支撑的分析结论，要求模型自纠: "
                                        f"{', '.join((invalid_names + invalid_claims)[:6])}"
                                    )
                                    continue

                            final_content = step.content
                            if analysis_tracker is not None:
                                final_content = _normalize_analysis_answer_content(
                                    analysis_tracker,
                                    step.content,
                                )
                            step.content = final_content
                            builder.add_assistant(final_content)

                            if memory_pipeline is not None:
                                memory_pipeline.record_assistant_reply(
                                    working_memory,
                                    content=final_content,
                                )
                            else:
                                decisions = extract_decisions_from_assistant(final_content)
                                for decision in decisions:
                                    working_memory.protect(
                                        decision,
                                        entry_type="key_decision",
                                        ttl_seconds=3600,
                                        importance=0.95,
                                    )
                                    working_memory.protect(
                                        decision,
                                        entry_type="reflection_decision",
                                        ttl_seconds=3600,
                                        importance=0.95,
                                    )

                            step_cost = time.perf_counter() - step_started_at
                            total_cost = time.perf_counter() - loop_started_at
                            log_event(
                                f"[session={session_id or '-'}] 第 {step_index + 1} 轮恢复后直接返回答案 "
                                f"step耗时={step_cost:.3f}s 总耗时={total_cost:.3f}s"
                            )
                            return step, builder.build()

            # 模型调用异常时兜底为最终回答，避免主循环直接崩掉
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮模型调用异常: {error}"
            )
            # 记录最近一次模型失败，后面做 prompt 注入时可以提醒模型避坑
            working_memory.protect(
                f"模型调用失败: {error}",
                entry_type="error_context",
                ttl_seconds=1800,
                importance=0.9,
            )
            working_memory.protect(
                f"模型调用失败: {error}",
                entry_type="reflection_failure",
                ttl_seconds=1800,
                importance=0.9,
            )
            fallback = AgentStep(
                type="assistant",
                content=f"模型调用失败: {error}",
                kind="final",
            )
            builder.add_assistant(fallback.content)
            return fallback, builder.build() # type: ignore

        # 情况一：模型直接返回最终答案
        if step.type == "assistant":
            if step.kind == "progress":
                builder.add_progress(step.content)
                if analysis_tracker is not None and _has_sufficient_analysis_evidence(analysis_tracker):
                    pending_user_nudge = _build_analysis_convergence_nudge(analysis_tracker)
                else:
                    pending_user_nudge = NUDGE_CONTINUE
                step_cost = time.perf_counter() - step_started_at
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮收到 progress，继续下一轮 "
                    f"step耗时={step_cost:.3f}s"
                )
                continue

            # 对代码分析类回答做最后一道符号校验，避免把猜测出的函数名直接放行。
            if (
                analysis_tracker is not None
                and _has_sufficient_analysis_evidence(analysis_tracker)
            ):
                invalid_names = _find_unobserved_answer_function_names(
                    analysis_tracker,
                    step.content,
                )
                invalid_claims = _find_unsupported_analysis_claims(
                    analysis_tracker,
                    step.content,
                )
                if (invalid_names or invalid_claims) and step_index < max_steps - 1:
                    pending_user_nudge = _build_analysis_fact_correction_nudge(
                        analysis_tracker,
                        invalid_names,
                        invalid_claims,
                    )
                    log_event(
                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮检测到缺少证据支撑的分析结论，要求模型自纠: "
                        f"{', '.join((invalid_names + invalid_claims)[:6])}"
                    )
                    continue

            final_content = step.content
            if analysis_tracker is not None:
                final_content = _normalize_analysis_answer_content(
                    analysis_tracker,
                    step.content,
                )
                step.content = final_content

            builder.add_assistant(final_content)

            # 从最终 assistant 回复里尝试抽一条关键决策。
            # 这不是为了记录所有回答，而是尽量保留“已经确认的方向或约束”。
            if memory_pipeline is not None:
                memory_pipeline.record_assistant_reply(
                    working_memory,
                    content=final_content,
                )
            else:
                decisions = extract_decisions_from_assistant(final_content)
                for decision in decisions:
                    working_memory.protect(
                        decision,
                        entry_type="key_decision",
                        ttl_seconds=3600,
                        importance=0.95,
                    )
                    working_memory.protect(
                        decision,
                        entry_type="reflection_decision",
                        ttl_seconds=3600,
                        importance=0.95,
                    )

            # 记录当前 step 和整轮总耗时
            step_cost = time.perf_counter() - step_started_at
            total_cost = time.perf_counter() - loop_started_at
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮直接返回答案 "
                f"step耗时={step_cost:.3f}s 总耗时={total_cost:.3f}s"
            )
            return step, builder.build()

        # 情况二：模型要求调用一个或多个工具
        if step.type == "tool_calls":
            # 特殊情况：模型返回了空工具调用
            if not step.calls:
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具调用为空"
                )
                fallback = AgentStep(
                    type="assistant",
                    content="模型返回了空的工具调用。",
                    kind="final",
                )
                builder.add_assistant(fallback.content)
                return fallback, builder.build()

            if (
                analysis_tracker is not None
                and _should_redirect_analysis_to_structure_first(analysis_tracker, step.calls)
            ):
                pending_user_nudge = NUDGE_ANALYSIS_TOOL_PRIORITY + "\n" + NUDGE_ANALYSIS_STRUCTURE_FIRST
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮检测到分析任务过早直接 read_file，要求先获取结构化证据"
                )
                continue

            if (
                analysis_tracker is not None
                and _should_block_redundant_analysis_calls(
                    analysis_tracker,
                    calls=step.calls,
                    step_index=step_index,
                    max_steps=max_steps,
                )
            ):
                blocked_analysis_tool_call_count += 1
                if blocked_analysis_tool_call_count >= 2:
                    pending_user_nudge = _build_analysis_force_answer_nudge(analysis_tracker)
                else:
                    pending_user_nudge = _build_analysis_convergence_nudge(analysis_tracker)
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮检测到证据已足够，拦截重复探索工具并要求直接作答"
                )
                continue

            # 依次记录、执行并回写每个工具调用结果
            blocked_analysis_tool_call_count = 0
            for call in step.calls:
                tool_name = call["tool_name"]
                tool_input = call["input"]
                tool_use_id = call["id"]
                tool_target = _extract_tool_target(tool_input)

                # 从工具输入里尽量提取活跃路径。
                # 这一步会覆盖 path / file_path / directory / run_command 等常见形式。
                if memory_pipeline is not None:
                    memory_pipeline.record_tool_call(
                        working_memory,
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                else:
                    for path in extract_active_paths(tool_name, tool_input):
                        working_memory.protect(
                            path,
                            entry_type="active_task",
                            ttl_seconds=1800,
                            importance=0.8,
                        )
                        working_memory.protect(
                            path,
                            entry_type="reflection_file",
                            ttl_seconds=1800,
                            importance=0.7,
                        )

                # 先把工具调用请求记到历史里
                builder.add_tool_call(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    input_data=tool_input,
                )

                # 记录即将调用哪个工具
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮准备调用工具: {tool_name}"
                )

                # 记录单个工具开始时间
                tool_started_at = time.perf_counter()

                try:
                    # 统一通过 registry 执行工具
                    result = tool_registry.execute_tool(
                        tool_name=tool_name,
                        input_data=tool_input,
                        context=tool_context,
                    )

                except Exception as error:
                    # 理论上 registry 已经兜底，这里是主循环最后一层保险
                    tool_cost = time.perf_counter() - tool_started_at
                    log_event(
                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具 {tool_name} "
                        f"执行异常: {error} 耗时={tool_cost:.3f}s"
                    )
                    result = ToolResult(
                        ok=False,
                        output=f"工具调用发生未捕获异常: {error}",
                        error="UNCAUGHT_TOOL_ERROR",
                        meta={"tool_name": tool_name},
                    )

                tool_cost = time.perf_counter() - tool_started_at
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具 {tool_name} "
                    f"返回 ok={result.ok} error={result.error} 耗时={tool_cost:.3f}s"
                )

                if _is_exploration_tool(tool_name):
                    exploration_history.append((tool_name, tool_target))

                if analysis_tracker is not None:
                    _record_analysis_evidence(
                        analysis_tracker,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        result=result,
                    )

                # 工具失败时，把错误压成短摘要写进短期工作记忆。
                if not result.ok:
                    if memory_pipeline is not None:
                        memory_pipeline.record_tool_failure(
                            working_memory,
                            tool_name=tool_name,
                            result=result,
                        )
                    else:
                        failure_summary = summarize_failure(tool_name, result)
                        working_memory.protect(
                            failure_summary,
                            entry_type="error_context",
                            ttl_seconds=1800,
                            importance=0.9,
                        )
                        working_memory.protect(
                            failure_summary,
                            entry_type="reflection_failure",
                            ttl_seconds=1800,
                            importance=0.9,
                        )

                # 命中“需要授权”时，不继续喂模型，而是把审批请求返回给 main
                if result.error=="PERMISSION_REQUIRED":
                    command=str(result.meta.get("command", ""))
                    reason = str(result.meta.get("reason", ""))
                    action_key = str(result.meta.get("action_key", ""))

                    # 授权中断前也要补一条 tool_result，避免历史里只留下 tool_call
                    # 否则下一轮把这段历史再发给模型时，会因为协议断链而报 400
                    builder.add_tool_result(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        content="该操作需要用户授权，当前尚未执行。",
                        is_error=True,
                        meta=dict(result.meta),
                    )

                    approval_message = (
                        "该操作需要用户授权。\n"
                        f"工具: {tool_name}\n"
                        f"命令: {command}\n"
                        f"原因: {reason}"
                    )

                    approval_step = AgentStep(
                        type="approval",
                        content=approval_message,
                        approval=ApprovalRequest(
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            action_key=action_key,
                            message=approval_message,
                            input_data=tool_input,
                        ),
                    )
                    return approval_step, builder.build()

                # 正常情况才把工具结果写回消息历史
                context_output = result.meta.get("context_output", result.output)
                if not isinstance(context_output, str) or not context_output.strip():
                    context_output = result.output
                builder.add_tool_result(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    content=context_output,
                    is_error=not result.ok,
                    meta=dict(result.meta),
                )

            if _should_inject_convergence_nudge(
                exploration_history=exploration_history,
                step_index=step_index,
                max_steps=max_steps,
            ):
                if analysis_tracker is not None and _has_sufficient_analysis_evidence(analysis_tracker):
                    pending_user_nudge = _build_analysis_convergence_nudge(analysis_tracker)
                else:
                    pending_user_nudge = NUDGE_AFTER_TOOL_RESULT
                log_event(
                    f"[session={session_id or '-'}] 第 {step_index + 1} 轮注入临时工具结果提示"
                )

            # 记录当前工具阶段结束
            step_cost = time.perf_counter() - step_started_at
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具阶段结束 step耗时={step_cost:.3f}s"
            )
            continue

        # 情况三：遇到未知返回类型时兜底退出
        fallback = AgentStep(
            type="assistant",
            content="未识别的模型返回类型。",
            kind="final",
        )
        builder.add_assistant(fallback.content)
        return fallback, builder.build()

    # 达到最大步数时停止，防止死循环
    total_cost = time.perf_counter() - loop_started_at
    log_event(
        f"[session={session_id or '-'}] 达到最大循环步数 {max_steps} 总耗时={total_cost:.3f}s"
    )
    fallback = AgentStep(
        type="assistant",
        content="已达到最大循环步数，本轮已停止。",
        kind="final",
    )
    builder.add_assistant(fallback.content)
    return fallback, builder.build()
