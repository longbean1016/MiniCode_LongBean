from __future__ import annotations

"""代码分析专项护栏，负责证据跟踪、答案校验和收敛提示。"""

import re

from app.types import ToolResult

NUDGE_ANALYSIS_CONVERGE = (
    "现有证据已经足够回答这次代码分析问题。"
    "禁止继续重复读取相同文件、相同符号或相同目录；"
    "请直接基于已有证据给出最终答案，并明确写出仍然不确定的点。"
)
NUDGE_ANALYSIS_TOOL_PRIORITY = (
    "这是代码链路分析任务。优先使用 grep_files 或 glob_files 这类搜索工具确认真实函数位置，"
    "再按需使用 read_file 看局部分块；不要一上来只靠连续分块阅读。"
)
NUDGE_ANALYSIS_STRUCTURE_FIRST = (
    "当前仍处于代码分析取证阶段。请先使用 grep_files、glob_files "
    "确认文件结构和符号位置，再决定是否需要 read_file 看局部源码。"
)

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
# 分析阶段优先使用的工具（对齐新工具集）
_ANALYSIS_STRUCTURE_FIRST_TOOLS = {"glob_files", "grep_files"}


def _normalize_target(value: str) -> str:
    """统一路径/符号文本，方便后续做重复探索判断。"""
    return value.replace("\\", "/").strip()


def _basename_for_path(path: str) -> str:
    """从规范化路径里取出文件名，便于把“main.py”解析到真实路径。"""
    normalized = _normalize_target(path)
    return normalized.rsplit("/", 1)[-1]


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
    """从工具输出里提取真实符号名和函数名（已适配新工具集）。"""
    observed_symbols: set[str] = set()
    observed_functions: set[str] = set()

    if not raw_output.strip():
        return observed_symbols, observed_functions

    # grep_files content 模式 — 从匹配的行中提取标识符
    if tool_name == "grep_files":
        for raw_line in raw_output.splitlines():
            if ":" not in raw_line:
                continue
            line = raw_line.strip()
            for match in _FUNCTION_CALL_PATTERN.findall(line):
                if _IDENTIFIER_PATTERN.fullmatch(match):
                    observed_symbols.add(match)
            for match in _DOTTED_FUNCTION_CALL_PATTERN.findall(line):
                last = match.rsplit(".", 1)[-1]
                if _IDENTIFIER_PATTERN.fullmatch(last):
                    observed_symbols.add(last)
        return observed_symbols, observed_functions

    # read_file — 从源码中提取函数名和类名
    if tool_name == "read_file":
        for raw_line in raw_output.splitlines():
            line = raw_line.strip()
            # def / async def
            match = re.match(r"^(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if match:
                fn = match.group(1)
                observed_symbols.add(fn)
                observed_functions.add(fn)
                continue
            # class
            match = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if match:
                observed_symbols.add(match.group(1))
                continue
        return observed_symbols, observed_functions

    return observed_symbols, observed_functions


def _extract_analysis_facts_from_tool_result(
    tool_name: str,
    raw_output: str,
) -> tuple[set[str], int | None]:
    """从工具结果里提取 step.type 等可直接校验的事实。"""
    observed_step_types = set(_STEP_TYPE_PATTERN.findall(raw_output))
    line_count: int | None = None

    if tool_name in {"read_file", "grep_files", "glob_files"}:
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
        offset = int(raw_offset) if raw_offset and raw_offset.isdigit() else None

    if not isinstance(end, int):
        raw_end = parsed_headers.get("END")
        end = int(raw_end) if raw_end and raw_end.isdigit() else None

    if not isinstance(total_chars, int):
        raw_total = parsed_headers.get("TOTAL_CHARS")
        total_chars = int(raw_total) if raw_total and raw_total.isdigit() else None

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

    # 这里做的不是“保存原始工具输出”，而是把后面真正要校验的事实拆出来：
    # 例如目标文件路径、已确认的函数名、read_file 覆盖范围、CLI 参数等。
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
            # read_file 用“路径 + offset + limit”做精确片段签名。
            # 这样后面判断重复探索时，不会把不同片段误认为同一次读取。
            signatures = read_segments.setdefault(target, set())
            if isinstance(signatures, set):
                signatures.add(_build_analysis_read_signature(target, signature_offset, signature_limit))
        if offset == 0 and truncated is False:
            # 只有从 0 开始且明确未截断，才算完整读过文件。
            # 否则即使读到过这个路径，也不能支持“我已经完整分析了该文件”的结论。
            fully_read_paths.add(target)
            truncated_read_paths.discard(target)
        elif truncated is True:
            truncated_read_paths.add(target)
        return

    if tool_name == "glob_files" and result.ok:
        # glob_files 等同于结构探索，计入 structured_hits
        tracker["structured_hits"] = int(tracker["structured_hits"]) + 1
        return

    if (
        tool_name == "grep_files"
        and not result.ok
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

    if requested_basenames and not candidate_paths and len(matched_target_paths) != 1:
        return False

    if primary_path:
        if primary_path in truncated_read_paths and primary_path not in fully_read_paths:
            return False
        if primary_path not in covered_paths:
            return False

        if analysis_kind == "call_chain":
            # 链路分析比普通文件概览更严格：
            # 既要看过结构化入口，也要至少看过一段真实源码或 AST，避免只靠概览硬串流程。
            if primary_path not in overview_paths and primary_path not in ast_paths:
                return False
            if primary_path not in read_paths and primary_path not in ast_paths:
                return False
        return True

    if candidate_paths:
        return any(path in covered_paths for path in candidate_paths)

    return bool(covered_paths) and analysis_kind != "call_chain"


def _all_calls_are_exploration(
    calls: list[dict[str, object]],
    *,
    is_exploration_tool=lambda _tool_name: True,
) -> bool:
    return bool(calls) and all(is_exploration_tool(str(call.get("tool_name", ""))) for call in calls)


def _should_block_redundant_analysis_calls(
    tracker: dict[str, object],
    *,
    calls: list[dict[str, object]],
    step_index: int,
    max_steps: int,
    is_exploration_tool=lambda _tool_name: True,
) -> bool:
    """证据够了之后，拦掉明显重复的探索工具调用，逼模型进入答案收敛。"""
    if not _has_sufficient_analysis_evidence(tracker):
        return False
    if not _all_calls_are_exploration(calls, is_exploration_tool=is_exploration_tool):
        return False

    # 进入这里说明“这批 calls 都是探索类工具，且已有答案所需证据”。
    # 下面要判断的是：这些探索是不是还在重复读旧内容，而不是继续补新证据。
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
                # 如果这次读取的还是同一片 offset/limit，直接视为重复探索。
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
        # 接近最大步数时，宁可要求收敛，也不再给模型继续兜圈子的机会。
        return True

    return redundant_calls == len(calls) and redundant_calls > 0


def _build_analysis_convergence_nudge(tracker: dict[str, object]) -> str:
    """把已掌握的证据压成极短提示，提醒模型直接作答。"""
    evidence_lines: list[str] = []
    analysis_kind = str(tracker.get("analysis_kind", "file_summary"))

    # 这里只挑“足够约束答案”的少量事实，不把整份 tracker 倒给模型。
    # 目的不是复述工具输出，而是提醒模型：哪些内容已经确认、哪些内容禁止脑补。
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
