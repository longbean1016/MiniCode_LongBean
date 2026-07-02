from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

"""上下文压缩执行模块，负责裁剪消息历史并保留关键恢复信息。"""

from app.context.manager import estimate_messages_tokens
from app.types import ChatMessage

# 这里只对 read_file 做"路径 + 完整内容 hash"的精确去重；
# glob_files / grep_files 更像集合结果，直接复用同一规则容易误伤。
_READ_TOOLS = {"read_file"}
_COLLECTION_TOOLS = {"glob_files", "grep_files"}
_READ_FILE_PATH_PATTERN = re.compile(r"^FILE:\s*(.+)$", re.MULTILINE)
# 对标 Claude Code microCompact.ts TIME_BASED_MC_CLEARED_MESSAGE
_MICROCOMPACT_MARKER_PREFIX = "[Old tool result content cleared]"
_EMPTY_SUCCESS_TOOL_RESULT_MARKER = "[工具执行成功，内容无额外信息]"
_EMPTY_SUCCESS_PATTERNS = (
    "文件写入成功",
    "目录创建成功",
    "命令执行成功",
    "工具执行成功",
    "执行成功",
    "保存成功",
    "创建成功",
    "写入成功",
    "success",
    "ok",
)


@dataclass(slots=True)
class CompactionResult:
    """保存 recent window 微压缩后的结果和统计信息。"""

    messages: list[ChatMessage]
    truncated_tool_results: int = 0
    cleared_old_tool_results: int = 0
    deduped_read_results: int = 0
    semantic_compacted_pairs: int = 0
    filtered_empty_tool_results: int = 0
    deduped_assistant_messages: int = 0
    dropped_progress_messages: int = 0
    priority_dropped_messages: int = 0
    tokens_freed_estimate: int = 0
    # microcompact 清正文之前会尝试用轻量模型承接关键语义，
    # 这里按 working memory 的槽位拆开，避免后续再靠规则猜测含义。
    carried_tool_findings: list[str] = field(default_factory=list)
    carried_open_issues: list[str] = field(default_factory=list)
    carried_key_decisions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReadDedupeIdentity:
    """标识一次 read_file 结果，用于和 minicode 一样做精确去重。"""

    path_key: str
    display_path: str
    content_hash: str


def compact_recent_messages(
    messages: list[ChatMessage],
    *,
    pinned_tool_names: set[str] | None = None,
    target_tokens: int | None = None,
    protected_recent_messages: int = 6,
) -> CompactionResult:
    """
    对 recent window 做微压缩。

    阶段顺序：
    1. 删除 assistant_progress
    2. 对重复读取结果做语义去重
    3. 过滤纯确认类 tool_result
    4. 去掉连续重复 assistant 回复
    5. 如果仍超预算，再做一次"保护最近消息"的优先裁剪
    """
    compacted = [
        dict(message)
        for message in messages
        if message.get("role") != "assistant_progress"
    ]
    result = CompactionResult(messages=compacted)
    result.dropped_progress_messages = max(0, len(messages) - len(compacted))

    original_tool_contents: dict[int, str] = {
        index: _get_tool_original_content(message)
        for index, message in enumerate(compacted)
        if message.get("role") == "tool_result"
    }

    _dedupe_tool_results(
        compacted=compacted,
        original_tool_contents=original_tool_contents,
        result=result,
    )
    _filter_empty_success_tool_results(compacted=compacted, result=result)
    compacted = _dedupe_consecutive_assistant_messages(
        compacted=compacted,
        result=result,
    )
    current_tokens = estimate_messages_tokens(compacted)
    if _is_within_target(current_tokens=current_tokens, target_tokens=target_tokens):
        result.messages = compacted
        return result

    compacted = _drop_low_priority_old_messages(
        compacted=compacted,
        target_tokens=target_tokens,
        protected_recent_messages=protected_recent_messages,
        result=result,
    )
    result.messages = compacted
    return result


def microcompact_old_tool_results(
    messages: list[ChatMessage],
    *,
    keep_recent_tool_rounds: int,
    pinned_tool_names: set[str] | None = None,
    semantic_summarizer: object | None = None,
) -> CompactionResult:
    """
    轻量清理较旧 tool_result 的正文，保留协议结构和最近证据。

    这一步不折叠为 assistant 摘要，也不删除 tool_call，只把较旧结果替换成
    一个可识别的占位说明，行为上更接近新版 minicode 的 microcompact。
    """
    compacted = [
        dict(message)
        for message in messages
        if message.get("role") != "assistant_progress"
    ]
    result = CompactionResult(messages=compacted)
    result.dropped_progress_messages = max(0, len(messages) - len(compacted))

    protected_tool_indexes = _collect_protected_tool_indexes_by_rounds(
        compacted=compacted,
        max_recent_tool_rounds=max(0, keep_recent_tool_rounds),
        pinned_tools={
            str(tool_name).strip()
            for tool_name in (pinned_tool_names or set())
            if str(tool_name).strip()
        },
    )
    original_tool_contents = {
        index: _get_tool_original_content(message)
        for index, message in enumerate(compacted)
        if message.get("role") == "tool_result"
    }
    _microcompact_tool_results(
        compacted=compacted,
        protected_tool_indexes=protected_tool_indexes,
        original_tool_contents=original_tool_contents,
        semantic_summarizer=semantic_summarizer,
        result=result,
    )
    result.messages = compacted
    return result


def _is_within_target(*, current_tokens: int, target_tokens: int | None) -> bool:
    """只有显式给了目标预算时，才做分阶段早停。"""
    if target_tokens is None or target_tokens <= 0:
        return False
    return current_tokens <= target_tokens


def _microcompact_tool_results(
    *,
    compacted: list[ChatMessage],
    protected_tool_indexes: set[int],
    original_tool_contents: dict[int, str] | None = None,
    semantic_summarizer: object | None = None,
    result: CompactionResult,
) -> None:
    """对较旧 tool_result 做正文清理，替换为统一占位文本。

       对标 Claude Code microCompact.ts 的 TIME_BASED_MC_CLEARED_MESSAGE。
       不再调用模型做语义摘要，纯裁剪 + 占位，零额外 API 消耗。
    """
    for index, message in enumerate(compacted):
        if message.get("role") != "tool_result":
            continue
        if index in protected_tool_indexes:
            continue
        if _is_already_microcompacted_tool_result(message):
            continue

        original_content = str(message.get("content", ""))
        # 记录原文长度用于 token 节省估算
        summary = _MICROCOMPACT_MARKER_PREFIX
        if summary == original_content:
            continue

        compacted[index]["content"] = summary
        compacted[index]["_microcompacted"] = True
        result.cleared_old_tool_results += 1
        result.tokens_freed_estimate += max(0, len(original_content) - len(summary)) // 4


def _filter_empty_success_tool_results(
    *,
    compacted: list[ChatMessage],
    result: CompactionResult,
) -> None:
    """把纯成功确认类 tool_result 改成固定占位，避免无信息日志占 prompt。"""
    for message in compacted:
        if message.get("role") != "tool_result":
            continue
        content = str(message.get("content", "")).strip()
        if not _is_empty_success_tool_result(content):
            continue
        if content == _EMPTY_SUCCESS_TOOL_RESULT_MARKER:
            continue
        message["content"] = _EMPTY_SUCCESS_TOOL_RESULT_MARKER
        message["_empty_success_tool_result"] = True
        result.filtered_empty_tool_results += 1
        result.tokens_freed_estimate += max(0, len(content) - len(_EMPTY_SUCCESS_TOOL_RESULT_MARKER)) // 4


def _is_empty_success_tool_result(content: str) -> bool:
    """识别没有额外事实的成功确认，避免误删包含路径、diff、错误详情的结果。"""
    normalized = " ".join(content.strip().split()).lower()
    if not normalized:
        return False
    if len(normalized) > 80:
        return False
    return any(normalized == pattern.lower() for pattern in _EMPTY_SUCCESS_PATTERNS)


def _dedupe_consecutive_assistant_messages(
    *,
    compacted: list[ChatMessage],
    result: CompactionResult,
) -> list[ChatMessage]:
    """连续 assistant 内容完全一致时保留最新一条，减少恢复后重复答复噪声。"""
    output: list[ChatMessage] = []
    index = 0
    while index < len(compacted):
        message = compacted[index]
        if message.get("role") != "assistant":
            output.append(dict(message))
            index += 1
            continue

        duplicate_end = index
        content = str(message.get("content", ""))
        while (
            duplicate_end + 1 < len(compacted)
            and compacted[duplicate_end + 1].get("role") == "assistant"
            and str(compacted[duplicate_end + 1].get("content", "")) == content
        ):
            duplicate_end += 1

        output.append(dict(compacted[duplicate_end]))
        result.deduped_assistant_messages += duplicate_end - index
        index = duplicate_end + 1
    return output


def _build_tool_call_input_by_id(compacted: list[ChatMessage]) -> dict[str, object]:
    """按 tool_use_id 找回原始 tool_call input，供 microcompact 模型理解工具上下文。"""
    mapping: dict[str, object] = {}
    for message in compacted:
        if message.get("role") != "assistant_tool_call":
            continue
        tool_use_id = str(message.get("tool_use_id", "")).strip()
        if not tool_use_id:
            continue
        mapping[tool_use_id] = message.get("input", {})
    return mapping


def _normalize_extracted_lines(raw_value: object, key: str) -> list[str]:
    """清洗轻量模型返回的结构化数组，保证后续 WM 写入只处理非空短文本。"""
    if not isinstance(raw_value, dict):
        return []
    raw_lines = raw_value.get(key, [])
    if not isinstance(raw_lines, list):
        return []
    lines: list[str] = []
    seen: set[str] = set()
    for item in raw_lines:
        line = " ".join(str(item).strip().split())
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def _is_already_microcompacted_tool_result(message: ChatMessage) -> bool:
    content = str(message.get("content", "")).strip()
    if not content:
        return False
    if content.startswith(_MICROCOMPACT_MARKER_PREFIX):
        return True
    return bool(
        message.get("_microcompacted")
        or message.get("_deduped_read_result")
    )


def _dedupe_tool_results(
    *,
    compacted: list[ChatMessage],
    original_tool_contents: dict[int, str],
    result: CompactionResult,
) -> None:
    """对重复读取结果做去重，但保留最新一份完整证据。"""
    # 反向扫描意味着"保留最新一份，折掉更旧的一份"。
    # 这样模型看到的仍然是离当前推理最近的结果，不会因为去重把最新证据删掉。
    last_seen_by_key: dict[tuple[str, str, str], int] = {}
    last_seen_collection_key: dict[tuple[str, str], int] = {}
    for index in range(len(compacted) - 1, -1, -1):
        message = compacted[index]
        if message.get("role") != "tool_result":
            continue

        tool_name = str(message.get("tool_name", ""))
        if tool_name not in _READ_TOOLS:
            if tool_name not in _COLLECTION_TOOLS:
                continue
            collection_key = _build_collection_dedupe_identity(
                tool_name=tool_name,
                content=original_tool_contents[index],
            )
            if collection_key is None:
                continue
            if collection_key not in last_seen_collection_key:
                last_seen_collection_key[collection_key] = index
                continue
            # list/grep 更像"扫描快照"，这里不要求逐字符完全一致，
            # 而是按头部统计 + 结果集合做轻量签名，避免同一轮反复扫目录把上下文刷爆。
            compacted[index]["content"] = (
                "[扫描结果已去重：保留本轮较新的同类结果]\n"
                f"工具: {tool_name}"
            )
            compacted[index]["_deduped_read_result"] = True
            result.deduped_read_results += 1
            continue

        read_identity = _build_read_dedupe_identity(
            message=message,
            original_content=original_tool_contents[index],
        )
        if read_identity is None:
            continue

        dedupe_key = (tool_name, read_identity.path_key, read_identity.content_hash)
        if dedupe_key not in last_seen_by_key:
            last_seen_by_key[dedupe_key] = index
            continue

        # read_file 则必须更严格。
        # 只有"规范化路径 + 完整内容 hash"都相同，才认为是同一份文件事实；
        # 任何一个字符变化，都必须保留为新证据，不能偷懒按文件名模糊折叠。
        compacted[index]["content"] = (
            "[读取结果已去重：文件内容未变化，保留本轮较新的同文件结果]\n"
            f"文件: {read_identity.display_path}"
        )
        compacted[index]["_deduped_read_result"] = True
        result.deduped_read_results += 1


def _collect_protected_tool_indexes_by_rounds(
    *,
    compacted: list[ChatMessage],
    max_recent_tool_rounds: int,
    pinned_tools: set[str],
) -> set[int]:
    """保护最近 N 轮的全部 tool_result（按 user 消息切分轮次）。

    这些结果往往承载真实文件事实，不能在微压缩阶段被提前折掉。
    """
    keep_indexes: set[int] = set()

    if max_recent_tool_rounds > 0:
        # 找到所有 user 消息的位置作为轮边界
        round_boundaries: list[int] = []
        for i, msg in enumerate(compacted):
            if str(msg.get("role", "")) == "user":
                round_boundaries.append(i)

        if round_boundaries:
            protected_start = round_boundaries[
                -min(max_recent_tool_rounds, len(round_boundaries))
            ]
            for i, msg in enumerate(compacted):
                if msg.get("role") == "tool_result" and i >= protected_start:
                    keep_indexes.add(i)
        else:
            # 没有 user 消息，回退：保护全部
            for i, msg in enumerate(compacted):
                if msg.get("role") == "tool_result":
                    keep_indexes.add(i)
    # max_recent_tool_rounds <= 0 时不保护任何结果

    # pinned_tools：保护指定工具的最新一次结果
    if pinned_tools:
        pending = set(pinned_tools)
        for i in range(len(compacted) - 1, -1, -1):
            msg = compacted[i]
            if msg.get("role") != "tool_result":
                continue
            tool_name = str(msg.get("tool_name", "")).strip()
            if tool_name in pending:
                keep_indexes.add(i)
                pending.remove(tool_name)
            if not pending:
                break

    return keep_indexes


def _drop_low_priority_old_messages(
    *,
    compacted: list[ChatMessage],
    target_tokens: int | None,
    protected_recent_messages: int,
    result: CompactionResult,
) -> list[ChatMessage]:
    """
    如果前面几段处理完仍超预算，再做一次保守末段裁剪。

    这里不碰幸存下来的结构化 tool pair，只裁剪：
    - 旧的语义工具摘要
    - 更早的普通 assistant 消息
    - 最后才动 user 消息
    """
    if target_tokens is None or target_tokens <= 0:
        return compacted

    working = [dict(message) for message in compacted]
    protected_recent_messages = max(0, protected_recent_messages)

    # 这一步是兜底，不是主力压缩手段。
    # 只有前面的"预览化 / 去重 / 语义折叠"都做完仍超预算，才允许直接删旧消息。
    while estimate_messages_tokens(working) > target_tokens:
        protected_start = max(0, len(working) - protected_recent_messages)
        candidate_index = _find_low_priority_drop_index(
            messages=working,
            protected_start=protected_start,
        )
        if candidate_index is None:
            break

        removed_message = working.pop(candidate_index)
        result.priority_dropped_messages += 1
        result.tokens_freed_estimate += estimate_messages_tokens([removed_message])

    return working


def _find_low_priority_drop_index(
    *,
    messages: list[ChatMessage],
    protected_start: int,
) -> int | None:
    """从可裁剪区间里挑出一条最该先被删的旧消息。"""
    best_index: int | None = None
    best_priority = -1

    for index, message in enumerate(messages):
        if index >= protected_start:
            break
        if _is_non_droppable_structured_tool_message(message):
            continue

        priority = _drop_priority(message)
        if priority > best_priority:
            best_priority = priority
            best_index = index

    return best_index


def _is_non_droppable_structured_tool_message(message: ChatMessage) -> bool:
    """幸存到这一步的结构化 tool pair，视为最近证据，不在末段裁剪里再拆。"""
    role = str(message.get("role", ""))
    return role in {"assistant_tool_call", "tool_result"}


def _drop_priority(message: ChatMessage) -> int:
    """
    数值越大越先被裁。

    顺序：
    - 旧工具语义摘要
    - 普通 assistant
    - user
    """
    role = str(message.get("role", ""))
    if role == "assistant" and bool(message.get("_semantic_tool_summary")):
        return 3
    if role == "assistant":
        return 2
    if role == "user":
        return 1
    return 0
def _get_tool_original_content(message: ChatMessage) -> str:
    """从消息里优先取 raw_output，没有时再退回当前 content。"""
    meta = message.get("meta", {})
    if isinstance(meta, dict):
        raw_output = meta.get("raw_output")
        if isinstance(raw_output, str) and raw_output:
            return raw_output
    return str(message.get("content", ""))


def _build_read_dedupe_identity(
    *,
    message: ChatMessage,
    original_content: str,
) -> ReadDedupeIdentity | None:
    """基于文件路径和完整内容 hash 构建 read_file 去重标识。"""
    display_path = _extract_read_file_path(message, original_content)
    if not display_path:
        return None

    # 这里和 minicode 一样按完整文本做精确 hash。
    # 只要内容有一个字符变化，就会视为新版本，避免误去重。
    content_hash = hashlib.md5(
        original_content.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return ReadDedupeIdentity(
        path_key=_normalize_read_path(display_path),
        display_path=display_path,
        content_hash=content_hash,
    )


def _extract_read_file_path(message: ChatMessage, original_content: str) -> str | None:
    """优先从 meta.path 取 read_file 路径，缺失时再回退解析 FILE 头。"""
    meta = message.get("meta", {})
    if isinstance(meta, dict):
        meta_path = meta.get("path")
        if isinstance(meta_path, str) and meta_path.strip():
            return meta_path.strip()

    match = _READ_FILE_PATH_PATTERN.search(original_content)
    if not match:
        return None
    return match.group(1).strip() or None


def _normalize_read_path(path: str) -> str:
    """对路径做轻量规范化，减少斜杠和大小写差异带来的误判。"""
    normalized = os.path.normpath(path.strip())
    return os.path.normcase(normalized)


def _build_collection_dedupe_identity(*, tool_name: str, content: str) -> tuple[str, str] | None:
    """给 glob_files / grep_files 生成轻量语义去重标识。"""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None

    header_lines: list[str] = []
    body_lines: list[str] = []
    in_body = False
    for line in lines:
        if not in_body and (line.startswith("file ") or ":" in line and tool_name == "grep_files"):
            in_body = True
        if in_body:
            body_lines.append(line)
        else:
            header_lines.append(line)

    normalized_header = "|".join(header_lines[:6]).lower()
    normalized_body = "|".join(sorted(body_lines[:12])).lower()
    return (tool_name, f"{normalized_header}::{normalized_body}")
