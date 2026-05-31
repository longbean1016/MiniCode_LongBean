from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

"""上下文压缩执行模块，负责裁剪消息历史并保留关键恢复信息。"""

from app.context_manager import estimate_messages_tokens
from app.types import ChatMessage

# 这里只对 read_file 做“路径 + 完整内容 hash”的精确去重；
# list_files / grep_files 更像集合结果，直接复用同一规则容易误伤。
_READ_TOOLS = {"read_file"}
_COLLECTION_TOOLS = {"list_files", "grep_files"}
_PREVIEW_HEAD_LINES = 8
_PREVIEW_TAIL_LINES = 3
_PREVIEW_LINE_CHAR_LIMIT = 160
_READ_FILE_PATH_PATTERN = re.compile(r"^FILE:\s*(.+)$", re.MULTILINE)
_HEADER_PATTERN = re.compile(r"^([A-Z_]+):\s*(.+)$")
_SUMMARY_LINE_CHAR_LIMIT = 120


@dataclass(slots=True)
class CompactionResult:
    """保存 recent window 微压缩后的结果和统计信息。"""

    messages: list[ChatMessage]
    truncated_tool_results: int = 0
    cleared_old_tool_results: int = 0
    deduped_read_results: int = 0
    semantic_compacted_pairs: int = 0
    dropped_progress_messages: int = 0
    priority_dropped_messages: int = 0
    tokens_freed_estimate: int = 0


@dataclass(slots=True)
class ReadDedupeIdentity:
    """标识一次 read_file 结果，用于和 minicode 一样做精确去重。"""

    path_key: str
    display_path: str
    content_hash: str


def compact_recent_messages(
    messages: list[ChatMessage],
    *,
    max_recent_tool_results: int,
    truncate_tool_result_chars: int,
    workspace: str | None = None,
    pinned_tool_names: set[str] | None = None,
    target_tokens: int | None = None,
    protected_recent_messages: int = 6,
) -> CompactionResult:
    """
    对 recent window 做更接近 minicode 的微压缩。

    关键点有两条：
    1. 先按阶段处理，再在每一阶段后检查“预算是否已经足够”
    2. 如果预算还不够，优先压旧消息，不提前破坏最近结构化证据

    阶段顺序：
    1. 删除 assistant_progress
    2. 对超大 tool_result 落盘并替换成首尾预览
    3. 对重复读取结果做语义去重
    4. 把过旧的 tool_call + tool_result 折叠成 assistant 摘要
    5. 如果仍超预算，再做一次“保护最近消息”的优先裁剪
    """
    # 第一步先去掉 progress 类消息。
    # 这类消息只对“流式展示过程”有用，对下一次推理几乎没有事实价值，
    # 越早删掉，后面的压缩就越能把预算留给真实证据。
    compacted = [
        dict(message)
        for message in messages
        if message.get("role") != "assistant_progress"
    ]
    result = CompactionResult(messages=compacted)
    result.dropped_progress_messages = max(0, len(messages) - len(compacted))

    workspace_path = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    cache_dir = workspace_path / ".cache"
    pinned_tools = {
        str(tool_name).strip()
        for tool_name in (pinned_tool_names or set())
        if str(tool_name).strip()
    }

    # 优先使用工具原始输出；如果 ToolRegistry 已经做过 smart truncate，
    # 会把完整原文放进 meta.raw_output。
    # 这里单独缓存“工具原始输出”，后面的每个压缩阶段都基于同一份原文做判断。
    # 这样即使 message["content"] 已经被预览版替换，去重/摘要阶段仍能基于完整证据工作。
    original_tool_contents: dict[int, str] = {
        index: _get_tool_original_content(message)
        for index, message in enumerate(compacted)
        if message.get("role") == "tool_result"
    }

    current_tokens = estimate_messages_tokens(compacted)
    if _is_within_target(current_tokens=current_tokens, target_tokens=target_tokens):
        result.messages = compacted
        return result

    effective_truncate_chars = _resolve_truncate_budget(
        base_truncate_chars=truncate_tool_result_chars,
        current_tokens=current_tokens,
        target_tokens=target_tokens,
    )
    _truncate_large_tool_results(
        compacted=compacted,
        original_tool_contents=original_tool_contents,
        truncate_tool_result_chars=effective_truncate_chars,
        cache_dir=cache_dir,
        workspace_path=workspace_path,
        result=result,
    )
    current_tokens = estimate_messages_tokens(compacted)
    if _is_within_target(current_tokens=current_tokens, target_tokens=target_tokens):
        result.messages = compacted
        return result

    _dedupe_tool_results(
        compacted=compacted,
        original_tool_contents=original_tool_contents,
        result=result,
    )
    current_tokens = estimate_messages_tokens(compacted)
    if _is_within_target(current_tokens=current_tokens, target_tokens=target_tokens):
        result.messages = compacted
        return result

    # 进入语义压缩前，先挑出“绝不能折”的最近工具结果。
    # recent window 的核心目标不是尽可能短，而是尽量让模型下一步还能继续引用最近证据。
    protected_tool_indexes = _collect_protected_tool_indexes(
        compacted=compacted,
        max_recent_tool_results=max_recent_tool_results,
        pinned_tools=pinned_tools,
    )
    protected_pair_indexes = _expand_protected_pair_indexes(
        compacted=compacted,
        protected_tool_indexes=protected_tool_indexes,
    )
    compacted = _semantic_compact_old_tool_interactions(
        compacted=compacted,
        original_tool_contents=original_tool_contents,
        protected_indexes=protected_pair_indexes,
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


def _is_within_target(*, current_tokens: int, target_tokens: int | None) -> bool:
    """只有显式给了目标预算时，才做分阶段早停。"""
    if target_tokens is None or target_tokens <= 0:
        return False
    return current_tokens <= target_tokens


def _resolve_truncate_budget(
    *,
    base_truncate_chars: int,
    current_tokens: int,
    target_tokens: int | None,
) -> int:
    """
    根据当前预算压力动态收紧 tool_result 的触发阈值。

    压力越大，就越早把大结果替换成 preview；
    压力不大时，尽量沿用外层 policy 给出的阈值，避免过度压缩。
    """
    if base_truncate_chars <= 0:
        return 0
    if target_tokens is None or target_tokens <= 0 or current_tokens <= target_tokens:
        return base_truncate_chars

    pressure_ratio = target_tokens / max(current_tokens, 1)
    scaled = int(base_truncate_chars * max(0.35, min(1.0, pressure_ratio)))
    return max(600, min(base_truncate_chars, scaled))


def _truncate_large_tool_results(
    *,
    compacted: list[ChatMessage],
    original_tool_contents: dict[int, str],
    truncate_tool_result_chars: int,
    cache_dir: Path,
    workspace_path: Path,
    result: CompactionResult,
) -> None:
    """先把过大的 tool_result 压成可追溯的首尾预览，而不是整段砍头去尾。"""
    for index, message in enumerate(compacted):
        if message.get("role") != "tool_result":
            continue

        original_content = original_tool_contents[index]
        if len(original_content) <= truncate_tool_result_chars:
            continue

        tool_name = str(message.get("tool_name", "")) or "unknown"
        persisted_path = _persist_tool_result(
            cache_dir=cache_dir,
            content=original_content,
            tool_name=tool_name,
            message_index=index,
        )
        preview = _generate_tool_result_preview(
            content=original_content,
            tool_name=tool_name,
            persisted_path=persisted_path,
            workspace_path=workspace_path,
        )

        compacted[index]["content"] = preview
        compacted[index]["_persisted_path"] = str(persisted_path)
        compacted[index]["_tool_result_preview"] = True
        result.truncated_tool_results += 1
        result.tokens_freed_estimate += max(0, len(original_content) - len(preview)) // 4


def _dedupe_tool_results(
    *,
    compacted: list[ChatMessage],
    original_tool_contents: dict[int, str],
    result: CompactionResult,
) -> None:
    """对重复读取结果做去重，但保留最新一份完整证据。"""
    # 反向扫描意味着“保留最新一份，折掉更旧的一份”。
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
            # list/grep 更像“扫描快照”，这里不要求逐字符完全一致，
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
        # 只有“规范化路径 + 完整内容 hash”都相同，才认为是同一份文件事实；
        # 任何一个字符变化，都必须保留为新证据，不能偷懒按文件名模糊折叠。
        compacted[index]["content"] = (
            "[读取结果已去重：文件内容未变化，保留本轮较新的同文件结果]\n"
            f"文件: {read_identity.display_path}"
        )
        compacted[index]["_deduped_read_result"] = True
        result.deduped_read_results += 1


def _collect_protected_tool_indexes(
    *,
    compacted: list[ChatMessage],
    max_recent_tool_results: int,
    pinned_tools: set[str],
) -> set[int]:
    """
    保护最近的 tool_result，以及分析模式下指定工具的最新一次结果。

    这些结果往往承载真实文件事实和符号表，不能在微压缩阶段被提前折掉。
    """
    tool_result_indexes = [
        index
        for index, message in enumerate(compacted)
        if message.get("role") == "tool_result"
    ]
    keep_indexes = set(tool_result_indexes[-max_recent_tool_results:])

    if pinned_tools:
        pending = set(pinned_tools)
        for index in range(len(compacted) - 1, -1, -1):
            message = compacted[index]
            if message.get("role") != "tool_result":
                continue
            tool_name = str(message.get("tool_name", "")).strip()
            if tool_name in pending:
                keep_indexes.add(index)
                pending.remove(tool_name)
            if not pending:
                break

    return keep_indexes


def _expand_protected_pair_indexes(
    *,
    compacted: list[ChatMessage],
    protected_tool_indexes: set[int],
) -> set[int]:
    """
    保护最近 tool_result 时，同步保护其对应 assistant_tool_call。

    否则后续 normalize_tool_call_pairs 会把孤立的 tool_result 丢掉，
    等于我们自己先压坏了协议结构。
    """
    protected_indexes = set(protected_tool_indexes)
    call_index_by_id: dict[str, int] = {}
    for index, message in enumerate(compacted):
        if message.get("role") != "assistant_tool_call":
            continue
        tool_use_id = str(message.get("tool_use_id", "")).strip()
        if tool_use_id:
            call_index_by_id[tool_use_id] = index

    for tool_index in protected_tool_indexes:
        tool_use_id = str(compacted[tool_index].get("tool_use_id", "")).strip()
        if tool_use_id and tool_use_id in call_index_by_id:
            protected_indexes.add(call_index_by_id[tool_use_id])
    return protected_indexes


def _semantic_compact_old_tool_interactions(
    *,
    compacted: list[ChatMessage],
    original_tool_contents: dict[int, str],
    protected_indexes: set[int],
    result: CompactionResult,
) -> list[ChatMessage]:
    """
    把过旧工具轮次压成 assistant 语义摘要。

    这一步比“直接把旧 tool_result 改成已省略占位”更安全：
    - 模型还能看到“调用了什么、产出了什么”
    - 旧 pair 会被压扁为普通消息，不再依赖 tool_call/tool_result 协议
    - 最近和被 pin 的结果仍保留原始结构，继续支持精确引用
    """
    # 先把 tool_use_id 建立成索引，后面才能按“完整 pair”做折叠。
    # 这里的核心不是压某一条消息，而是把一次旧工具交互整体沉淀成语义摘要。
    call_index_by_id: dict[str, int] = {}
    tool_index_by_id: dict[str, int] = {}
    for index, message in enumerate(compacted):
        tool_use_id = str(message.get("tool_use_id", "")).strip()
        if not tool_use_id:
            continue
        if message.get("role") == "assistant_tool_call":
            call_index_by_id[tool_use_id] = index
        elif message.get("role") == "tool_result":
            tool_index_by_id[tool_use_id] = index

    indexes_to_drop: set[int] = set()
    rewritten_messages: list[ChatMessage] = []
    for index, message in enumerate(compacted):
        if index in indexes_to_drop:
            continue

        role = message.get("role")
        if role == "assistant_tool_call":
            tool_use_id = str(message.get("tool_use_id", "")).strip()
            pair_index = tool_index_by_id.get(tool_use_id, -1)
            if index in protected_indexes or pair_index in protected_indexes:
                rewritten_messages.append(dict(message))
                continue

            # 老的 tool_call 由后面的 tool_result 摘要统一承接，这里直接丢掉。
            indexes_to_drop.add(index)
            continue

        if role == "tool_result":
            if index in protected_indexes:
                rewritten_messages.append(dict(message))
                continue

            tool_use_id = str(message.get("tool_use_id", "")).strip()
            call_index = call_index_by_id.get(tool_use_id, -1)
            if call_index >= 0:
                indexes_to_drop.add(call_index)

            # 只有在确定这条 tool_result 已经不需要保留结构协议时，
            # 才把它压成 assistant 文本。这样压完之后，模型看到的是一条普通语义事实，
            # 不再误以为还存在一个待继续衔接的 tool_call/tool_result 协议对。
            summary = _build_semantic_tool_summary(
                tool_result=message,
                tool_call=compacted[call_index] if call_index >= 0 else None,
                original_content=original_tool_contents.get(index, str(message.get("content", ""))),
            )
            current_content = str(message.get("content", ""))
            rewritten_messages.append(
                {
                    "role": "assistant",
                    "content": summary,
                    "_semantic_tool_summary": True,
                }
            )
            result.semantic_compacted_pairs += 1
            result.cleared_old_tool_results += 1
            result.tokens_freed_estimate += max(0, len(current_content) - len(summary)) // 4
            continue

        rewritten_messages.append(dict(message))

    return rewritten_messages


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
    # 只有前面的“预览化 / 去重 / 语义折叠”都做完仍超预算，才允许直接删旧消息。
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


def _build_semantic_tool_summary(
    *,
    tool_result: ChatMessage,
    tool_call: ChatMessage | None,
    original_content: str,
) -> str:
    """把一对旧工具消息压成一条尽量保住事实的语义摘要。"""
    tool_name = str(tool_result.get("tool_name", "") or "unknown")
    is_error = bool(tool_result.get("is_error"))
    if is_error:
        error_line = _extract_first_meaningful_line(original_content)
        error_text = _shorten_text(error_line or "工具执行失败")
        return f"[旧工具结果摘要] {tool_name} 失败：{error_text}"

    if tool_name == "read_file":
        return _build_read_file_summary(tool_result=tool_result, original_content=original_content)
    if tool_name in _COLLECTION_TOOLS:
        return _build_collection_summary(tool_name=tool_name, original_content=original_content)

    input_summary = _summarize_tool_input(tool_call.get("input") if tool_call else None)
    result_summary = _extract_first_meaningful_line(original_content)
    if input_summary and result_summary:
        return f"[旧工具结果摘要] {tool_name}({input_summary}) -> {_shorten_text(result_summary)}"
    if result_summary:
        return f"[旧工具结果摘要] {tool_name} -> {_shorten_text(result_summary)}"
    if input_summary:
        return f"[旧工具结果摘要] {tool_name}({input_summary}) 已执行"
    return f"[旧工具结果摘要] {tool_name} 已执行"


def _build_read_file_summary(*, tool_result: ChatMessage, original_content: str) -> str:
    """优先保留 read_file 的文件路径与覆盖范围。"""
    meta = tool_result.get("meta", {})
    file_path = ""
    if isinstance(meta, dict):
        raw_path = meta.get("path")
        if isinstance(raw_path, str):
            file_path = raw_path.strip()
    if not file_path:
        file_path = _extract_read_file_path(tool_result, original_content) or "unknown"

    headers = _parse_key_value_headers(original_content)
    offset = headers.get("OFFSET", "?")
    end = headers.get("END", "?")
    total_chars = headers.get("TOTAL_CHARS", "?")
    truncated = headers.get("TRUNCATED", "?")
    return (
        f"[旧工具结果摘要] read_file -> {file_path} "
        f"(offset={offset}, end={end}, total_chars={total_chars}, truncated={truncated})"
    )


def _build_collection_summary(*, tool_name: str, original_content: str) -> str:
    """优先保留 list/grep 这类集合工具的头部统计字段。"""
    headers = _parse_key_value_headers(original_content)
    if tool_name == "list_files":
        root = headers.get("ROOT", "?")
        total_entries = headers.get("TOTAL_ENTRIES", "?")
        returned_entries = headers.get("RETURNED_ENTRIES", "?")
        truncated = headers.get("TRUNCATED", "?")
        return (
            f"[旧工具结果摘要] list_files -> root={root}, "
            f"total_entries={total_entries}, returned_entries={returned_entries}, truncated={truncated}"
        )

    pattern = headers.get("PATTERN", "?")
    total_matches = headers.get("TOTAL_MATCHES", headers.get("MATCHES", "?"))
    returned_matches = headers.get("RETURNED_MATCHES", headers.get("RETURNED", "?"))
    truncated = headers.get("TRUNCATED", "?")
    return (
        f"[旧工具结果摘要] grep_files -> pattern={pattern}, "
        f"total_matches={total_matches}, returned_matches={returned_matches}, truncated={truncated}"
    )


def _summarize_tool_input(raw_input: object) -> str:
    """把工具入参压成一段短文本，避免摘要反过来膨胀。"""
    if raw_input is None:
        return ""
    if isinstance(raw_input, dict):
        preferred_keys = ("path", "pattern", "root", "symbol", "query", "command")
        parts: list[str] = []
        for key in preferred_keys:
            value = raw_input.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                parts.append(f"{key}={_shorten_text(text, limit=36)}")
        if parts:
            return ", ".join(parts[:3])
        try:
            return _shorten_text(json.dumps(raw_input, ensure_ascii=False), limit=48)
        except TypeError:
            return _shorten_text(str(raw_input), limit=48)
    return _shorten_text(str(raw_input), limit=48)


def _extract_first_meaningful_line(text: str) -> str:
    """提取第一条适合放进摘要的有效文本，跳过结构头和空行。"""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _HEADER_PATTERN.match(line):
            continue
        return line
    return ""


def _parse_key_value_headers(text: str) -> dict[str, str]:
    """解析 read_file / list_files / grep_files 常见的 KEY: VALUE 头部。"""
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            break
        match = _HEADER_PATTERN.match(line)
        if not match:
            continue
        parsed[match.group(1)] = match.group(2).strip()
    return parsed


def _shorten_text(text: str, *, limit: int = _SUMMARY_LINE_CHAR_LIMIT) -> str:
    """把摘要单行限制到较短长度，避免微压缩摘要再次撑大上下文。"""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


def _get_tool_original_content(message: ChatMessage) -> str:
    """从消息里优先取 raw_output，没有时再退回当前 content。"""
    meta = message.get("meta", {})
    if isinstance(meta, dict):
        raw_output = meta.get("raw_output")
        if isinstance(raw_output, str) and raw_output:
            return raw_output
    return str(message.get("content", ""))


def _persist_tool_result(
    *,
    cache_dir: Path,
    content: str,
    tool_name: str,
    message_index: int,
) -> Path:
    """把超大 tool_result 原文落盘，方便后续排查和人工查看。"""
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_tool_name = _sanitize_tool_name(tool_name)
    file_name = f"tool_result_{safe_tool_name}_{message_index}_{int(time.time() * 1000)}.txt"
    persisted_path = cache_dir / file_name

    meta = {
        "tool_name": tool_name,
        "message_index": message_index,
        "original_size": len(content),
        "timestamp": time.time(),
    }
    header = json.dumps(meta, ensure_ascii=False) + "\n---CONTENT---\n"

    # 先写临时文件再原子替换，避免中途异常时留下半截文件，
    # 也避免并发读到不完整内容。
    temp_fd, temp_path = tempfile.mkstemp(
        dir=str(cache_dir),
        prefix=".tool_result_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            handle.write(header)
            handle.write(content)
        os.replace(temp_path, persisted_path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise

    return persisted_path


def _generate_tool_result_preview(
    *,
    content: str,
    tool_name: str,
    persisted_path: Path,
    workspace_path: Path,
) -> str:
    """生成 minicode 风格的按行预览，而不是整段按字符切头尾。"""
    lines = content.splitlines() or [content]
    head_lines = lines[:_PREVIEW_HEAD_LINES]

    tail_lines: list[str] = []
    if len(lines) > (_PREVIEW_HEAD_LINES + _PREVIEW_TAIL_LINES + 1):
        tail_lines = lines[-_PREVIEW_TAIL_LINES:]

    display_path = _format_display_path(persisted_path, workspace_path)
    parts = [
        f"[工具结果已落盘，为节省上下文已截断，原始字符数 {len(content)}]",
        f"工具: {tool_name}",
        f"路径: {display_path}",
        "",
        "--- 预览（首尾几行） ---",
    ]
    parts.extend(_trim_preview_line(line) for line in head_lines)

    omitted_lines = len(lines) - len(head_lines) - len(tail_lines)
    if omitted_lines > 0:
        parts.append(f"...（省略 {omitted_lines} 行）...")

    if tail_lines:
        parts.extend(_trim_preview_line(line) for line in tail_lines)

    return "\n".join(parts)


def _sanitize_tool_name(tool_name: str) -> str:
    """把工具名转换成适合落盘的文件名片段。"""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name).strip("._")
    return normalized or "unknown"


def _format_display_path(path: Path, workspace_path: Path) -> str:
    """尽量把路径显示成相对 workspace 的 .cache/... 形式。"""
    try:
        return path.relative_to(workspace_path).as_posix()
    except ValueError:
        return str(path)


def _trim_preview_line(line: str) -> str:
    """避免单行超长导致 preview 仍然过胖。"""
    if len(line) <= _PREVIEW_LINE_CHAR_LIMIT:
        return line
    return f"{line[:_PREVIEW_LINE_CHAR_LIMIT]} ...[本行已截断]"


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
    """给 list_files / grep_files 生成轻量语义去重标识。"""
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
