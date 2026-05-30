from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.types import ChatMessage

# 这里先只对 read_file 做真正的“路径 + 内容 hash”去重。
# list_files / grep_files 更像集合型结果，直接套同一规则容易误伤。
_READ_TOOLS = {"read_file"}
_COLLECTION_TOOLS = {"list_files", "grep_files"}
_PREVIEW_HEAD_LINES = 8
_PREVIEW_TAIL_LINES = 3
_PREVIEW_LINE_CHAR_LIMIT = 160
_READ_FILE_PATH_PATTERN = re.compile(r"^FILE:\s*(.+)$", re.MULTILINE)


@dataclass(slots=True)
class CompactionResult:
    """保存 recent window 压缩后的结果和统计信息。"""

    messages: list[ChatMessage]
    truncated_tool_results: int = 0
    cleared_old_tool_results: int = 0
    deduped_read_results: int = 0
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
) -> CompactionResult:
    """
    对 recent window 做轻量压缩。

    当前阶段仍然聚焦 tool_result，但已经具备三段治理：
    1. 超大 tool_result 落盘到 .cache/，并替换成 line-based preview
    2. 重复读取类结果做去重 stub
    3. 只保留最近若干条 tool_result，其余改成短占位
    """
    compacted = [dict(message) for message in messages]
    result = CompactionResult(messages=compacted)
    workspace_path = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    cache_dir = workspace_path / ".cache"

    # 这里优先使用工具原始输出：
    # 如果 ToolRegistry 已经做过 smart truncate，会把完整原文放进 meta.raw_output。
    original_tool_contents: dict[int, str] = {
        index: _get_tool_original_content(message)
        for index, message in enumerate(compacted)
        if message.get("role") == "tool_result"
    }

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

    # 读取类工具容易重复把同一段信息塞进上下文，因此只保留最后一份完整结果。
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

        compacted[index]["content"] = (
            "[读取结果已去重：文件内容未变化，保留本轮较新的同文件结果]\n"
            f"文件: {read_identity.display_path}"
        )
        compacted[index]["_deduped_read_result"] = True
        result.deduped_read_results += 1

    tool_result_indexes = [
        index
        for index, message in enumerate(compacted)
        if message.get("role") == "tool_result"
    ]
    keep_indexes = set(tool_result_indexes[-max_recent_tool_results:])

    # 即使已经生成 preview，过旧的 tool_result 也继续收缩，
    # 避免 recent window 长期堆着很多 preview。
    for index in tool_result_indexes:
        if index in keep_indexes:
            continue

        message = compacted[index]
        current_content = str(message.get("content", ""))
        if current_content.startswith("[旧工具结果已省略："):
            continue

        original_content = original_tool_contents[index]
        tool_name = str(message.get("tool_name", "")) or "unknown"
        omitted_message = f"[旧工具结果已省略：tool={tool_name} 原始长度={len(original_content)}]"
        compacted[index]["content"] = omitted_message
        compacted[index]["_old_tool_result_omitted"] = True
        result.cleared_old_tool_results += 1
        result.tokens_freed_estimate += max(0, len(current_content) - len(omitted_message)) // 4

    return result


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
        f"[工具结果已落盘，为节省上下文已截断，原始字符数={len(content)}]",
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
