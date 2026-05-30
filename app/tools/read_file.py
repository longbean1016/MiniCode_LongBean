from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from app.permissions import PermissionManager
from app.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# 参考 minicode 的思路：正文类读取工具天然支持分段，
# 避免“一次性把整份大文件塞进上下文”。
DEFAULT_READ_LIMIT = 8_000
MAX_READ_LIMIT = 20_000
_BLOCKED_STATE_DIRS = {".sessions", "sessions", ".context_state", "context_state"}
_BLOCKED_CACHE_FILE_PATTERN = re.compile(r"^(?:\.cache|cache)[\\/](?:\.?tool_result_).+", re.IGNORECASE)


def _validate(input_data: Any) -> dict[str, int | str]:
    """校验 read_file 输入，并支持 offset/limit 分段读取。"""
    if not isinstance(input_data, dict):
        raise ValueError("read_file 输入必须是字典，并且包含 path 字段。")

    path = input_data.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串。")

    raw_offset = input_data.get("offset", 0)
    raw_limit = input_data.get("limit", DEFAULT_READ_LIMIT)

    try:
        offset = int(raw_offset)
    except (TypeError, ValueError) as error:
        raise ValueError("offset 必须是整数。") from error

    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as error:
        raise ValueError("limit 必须是整数。") from error

    if offset < 0:
        raise ValueError("offset 必须大于等于 0。")
    if limit < 1 or limit > MAX_READ_LIMIT:
        raise ValueError(f"limit 必须在 1 到 {MAX_READ_LIMIT} 之间。")

    return {
        "path": path.strip(),
        "offset": offset,
        "limit": limit,
    }


def _run(validated_input: dict[str, int | str], context: ToolContext) -> ToolResult:
    """读取文件的一段文本，并明确告诉上层是否还有后续内容。"""
    permission_manager = PermissionManager(context.cwd)

    raw_path = str(validated_input["path"])
    target_path = permission_manager.ensure_path_access(raw_path)
    relative_path = _to_workspace_relative_path(target_path, context.cwd)

    # 先拦掉内部状态与大工具结果归档文件，避免模型把这些“调试产物”当成主分析材料反复回读。
    blocked_reason = _match_blocked_internal_path(relative_path)
    if blocked_reason is not None:
        return ToolResult(
            ok=False,
            output=(
                f"默认不允许读取内部上下文文件：{raw_path}\n"
                f"原因：{blocked_reason}\n"
                "请优先改用正常源码、配置或工具摘要继续分析。"
            ),
            error="READ_POLICY_BLOCKED",
            meta={
                "path": raw_path,
                "normalized_path": relative_path,
                "blocked_reason": blocked_reason,
            },
        )

    if not target_path.exists():
        return ToolResult(
            ok=False,
            output=f"文件不存在：{raw_path}",
        )

    if not target_path.is_file():
        return ToolResult(
            ok=False,
            output=f"目标不是文件：{raw_path}",
        )

    # 优先按 utf-8 读取；如果文件是常见的本地编码文本，再回退一次。
    # 这样日志和历史导出这类 gbk 文本也能进入上下文链路。
    content = _read_text_with_fallback(target_path)

    offset = int(validated_input["offset"])
    limit = int(validated_input["limit"])
    read_signature = _build_read_signature(relative_path, offset, limit)

    # 同一轮里同一路径、同一区间再次读取，通常意味着模型开始打转。
    # 这里直接短路，逼它换工具或换区间，避免无效上下文堆积。
    if read_signature in context.read_file_signatures:
        return ToolResult(
            ok=False,
            output=(
                f"同一轮里已经读取过相同区间：{raw_path}\n"
                f"offset={offset}, limit={limit}\n"
                "请改读新的 offset/limit 区间，或改用 grep_files / file_overview 等工具。"
            ),
            error="READ_REPEAT_BLOCKED",
            meta={
                "path": raw_path,
                "normalized_path": relative_path,
                "offset": offset,
                "limit": limit,
            },
        )

    context.read_file_signatures.add(read_signature)
    end = min(len(content), offset + limit)
    chunk = content[offset:end]
    truncated = end < len(content)

    header_lines = [
        f"FILE: {raw_path}",
        f"OFFSET: {offset}",
        f"END: {end}",
        f"TOTAL_CHARS: {len(content)}",
        (
            f"TRUNCATED: yes - call read_file again with offset {end}"
            if truncated
            else "TRUNCATED: no"
        ),
        "",
    ]
    output = "\n".join(header_lines) + chunk

    return ToolResult(
        ok=True,
        output=output,
        meta={
            "path": raw_path,
            "offset": offset,
            "end": end,
            "total_chars": len(content),
            "truncated": truncated,
        },
    )


def _read_text_with_fallback(target_path: Path) -> str:
    """读取文本文件时做常见编码回退，尽量避免编码问题导致直接失败。"""
    raw_bytes = target_path.read_bytes()
    for encoding in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    # 最后一层兜底：忽略非法字节，至少保留可读片段。
    return raw_bytes.decode("utf-8", errors="ignore")


def _to_workspace_relative_path(target_path: Path, cwd: str) -> str:
    """把绝对路径稳定转成相对工作区路径，便于做策略判断和去重签名。"""
    workspace_root = Path(cwd).resolve()
    try:
        relative_path = target_path.resolve().relative_to(workspace_root)
    except ValueError:
        return target_path.resolve().as_posix()
    return relative_path.as_posix()


def _match_blocked_internal_path(relative_path: str) -> str | None:
    """识别默认不应该被 read_file 回读的内部上下文文件。"""
    normalized = relative_path.replace("\\", "/").lstrip("./")
    if not normalized:
        return None

    first_part = normalized.split("/", 1)[0].lower()
    if first_part in _BLOCKED_STATE_DIRS:
        return "这是会话状态/历史文件，默认不作为当前分析上下文直接回读。"

    if _BLOCKED_CACHE_FILE_PATTERN.match(normalized):
        return "这是大工具结果归档文件，默认不作为当前分析上下文直接回读。"

    return None


def _build_read_signature(relative_path: str, offset: int, limit: int) -> str:
    """为同轮重复读取熔断生成稳定签名。"""
    normalized = relative_path.replace("\\", "/")
    return f"{normalized}::{offset}::{limit}"


read_file_tool = ToolDefinition(
    name="read_file",
    description="读取指定文件内容，支持 offset/limit 分段读取。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要读取的文件路径，必须位于工作目录内。",
            },
            "offset": {
                "type": "integer",
                "description": "从文件第几个字符开始读取，默认 0。",
            },
            "limit": {
                "type": "integer",
                "description": f"本次最多读取多少字符，默认 {DEFAULT_READ_LIMIT}，最大 {MAX_READ_LIMIT}。",
            },
        },
        "required": ["path"],
    },
)
