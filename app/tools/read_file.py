from __future__ import annotations

from pathlib import Path
from typing import Any

from app.permissions import PermissionManager
from app.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# 参考 minicode 的思路：正文类读取工具天然支持分段，
# 避免“一次性把整份大文件塞进上下文”。
DEFAULT_READ_LIMIT = 8_000
MAX_READ_LIMIT = 20_000


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
