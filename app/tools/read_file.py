"""读取文件工具，按行级 offset/limit 返回源码内容并附带覆盖范围信息。
   参考 Claude Code FileReadTool 语义实现。"""

from pathlib import Path
import re
from typing import Any

from app.agent.permissions import PathAccessStatus, PermissionManager
from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# 参考 Claude Code FileReadTool：行级读取，默认上限和最大上限
DEFAULT_READ_LIMIT = 200   # 默认返回行数
MAX_READ_LIMIT = 2000      # 最大返回行数
MAX_SIZE_BYTES = 1_024_000 # 最大文件大小（字节），超过则报错提示用 offset/limit
# 读取超大文件的字符截断保护
MAX_OUTPUT_CHARS = 40_000
# 禁止回读的内部目录（会话状态/缓存目录）
_BLOCKED_STATE_DIRS = {".sessions", "sessions", ".context_state", "context_state"}
_BLOCKED_CACHE_FILE_PATTERN = re.compile(r"^(?:\.cache|cache)[\\/](?:\.?tool_result_).+", re.IGNORECASE)


def _validate(input_data: Any) -> dict[str, Any]:
    """校验 read_file 输入，支持 file_path（新）+ 行级 offset/limit。

       兼容旧字段 path（当 file_path 不存在时从 path 回退）。
    """
    if not isinstance(input_data, dict):
        raise ValueError("read_file 输入必须是字典，包含 file_path 字段。")

    # ── 采用 Claude Code 风格主字段 file_path，兼容旧 path ──
    raw_path = input_data.get("file_path") or input_data.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("file_path 必须是非空字符串。")

    # ── offset: 行号（1-indexed，默认第1行）──
    raw_offset = input_data.get("offset", 1)
    try:
        offset = int(raw_offset)
    except (TypeError, ValueError) as exc:
        raise ValueError("offset 必须是整数。") from exc
    if offset < 1:
        raise ValueError("offset 必须 >= 1（行号从 1 开始）。")

    # ── limit: 最多返回行数 ──
    raw_limit = input_data.get("limit")
    if raw_limit is None:
        limit = DEFAULT_READ_LIMIT
    else:
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit 必须是整数。") from exc
    if limit < 1 or limit > MAX_READ_LIMIT:
        raise ValueError(f"limit 必须在 1 到 {MAX_READ_LIMIT} 之间。")

    return {
        "file_path": raw_path.strip(),
        "offset": offset,
        "limit": limit,
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """读取文件的指定行范围，并明确告知是否还有更多内容。"""
    permission_manager = PermissionManager(
        context.cwd,
        additional_workspaces={Path(p) for p in context.additional_workspaces},
        permanent_workspaces={Path(p) for p in context.permanent_workspaces},
    )

    raw_path = validated_input["file_path"]
    offset = validated_input["offset"]
    limit = validated_input["limit"]

    # ── 路径权限检查 ──
    check = permission_manager.check_path_access(raw_path)
    if check.status == PathAccessStatus.OUTSIDE_WORKSPACE:
        return ToolResult(
            ok=False,
            output=f"目标路径不在工作目录范围内：{raw_path}",
            error="WORKSPACE_ACCESS_REQUIRED",
            meta={
                "file_path": raw_path,
                "resolved_path": str(check.resolved_path) if check.resolved_path else raw_path,
                "action_key": f"workspace::{raw_path}",
                "reason": check.message,
            },
        )
    target_path = check.resolved_path
    relative_path = _to_workspace_relative_path(target_path, context.cwd)

    # ── 内部上下文文件阻断 ──
    blocked_reason = _match_blocked_internal_path(relative_path)
    if blocked_reason is not None:
        return ToolResult(
            ok=False,
            output=(
                f"默认不允许读取内部上下文文件：{raw_path}\n"
                f"原因：{blocked_reason}\n"
                "请优先改用正常源码、配置或业务文件继续分析。"
            ),
            error="READ_POLICY_BLOCKED",
            meta={
                "file_path": raw_path,
                "normalized_path": relative_path,
                "blocked_reason": blocked_reason,
            },
        )

    # ── 文件不存在时尝试提供相似路径建议（参考 Claude Code FileReadTool）──
    if not target_path.exists():
        suggestion = _suggest_similar_path(target_path, context.cwd)
        msg = f"文件不存在：{raw_path}\n当前工作目录：{context.cwd}"
        if suggestion:
            msg += f"\n你是指 {suggestion} 吗？"
        return ToolResult(ok=False, output=msg, meta={"file_path": raw_path})

    if not target_path.is_file():
        return ToolResult(ok=False, output=f"目标不是文件：{raw_path}",
                          meta={"file_path": raw_path})

    # ── 大小保护 ──
    file_size = target_path.stat().st_size
    if file_size > MAX_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return ToolResult(
            ok=False,
            output=f"文件过大（{size_mb:.1f} MB），超过单次读取上限。请使用 offset 和 limit 分段读取。",
            error="FILE_TOO_LARGE",
            meta={"file_path": raw_path, "size_bytes": file_size},
        )

    # ── 读取内容 ──
    content = _read_text_with_fallback(target_path)
    lines = content.splitlines()
    total_lines = len(lines)

    # offset 是 1-indexed 行号，转为 0-indexed
    start_idx = offset - 1
    end_idx = min(total_lines, start_idx + limit)

    if start_idx >= total_lines:
        return ToolResult(
            ok=False,
            output=(
                f"offset ({offset}) 超出文件总行数（{total_lines}）。"
                "请减小 offset 重试。"
            ),
            meta={"file_path": raw_path, "offset": offset, "total_lines": total_lines},
        )

    chunk_lines = lines[start_idx:end_idx]
    chunk = "\n".join(chunk_lines)
    truncated = end_idx < total_lines

    # ── 构建输出（行级标注，对齐 Claude Code FileReadTool）──
    header = (
        f"FILE: {raw_path}\n"
        f"LINES: {offset}-{end_idx} / {total_lines}\n"
        f"SIZE: {file_size} bytes\n"
    )
    if truncated:
        header += f"TRUNCATED: yes — 使用 offset={end_idx + 1} 读取下一段\n"
    else:
        header += "TRUNCATED: no\n"
    header += "\n"

    # 字符截断保护
    output = header + chunk
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n\n(输出过大已截断，共 {len(output)} 字符)"

    return ToolResult(
        ok=True,
        output=output,
        meta={
            "file_path": raw_path,
            "offset": offset,
            "end_line": end_idx,
            "total_lines": total_lines,
            "truncated": truncated,
            "size_bytes": file_size,
        },
    )


# ── 辅助函数 ──

def _read_text_with_fallback(target_path: Path) -> str:
    """读取文本文件时做常见编码回退（utf-8 → gbk → utf-8-sig），避免编码问题直接失败。"""
    raw_bytes = target_path.read_bytes()
    for encoding in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    # 最后一层兜底：忽略非法字节，至少保留可读片段
    return raw_bytes.decode("utf-8", errors="ignore")


def _to_workspace_relative_path(target_path: Path, cwd: str) -> str:
    """把绝对路径稳定转成相对工作区路径，用于策略判断和去重。"""
    workspace_root = Path(cwd).resolve()
    try:
        return target_path.resolve().relative_to(workspace_root).as_posix()
    except ValueError:
        return target_path.resolve().as_posix()


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


def _suggest_similar_path(target_path: Path, cwd: str) -> str | None:
    """文件不存在时，尝试推荐工作区下名称最接近的候选文件（参考 Claude Code FileReadTool 行为）。"""
    filename = target_path.name
    if not filename:
        return None
    # 在工作区下搜索同名或名称相近的文件
    root = Path(cwd).resolve()
    candidates: list[tuple[float, str]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name == filename:
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = p.as_posix()
            return rel  # 找到同名文件直接返回
        # 简单的编辑距离相近判断
        if _name_similar(p.name, filename):
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = p.as_posix()
            candidates.append((0.0, rel))
    if candidates:
        return candidates[0][1]
    return None


def _name_similar(name1: str, name2: str) -> bool:
    """简单判断两个文件名是否相似（共享前缀字符 >= 3）。"""
    common = 0
    for a, b in zip(name1.lower(), name2.lower()):
        if a == b:
            common += 1
        else:
            break
    return common >= 3


# ── 注册工具 ──
read_file_tool = ToolDefinition(
    name="read_file",
    description="读取文件内容，支持行级 offset/limit 分段读取。",
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要读取的文件绝对路径。",
            },
            "offset": {
                "type": "integer",
                "description": "从第几行开始读取（1-indexed），默认 1。仅当文件过大需要分段时提供。",
            },
            "limit": {
                "type": "integer",
                "description": f"最多返回多少行，默认 {DEFAULT_READ_LIMIT}，最大 {MAX_READ_LIMIT}。仅当文件过大需要分段时提供。",
            },
            "path": {
                "type": "string",
                "description": "(已弃用) 请使用 file_path。",
            },
        },
        "required": ["file_path"],
    },
)
