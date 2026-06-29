from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.tooling import ToolDefinition
from app.memory.memory_store import MemoryStore
from app.types import ToolContext, ToolResult


@dataclass(slots=True)
class MemoryToolInput:
    action: str
    target: str
    content: str = ""
    old_text: str = ""


_MEMORY_STORE: MemoryStore | None = None


def configure_memory_tool(memory_store: MemoryStore) -> None:
    global _MEMORY_STORE
    _MEMORY_STORE = memory_store


def get_memory_store() -> MemoryStore:
    if _MEMORY_STORE is None:
        raise RuntimeError("memory tool 尚未配置 memory_store。")
    return _MEMORY_STORE


def _validate(input_data: Any) -> MemoryToolInput:
    if not isinstance(input_data, dict):
        raise ValueError("memory 输入必须是字典。")

    action = str(input_data.get("action", "")).strip().lower()
    target = str(input_data.get("target", "")).strip().lower()
    content = str(input_data.get("content", "")).rstrip()
    old_text = str(input_data.get("old_text", "")).strip()

    if action not in {"add", "remove", "view"}:
        raise ValueError("action 必须是 add、remove 或 view。")
    if target not in {"memory", "user"}:
        raise ValueError("target 必须是 memory 或 user。")
    if action == "add" and not content:
        raise ValueError("add 操作必须提供 content。")
    if action == "remove" and not old_text:
        raise ValueError("remove 操作必须提供 old_text。")

    return MemoryToolInput(
        action=action,
        target=target,
        content=content,
        old_text=old_text,
    )


def _run(validated_input: MemoryToolInput, context: ToolContext) -> ToolResult:
    _ = context
    memory_store = get_memory_store()

    try:
        if validated_input.action == "view":
            output = memory_store.view(validated_input.target)
            return ToolResult(
                ok=True,
                output=output or "当前没有记忆内容。",
                meta={"action": "view", "target": validated_input.target},
            )

        if validated_input.action == "add":
            result = memory_store.add(
                target=validated_input.target,
                content=validated_input.content,
                bypass_approval=not memory_store.write_approval,
            )
        else:
            result = memory_store.remove(
                target=validated_input.target,
                old_text=validated_input.old_text,
            )
    except Exception as error:
        return ToolResult(
            ok=False,
            output=f"memory 工具执行失败：{error}",
            error="MEMORY_TOOL_FAILED",
            meta={
                "action": validated_input.action,
                "target": validated_input.target,
            },
        )

    success = bool(result.get("success", False))
    message = str(result.get("message") or result.get("error") or "").strip()
    if validated_input.action == "add" and success:
        message = message or f"已写入 {validated_input.target} 记忆。"
    elif validated_input.action == "remove" and success:
        message = message or f"已删除 {validated_input.target} 记忆。"

    return ToolResult(
        ok=success,
        output=message or "memory 操作已完成。",
        error=None if success else "MEMORY_TOOL_REJECTED",
        meta={
            "action": validated_input.action,
            "target": validated_input.target,
            "content": validated_input.content,
            "old_text": validated_input.old_text,
        },
    )


memory_tool = ToolDefinition(
    name="memory",
    description=(
        "维护持久记忆。"
        "add 写入 Markdown 列表项到 MEMORY.md 或 USER.md；"
        "remove 按片段删除一条记忆；"
        "view 查看当前冻结快照。"
    ),
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "remove", "view"],
                "description": "要执行的记忆操作。",
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "写入项目记忆还是用户记忆。",
            },
            "content": {
                "type": "string",
                "description": "add 时必填，必须以 '- ' 开头的 Markdown 列表项。",
            },
            "old_text": {
                "type": "string",
                "description": "remove 时必填，要删除的记忆片段。",
            },
        },
        "required": ["action", "target"],
    },
)
