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
        "写入跨会话持久记忆，每次对话自动注入，保持条目精简高价值。\n\n"
        "操作方式：通过 add/remove/view 单次操作维护记忆。"
        "add 写入 Markdown 列表项到 MEMORY.md 或 USER.md；"
        "remove 按关键词片段删除匹配的记忆条目；"
        "view 查看当前已保存的全部内容。"
        "每次调用只做一件事，如果需要同时增删，分两次调用。\n\n"
        "触发时机：当用户表达偏好、纠正你说错的信息、说出个人身份"
        "或习惯、说明工作环境约定或流程时，主动调用此工具保存。"
        "优先级：用户偏好与纠正 > 环境事实 > 操作流程。"
        "最好的记忆是让用户不用重复说过的话。\n\n"
        "容量处理：add 超出字符限制时会提示当前已有条目，"
        "随后可通过 remove 清理过时条目释放空间后再 add。\n\n"
        "存储目标：target='user' 存用户身份/名称/偏好/风格；"
        "target='memory' 存项目规范/环境约定/工具技巧/经验教训。\n\n"
        "不要保存：琐碎或显而易见的信息、能轻易查到的内容、"
        "任务进度日志、已完成的临时状态。可复用的流程应写成 skill。"
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
