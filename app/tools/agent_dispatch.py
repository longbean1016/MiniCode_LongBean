"""agent_dispatch 子代理调度工具，支持 spawn 子任务和获取结果。
   参考 Claude Code AgentTool 语义实现，第一版做同步子任务调度。"""

from typing import Any

from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, Any]:
    """校验 agent_dispatch 输入：description + prompt 必填。"""
    if not isinstance(input_data, dict):
        raise ValueError("agent_dispatch 输入必须是字典，包含 description 和 prompt 字段。")

    description = input_data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description 必须是非空字符串（3-5 词简短描述）。")

    prompt = input_data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt 必须是非空字符串（子代理要执行的任务描述）。")

    # 子代理类型（可选）：默认使用通用代理
    subagent_type = input_data.get("subagent_type")
    if subagent_type is not None and not isinstance(subagent_type, str):
        raise ValueError("subagent_type 必须是字符串。")

    return {
        "description": description.strip(),
        "prompt": prompt.strip(),
        "subagent_type": subagent_type.strip() if subagent_type else "general-purpose",
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """第一版：返回任务描述，将实际调度交给上层 agent loop 处理。
       后续版本会实现完整的子代理生命周期管理。"""
    description = validated_input["description"]
    prompt = validated_input["prompt"]
    subagent_type = validated_input["subagent_type"]

    # 第一版只做任务转发，不自己启动子代理。
    # 上层 agent loop 看到 agent_dispatch 调用后，
    # 重新进入推理循环，让子代理工具可用时模型再做具体调度。
    return ToolResult(
        ok=True,
        output=(
            f"子代理任务已注册：{description}\n"
            f"类型：{subagent_type}\n"
            f"任务：{prompt}"
        ),
        meta={
            "description": description,
            "prompt": prompt,
            "subagent_type": subagent_type,
            "status": "dispatched",
        },
    )


# 按 MiniCode 现有 ToolDefinition 模式注册工具
agent_dispatch_tool = ToolDefinition(
    name="agent_dispatch",
    description=(
        "派发子代理执行复杂多步骤任务。"
        "当任务需要独立上下文或长时间运行而不阻塞主对话时使用。"
    ),
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "任务简短描述（3-5 词）。",
            },
            "prompt": {
                "type": "string",
                "description": "子代理要执行的具体任务指令。",
            },
            "subagent_type": {
                "type": "string",
                "description": "子代理类型。默认使用通用代理。可用类型：general-purpose、explore、plan。",
            },
        },
        "required": ["description", "prompt"],
    },
)
