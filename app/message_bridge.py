
"""
    作用：
    一句话就是**“它是内部消息协议和模型消息协议之间的翻译层”**，
    让主循环只处理内部消息，不直接耦合厂商消息格式。
"""

import json
from typing import Any

from app.tooling import ToolDefinition
from app.types import AgentStep, ChatMessage, ToolCall


def build_openai_tools(tools:list[ToolDefinition])->list[dict[str,Any]]:
    """把内部工具定义转成 OpenAI/DeepSeek 的 tools 格式。"""
    result:list[dict[str,Any]]=[]

    for tool in tools:
        result.append(
            {
                "type":"function",
                "function":{
                    "name":tool.name,
                    "description":tool.description,
                    "parameters":tool.input_schema,
                },
            }
        )
    return result

def build_openai_messages(messages:list[ChatMessage])->list[dict[str,Any]]:
    """把内部消息转成模型接口消息。"""
    result:list[dict[str,Any]]=[]

    for msg in messages:
        role=msg.get("role") 
        content=msg.get("content")


        # system/user/assistant 直接透传
        if role in ("system","user","assistant"):
            result.append(
                {
                    "role":role,
                    "content":content
                }
            )
            continue

        # assistant_progress 用标记包起来，便于后续解析回内部 kind
        if role == "assistant_progress":
            result.append(
                {
                    "role": "assistant",
                    "content": f"<progress>\n{content}\n</progress>",
                }
            )
            continue

        # assistant_tool_call 映射为 OpenAI 的 assistant + tool_calls
        if role=="assistant_tool_call":
            result.append(
                {
                    "role":"assistant",
                    "content":None,
                    "tool_calls":[
                        {
                            "id": msg["tool_use_id"], # type: ignore
                            "type": "function",
                            "function": {
                                "name": msg["tool_name"], # type: ignore
                                "arguments": json.dumps(
                                    msg.get("input", {}),
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            )
            continue
        # tool_result 映射为 tool 角色，并保留 tool_call_id 对应关系
        if role == "tool_result":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg["tool_use_id"], # type: ignore
                    "content": content,
                }
            )
            continue

    return result

def parse_openai_response_message(message: Any) -> AgentStep:
    """把模型返回转回内部 AgentStep。"""
    tool_calls_raw = getattr(message, "tool_calls", None) or []

    # 模型返回了工具调用：组装成内部 tool_calls
    if tool_calls_raw:
        calls: list[ToolCall] = []
        for call in tool_calls_raw:
            tool_name = call.function.name
            tool_args = call.function.arguments or "{}"

            # 工具参数 JSON 解析失败时给空字典兜底，避免中断主循环
            try:
                parsed_input = json.loads(tool_args)
            except json.JSONDecodeError:
                parsed_input = {}

            calls.append(
                {
                    "id": call.id,
                    "tool_name": tool_name,
                    "input": parsed_input,
                }
            )

        return AgentStep(type="tool_calls", calls=calls)

    # 模型返回了文本回答：解析可能的 progress/final 标记
    content = getattr(message, "content", "") or ""
    content, kind = _unwrap_content(content)

    return AgentStep(
        type="assistant",
        content=content,
        kind=kind or "final", # type: ignore
    )


def _unwrap_content(content: str) -> tuple[str, str | None]:
    """把进度/最终标记从文本里拆出来。"""
    trimmed = content.strip()

    # 识别 progress 包裹
    if trimmed.startswith("<progress>") and trimmed.endswith("</progress>"):
        inner = trimmed[len("<progress>") : -len("</progress>")].strip()
        return inner, "progress"

    # 识别 final 包裹
    if trimmed.startswith("<final>") and trimmed.endswith("</final>"):
        inner = trimmed[len("<final>") : -len("</final>")].strip()
        return inner, "final"

    # 没有标记就按普通文本处理
    return trimmed, None
