
"""
    作用：
    一句话就是**“它是内部消息协议和模型消息协议之间的翻译层”**，
    让主循环只处理内部消息，不直接耦合厂商消息格式。
"""

import json
from typing import Any

from app.context.message_safety import normalize_tool_call_pairs
from app.agent.tooling import ToolDefinition
from app.types import AgentStep, ChatMessage, ToolCall


def build_openai_tools(tools:list[ToolDefinition])->list[dict[str,Any]]:
    """把内部工具定义转成 OpenAI/DeepSeek 的 tools 格式。"""
    result:list[dict[str,Any]]=[]

    for tool in tools:
        # 这里只做协议层映射，不掺入任何调度策略。
        # “该不该调用某个工具”是模型与主循环共同决定的，
        # bridge 只负责把注册表里的定义准确送到模型侧。
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
    """把内部消息转成模型接口消息。

       合并连续的 assistant_tool_call 到同一个 assistant 消息，
       保证 OpenAI 协议要求的所有 tool_call_id 都有对应 tool_result。
    """
    result: list[dict[str,Any]] = []
    normalized = normalize_tool_call_pairs(messages)
    idx = 0
    while idx < len(normalized):
        msg = normalized[idx]
        role = msg.get('role')
        content = msg.get('content')

        if role in ('system', 'user', 'assistant'):
            result.append({'role': role, 'content': content})
            idx += 1
            continue

        if role == 'assistant_progress':
            result.append({
                'role': 'assistant',
                'content': f'<progress>\n{content}\n</progress>',
            })
            idx += 1
            continue

        # 合并连续的 assistant_tool_call 到一个 assistant 消息
        if role == 'assistant_tool_call':
            tool_calls = []
            while idx < len(normalized) and normalized[idx].get('role') == 'assistant_tool_call':
                tc = normalized[idx]
                tool_calls.append({
                    'id': tc.get('tool_use_id', ''),
                    'type': 'function',
                    'function': {
                        'name': tc.get('tool_name', ''),
                        'arguments': json.dumps(tc.get('input', {}), ensure_ascii=False),
                    }
                })
                idx += 1
            result.append({'role': 'assistant', 'content': None, 'tool_calls': tool_calls})
            continue

        if role == 'tool_result':
            result.append({
                'role': 'tool',
                'tool_call_id': msg.get('tool_use_id', ''),
                'content': content,
            })
            idx += 1
            continue

        idx += 1

    return result

def parse_openai_response_message(message: Any) -> AgentStep:
    """把模型返回转回内部 AgentStep。"""
    tool_calls_raw = getattr(message, "tool_calls", None) or []

    # 先判 tool_calls，再判纯文本。
    # 因为一次响应只要带了 tool_calls，就意味着主循环下一步必须进入工具执行分支，
    # 不能再把它当成普通 assistant 文本继续向后走。
    if tool_calls_raw:
        calls: list[ToolCall] = []
        for call in tool_calls_raw:
            tool_name = call.function.name
            tool_args = call.function.arguments or "{}"

            # 参数 JSON 偶尔会被模型生成为非法字符串。
            # 这里兜底为空字典，是为了让主循环继续运行并把失败交给具体工具层处理，
            # 而不是在协议桥接层直接把整轮对话打崩。
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

    # 这里约定了一层极轻的“文本内协议”：
    # 当模型侧只能返回 assistant 文本时，用包裹标记把内部 kind 带回来。
    if trimmed.startswith("<progress>") and trimmed.endswith("</progress>"):
        inner = trimmed[len("<progress>") : -len("</progress>")].strip()
        return inner, "progress"

    # 识别 final 包裹
    if trimmed.startswith("<final>") and trimmed.endswith("</final>"):
        inner = trimmed[len("<final>") : -len("</final>")].strip()
        return inner, "final"

    # 没有标记就按普通文本处理
    return trimmed, None
