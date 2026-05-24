
 

import json
from typing import Any

from openai import OpenAI
 

from app.tooling import ToolRegistry
from app.types import AgentStep, ChatMessage, ToolCall


class OpenAIModelAdapter:
    """
    模型适配器：
    负责把消息发送给模型，并把模型返回结果整理成统一的 AgentStep。
    """

    def __init__(self,api_key:str,base_url:str,model_name:str,tool_registry:ToolRegistry) -> None:
        # 创建openai客户端实例
        # 如果你后面把 base_url 换成 DeepSeek 的兼容地址，这里也可以继续复用。
        self.client=OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # 保存当前使用的模型名称
        self.model_name=model_name

        # 保存工具注册表，后面需要把工具描述传给模型
        self.tool_registry=tool_registry



    def _build_openai_tools(self)->list[dict[str, Any]]:
        """
        把内部工具定义转换成 OpenAI / DeepSeek 兼容的 tools 格式。
        """
        openai_tools: list[dict[str, Any]] = []
        for tool in self.tool_registry.list_tools():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name, # 工具名称
                    "description": tool.description, # 工具说明
                    "parameters": tool.input_schema # 工具参数结构，符合 OpenAI function call 的规范
                }
            })
        return openai_tools

    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:

        """
        把项目内部消息格式转换成 OpenAI 接口需要的 messages 格式。
        第一版先只保留最核心的 role 和 content。
        """

        result: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # 只转换模型当前能直接理解的三类基础消息
            if role in ("system", "user", "assistant"):
                result.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

            # 工具结果消息，第一版先转成 assistant 可读的普通文本
            elif role == "tool_result":
                result.append(
                    {
                        "role": "assistant",
                        "content": f"[工具调用结果] {content}",
                    }
                )

        # 循环结束后再统一返回，确保所有消息都被转换进去
        return result
        
    def next(
        self,
        messages: list[ChatMessage],
        on_stream_chunk=None,
        store=None
    )-> AgentStep: # type: ignore
        
        """
        调用模型并返回统一的 AgentStep。
        当前版本支持两种结果：
        1. 普通文本回答
        2. 模型发起工具调用
        """

        # 把内部消息格式转换成接口需要的格式
        openai_messages = self._to_openai_messages(messages)
        
        # 构造当前可用工具列表，传给模型做 function call
        openai_tools = self._build_openai_tools()

        # 调用聊天接口
        response=self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages, # type: ignore
            tools=openai_tools, # 把工具描述交给模型 # type: ignore
        )

        # 取出模型返回的文本内容
        message=response.choices[0].message

        # 如果模型返回了工具调用，就先转为统一的tool_calls结构
        if message.tool_calls: # type: ignore
            calls: list[ToolCall] = []
            for call in message.tool_calls: # type: ignore
                tool_name=call.function.name # type: ignore
                tool_args=call.function.arguments or "{}" # type: ignore

                # 把模型返回的 JSON 字符串参数转成字典
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
            return AgentStep(
                type="tool_calls",
                calls=calls,
            )
        # 否则就是普通文本回答，直接返回
        content=message.content or "" # type: ignore
        return AgentStep(
            type="assistant",
            content=content,
            kind="final",
        )