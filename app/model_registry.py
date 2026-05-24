
 

import json
from typing import Any

from openai import OpenAI
 

from app.message_bridge import build_openai_messages, build_openai_tools, parse_openai_response_message
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
        openai_messages = build_openai_messages(messages)
        
        # 构造当前可用工具列表，传给模型做 function call
        openai_tools = build_openai_tools(self.tool_registry.list_tools()) 

        # 调用聊天接口
        response=self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages, # type: ignore
            tools=openai_tools, # 把工具描述交给模型 # type: ignore
        )

        return parse_openai_response_message(response.choices[0].message)