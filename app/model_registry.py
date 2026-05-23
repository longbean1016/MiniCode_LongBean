
 

from openai import OpenAI

from app.types import AgentStep, ChatMessage


class OpenAIModelAdapter:
    """
    模型适配器：
    负责把消息发送给模型，并把模型返回结果整理成统一的 AgentStep。
    """

    def __init__(self,api_key:str,base_url:str,model_name:str) -> None:
        # 创建openai客户端实例
        # 如果你后面把 base_url 换成 DeepSeek 的兼容地址，这里也可以继续复用。
        self.client=OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # 保存当前使用的模型名称
        self.model_name=model_name

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
        调用模型，返回统一的 AgentStep。
        第一版先只支持普通文本回答，不处理工具调用。
        """

        # 把内部消息格式转换成接口需要的格式
        openai_messages = self._to_openai_messages(messages)
        
        # 调用聊天接口
        response=self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages, # type: ignore
        )

        # 取出模型返回的文本内容
        content=response.choices[0].message.content or ""

        # 第一版统一按普通 assistant 回复处理
        return AgentStep(
            type="assistant",
            content=content,
            kind="final"
        )
