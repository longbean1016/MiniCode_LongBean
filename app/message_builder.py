

from dataclasses import dataclass, field
 

from app.types import ChatMessage


@dataclass(slots=True)
class MessageBuilder:

    """统一构造和追加对话消息。"""

    history:list[ChatMessage]=field(default_factory=list)

    def add_user(self,content:str)->None:
        """追加用户消息。"""
        self.history.append(
            {
                "role":"user",
                "content":content
            }
        )

    def add_assistant(self,content:str)->None:
        """追加模型普通回复。"""
        self.history.append(
            {
                "role":"assistant",
                "content":content
            }
        )

    def add_progress(self,content:str)->None:
        """追加模型中间进度。"""
        self.history.append({"role": "assistant_progress", "content": content})

    
    def add_tool_call(self,tool_use_id:str,tool_name:str,input_data:object)->None:
        """追加模型发起的工具调用。"""
        self.history.append(
            {
                "role": "assistant_tool_call",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "input": input_data,
            }
        )

    def add_tool_result(
        self,
        tool_use_id:str,
        tool_name:str,
        content:str,
        is_error:bool=False,
    )->None:
        """追加工具执行结果。"""
        self.history.append(
            {
                "role": "tool_result",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "content": content,
                "is_error": is_error,
            }
        )

    def build(self)->list[ChatMessage]:
        """返回当前完整历史。"""
        return list(self.history)
    
    def extend(self, messages: list[ChatMessage]) -> None:
        """批量追加已有消息。"""
        self.history.extend(messages)
