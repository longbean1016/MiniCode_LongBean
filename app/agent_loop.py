

from app.prompt import build_system_prompt
from app.tooling import ToolRegistry
from app.types import AgentStep, ChatMessage, ModelAdapter


def run_agent_once(
        user_input:str,
        model:ModelAdapter,
        tool_registry:ToolRegistry,
        history:list[ChatMessage]|None=None
)-> tuple[AgentStep,list[ChatMessage]]: # type: ignore
    """
    执行一次最小 agent 主流程。

    第一版只做：
    1. 构建 system prompt
    2. 拼接历史消息
    3. 调用模型
    4. 返回本次结果和更新后的消息历史
    """

     # 没有历史消息时，先初始化为空列表
    if history is None:
        history = []
    
    # 根据当前工具注册表生成系统提示词
    system_prompt = build_system_prompt(tool_registry)

    # 先放入system消息，告诉模型当前的角色和规则
    messages:list[ChatMessage] = [
        {
            "role":"system",
            "content": system_prompt,
        }
    ]

    # 把历史消息的每一条ChatMessage追加进去，保持上下文连续
    messages.extend(history)

    # 再追加当前用户的输入
    messages.append(
        {
            "role":"user",
            "content": user_input,
        }
    )
    # 调用模型，获取本轮执行结果
    step=model.next(messages)


    # 组装新的历史消息，供下一轮继续使用
    new_history=history+[
        {
            "role":"user",
            "content": user_input,
        },
        {
            "role":"assistant",
            "content": step.content,
        },
    ]
    return step,new_history

    