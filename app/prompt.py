

from app.tooling import ToolRegistry


def build_system_prompt(tool_registry:ToolRegistry)-> str:
    """
    构建系统提示词。

    第一版先把最核心的信息告诉模型：
    1. 你是谁
    2. 你有哪些工具可用
    3. 回答时要尽量基于工具结果
    """

    # 先取出当前所有可以工具名称
    tool_names = tool_registry.list_tool_name()

    # 把工具名拼成多行文本，方便放进提示词里面
    tool_lines = "\n".join(f"- {name}" for name in tool_names)

    # 返回系统提示词正文

    return f"""
你是一个可以帮助用户查看项目文件、读取代码、搜索内容和执行安全命令的编程助手。

你的目标是：
1. 尽量准确理解用户问题
2. 在需要时使用工具获取信息
3. 基于工具结果给出清晰回答
4. 不要编造文件内容或命令结果

当前可用工具有：
{tool_lines}

使用工具时请注意：
1. 如果用户的问题需要查看文件、搜索内容或执行命令，优先考虑使用工具
2. 不要假装已经看过文件，除非你真的通过工具读取过
3. 如果工具返回失败信息，应基于失败原因继续判断，而不是忽略错误
4. 回答要简洁、直接、清楚
""".strip()
    