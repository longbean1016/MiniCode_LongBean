

from app.agent_loop import run_agent_once
from app.config import load_config
from app.model_registry import OpenAIModelAdapter
from app.tools import build_tool_registry
from app.types import ChatMessage
 


def main()-> None:
    """
    程序入口。

    第一版负责：
    1. 加载配置
    2. 初始化模型
    3. 初始化工具注册表
    4. 启动命令行对话循环
    """
    # 加载项目运行配置
    config=load_config()

    # 初始化模型适配器
    model=OpenAIModelAdapter(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model
    ) # type: ignore

    # 初始化默认工具注册表
    tool_registry=build_tool_registry()

    # 保存对话历史，便于多轮对话
    history:list[ChatMessage]=[]

    print("LongBean MiniCode Agent 已启动，输入 quit 或 exit 退出。")

    while True:
        # 读取用户输入
        user_input = input("You> ").strip()

        # 空输入直接跳过，避免无意义请求
        if not user_input:
            continue

        # 输入 quit 或 exit 时退出程序
        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        # 执行一轮agent主流程
        step,history=run_agent_once(
            user_input=user_input,
            model=model,
            tool_registry=tool_registry,
            history=history
        )

        # 打印模型返回内容
        print(f"Agent> {step.content}")

if __name__=="__main__":
    main()