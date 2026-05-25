

import argparse

from app.agent_loop import run_agent_once
from app.config import load_config
from app.model_registry import OpenAIModelAdapter
from app.session import SessionData, create_new_session, get_latest_session, load_session, save_session
from app.tools import build_tool_registry
from app.types import ChatMessage, ToolContext
 


def _build_arg_parser()->argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="LongBean MiniCode Agent",
    )

    # 显式指定某个 session_id 恢复会话
    parser.add_argument(
        "--session",
        type=str,
        default="",
        help="指定要恢复的会话 ID",
    )

    # 指定恢复策略，目前最小版只支持 latest
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="恢复最近一次会话，例如: --resume latest",
    )

    return parser


def _load_or_create_session(workspace: str,session_id: str,resume: str)->SessionData:
    """按参数决定是恢复旧会话还是创建新会话。"""
    # 优先级最高：显式指定 session_id
    if session_id:
        session = load_session(workspace, session_id)
        if session is None:
            raise FileNotFoundError(f"未找到会话: {session_id}")
        print(f"已恢复指定会话: {session.session_id}")
        return session
     # 其次：恢复当前工作区最近一次会话
    if resume == "latest":
        session = get_latest_session(workspace)
        if session is not None:
            print(f"已恢复最近会话: {session.session_id}")
            return session

        # 没有历史会话时自动新建
        session = create_new_session(workspace)
        print(f"未找到最近会话，已新建会话: {session.session_id}")
        return session

    # 默认直接创建新会话
    session = create_new_session(workspace)
    print(f"已新建会话: {session.session_id}")
    return session




def main()-> None:
    """
    程序入口。

    第一版负责：
    1. 加载配置
    2. 初始化模型
    3. 初始化工具注册表
    4. 启动命令行对话循环
    """

    # 解析命令行参数
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 加载项目运行配置
    config=load_config()

    # 初始化默认工具注册表
    tool_registry=build_tool_registry()

    # 初始化模型适配器
    model=OpenAIModelAdapter(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model,
        tool_registry=tool_registry # 初始化工具注册表
    ) # type: ignore

    # 创建工具执行上下文，规定工具默认工作目录
    tool_context = ToolContext(
        cwd=config.workspace_root,
    )

    # 创建或恢复会话
    session = _load_or_create_session(
        workspace=config.workspace_root,
        session_id=args.session.strip(),
        resume=args.resume.strip().lower(),
    )

    # 用已恢复的会话历史初始化当前运行时历史
    history:list[ChatMessage]=list(session.messages)

    print("LongBean MiniCode Agent 已启动，输入 quit 或 exit 退出。")

    while True:
        # 读取用户输入
        user_input = input("You> ").strip()

        # 空输入直接跳过，避免无意义请求
        if not user_input:
            continue

        # 退出前先保存当前会话
        if user_input.lower() in {"quit", "exit"}:
            session.replace_messages(history)
            save_session(session)
            print("Bye!")
            break

        # 执行一轮agent主流程
        step,history=run_agent_once(
            user_input=user_input,
            model=model,
            tool_registry=tool_registry,
            tool_context=tool_context,
            history=history
        )

        # 每轮结束后把最新历史写回会话并保存
        session.replace_messages(history)
        save_session(session)

        # 打印模型返回内容
        print(f"Agent> {step.content}")

if __name__=="__main__":
    main()
