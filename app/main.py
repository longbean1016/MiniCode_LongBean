from __future__ import annotations

import argparse

from app.agent_loop import run_agent_once
from app.config import load_config
from app.model_registry import OpenAIModelAdapter
from app.session import (
    SessionData,
    create_new_session,
    format_session_list,
    get_latest_session,
    list_sessions,
    load_session,
    save_session,
)
from app.tools import build_tool_registry
from app.types import ChatMessage, ToolContext


def _build_arg_parser() -> argparse.ArgumentParser:
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

    # 只允许固定恢复模式，避免误传其他值时误建会话
    parser.add_argument(
        "--resume",
        type=str,
        choices=["latest", "list"],
        default=None,
        help="恢复模式，只支持 latest 或 list，例如 --resume latest",
    )

    return parser


def _load_or_create_session(workspace: str, session_id: str, resume: str) -> SessionData:
    """按参数决定是恢复旧会话还是创建新会话。"""
    # 优先恢复指定会话
    if session_id:
        session = load_session(workspace, session_id)
        if session is None:
            raise FileNotFoundError(f"未找到会话: {session_id}")
        print(f"已恢复指定会话: {session.session_id}")
        return session

    # 其次恢复当前工作区最近一次会话
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


def main() -> None:
    """程序入口：加载配置、恢复会话、启动命令行对话循环。"""
    # 解析命令行参数
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 加载项目配置
    config = load_config()

    # 只要用户请求列会话，就打印后退出
    if args.resume == "list":
        metas = list_sessions(config.workspace_root)
        print(format_session_list(metas))
        return

    # 初始化工具注册表
    tool_registry = build_tool_registry()

    # 初始化模型适配器
    model = OpenAIModelAdapter(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model,
        tool_registry=tool_registry,
    )  # type: ignore[arg-type]

    # 创建工具执行上下文
    tool_context = ToolContext(
        cwd=config.workspace_root,
    )

    # 创建或恢复会话
    session = _load_or_create_session(
        workspace=config.workspace_root,
        session_id=args.session.strip(),
        # 没传 --resume 时兜底为空字符串，保持后续分支判断简单
        resume=args.resume or "",
    )

    # 用会话历史初始化当前运行时历史
    history: list[ChatMessage] = list(session.messages)

    print("LongBean MiniCode Agent 已启动，输入 quit 或 exit 退出。")

    while True:
        # 读取用户输入
        user_input = input("You> ").strip()

        # 空输入直接跳过
        if not user_input:
            continue

        # 退出前先保存当前会话
        if user_input.lower() in {"quit", "exit"}:
            session.replace_messages(history)
            save_session(session)
            print("Bye!")
            break

        # 先保存本轮开始前的历史。
        # 如果后面触发授权重试，需要从这个“干净历史”重新跑，
        # 避免把中途产生的 user / tool_call 重复写进会话。
        base_history = list(history)

        # 先正常跑一轮 Agent
        step, history = run_agent_once(
            user_input=user_input,
            model=model,
            tool_registry=tool_registry,
            tool_context=tool_context,
            history=history,
            session_id=session.session_id,
        )

        # 如果这一步要求授权，就在终端里询问用户
        if step.type == "approval" and step.approval is not None:
            print(step.approval.message)
            answer = input("是否允许本次执行？(y/n)> ").strip().lower()

            if answer == "y":
                # 批准后，把动作键加入当前会话上下文。
                # 同一会话里再次遇到同一条高风险命令时就可以直接放行。
                tool_context.approved_actions.add(step.approval.action_key)

                # 从本轮开始前的干净历史重新跑一遍。
                # 不直接复用 approval 返回后的 history，
                # 否则会重复追加同一条 user 消息和 tool_call 记录。
                step, history = run_agent_once(
                    user_input=user_input,
                    model=model,
                    tool_registry=tool_registry,
                    tool_context=tool_context,
                    history=base_history,
                    session_id=session.session_id,
                )
            else:
                # 拒绝后恢复到本轮开始前的历史，
                # 避免把未真正执行的授权请求残留到会话里。
                history = base_history

                # 拒绝后不执行危险操作，直接给出提示
                print("Agent> 用户已拒绝此次高风险操作。")
                continue

        # 每轮结束后保存最新会话
        session.replace_messages(history)
        save_session(session)

        # 打印模型返回内容
        print(f"Agent> {step.content}")


if __name__ == "__main__":
    main()
