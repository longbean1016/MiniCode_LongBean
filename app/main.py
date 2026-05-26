from __future__ import annotations

import argparse

from app.agent_loop import continue_agent_from_history, run_agent_once
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


def _replace_pending_tool_result(
    history: list[ChatMessage],
    tool_use_id: str,
    tool_name: str,
    content: str,
    is_error: bool,
) -> list[ChatMessage]:
    """按 tool_use_id 精确替换待授权的占位 tool_result，避免误删其他历史。"""
    updated_history: list[ChatMessage] = []
    replaced = False

    for msg in history:
        # 只替换目标 tool_use_id 对应的那条 tool_result
        if (
            msg.get("role") == "tool_result"
            and msg.get("tool_use_id") == tool_use_id
            and not replaced
        ):
            updated_history.append(
                {
                    "role": "tool_result",
                    "tool_use_id": tool_use_id,
                    "tool_name": tool_name,
                    "content": content,
                    "is_error": is_error,
                }
            )
            replaced = True
            continue

        # 其他历史原样保留
        updated_history.append(msg)

    # 正常情况下应当能找到占位结果；兜底时补一条真实结果，避免历史断链
    if not replaced:
        updated_history.append(
            {
                "role": "tool_result",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "content": content,
                "is_error": is_error,
            }
        )

    return updated_history


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

                # 批准后不再重问模型，而是直接把待授权工具真正执行掉。
                result = tool_registry.execute_tool(
                    tool_name=step.approval.tool_name,
                    input_data=step.approval.input_data,
                    context=tool_context,
                )

                # 按 tool_use_id 精确替换占位结果，避免误删或漏掉其他消息。
                approved_history = _replace_pending_tool_result(
                    history=history,
                    tool_use_id=step.approval.tool_use_id,
                    tool_name=step.approval.tool_name,
                    content=result.output,
                    is_error=not result.ok,
                )

                # 再从“真实 tool_result 已写回”的历史继续跑模型总结。
                step, history = continue_agent_from_history(
                    history=approved_history,
                    model=model,
                    tool_registry=tool_registry,
                    tool_context=tool_context,
                    session_id=session.session_id,
                )
            else:
                # 拒绝后保留授权提示这段历史，但不执行危险操作。
                print("Agent> 用户已拒绝此次高风险操作。")
                continue

        # 每轮结束后保存最新会话
        session.replace_messages(history)
        save_session(session)

        # 打印模型返回内容
        print(f"Agent> {step.content}")


if __name__ == "__main__":
    main()
