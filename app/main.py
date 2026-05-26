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
from app.working_memory import WorkingMemory


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
    if session_id:
        session = load_session(workspace, session_id)
        if session is None:
            raise FileNotFoundError(f"未找到会话 {session_id}")
        print(f"已恢复指定会话: {session.session_id}")
        return session

    if resume == "latest":
        session = get_latest_session(workspace)
        if session is not None:
            print(f"已恢复最近会话: {session.session_id}")
            return session

        session = create_new_session(workspace)
        print(f"未找到最近会话，已新建会话: {session.session_id}")
        return session

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
    """按 tool_use_id 精确替换待授权的占位 tool_result。"""
    updated_history: list[ChatMessage] = []
    replaced = False

    for msg in history:
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

        updated_history.append(msg)

    # 极端情况下没找到占位结果时，补一条真实结果，避免历史断链
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
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = load_config()

    if args.resume == "list":
        metas = list_sessions(config.workspace_root)
        print(format_session_list(metas))
        return

    tool_registry = build_tool_registry()

    model = OpenAIModelAdapter(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model,
        tool_registry=tool_registry,
    )  # type: ignore[arg-type]

    # 当前工具默认在工作区根目录下执行
    tool_context = ToolContext(
        cwd=config.workspace_root,
    )

    # 当前会话的短期工作记忆
    working_memory = WorkingMemory()

    session = _load_or_create_session(
        workspace=config.workspace_root,
        session_id=args.session.strip(),
        resume=args.resume or "",
    )

    history: list[ChatMessage] = list(session.messages)

    print("LongBean MiniCode Agent 已启动，输入 quit 或 exit 退出。")

    while True:
        user_input = input("You> ").strip()

        if not user_input:
            continue

        # 先把本轮用户输入记为当前主要目标
        working_memory.set_current_goal(user_input)

        if user_input.lower() in {"quit", "exit"}:
            session.replace_messages(history)
            save_session(session)
            print("Bye!")
            break

        step, history = run_agent_once(
            user_input=user_input,
            model=model,
            tool_registry=tool_registry,
            tool_context=tool_context,
            working_memory=working_memory,
            history=history,
            session_id=session.session_id,
        )

        if step.type == "approval" and step.approval is not None:
            print(step.approval.message)
            answer = input("是否允许本次执行？(y/n)> ").strip().lower()

            if answer == "y":
                # 同一会话里同一条高风险动作后续可直接放行
                tool_context.approved_actions.add(step.approval.action_key)

                # 批准后直接执行待授权工具，不重新问模型
                result = tool_registry.execute_tool(
                    tool_name=step.approval.tool_name,
                    input_data=step.approval.input_data,
                    context=tool_context,
                )

                approved_history = _replace_pending_tool_result(
                    history=history,
                    tool_use_id=step.approval.tool_use_id,
                    tool_name=step.approval.tool_name,
                    content=result.output,
                    is_error=not result.ok,
                )

                # 再从真实 tool_result 已写回的历史继续跑主循环
                step, history = continue_agent_from_history(
                    history=approved_history,
                    model=model,
                    tool_registry=tool_registry,
                    tool_context=tool_context,
                    working_memory=working_memory,
                    session_id=session.session_id,
                )
            else:
                # 记录一次授权拒绝，提醒后续模型不要默认继续危险操作
                working_memory.add_failure("用户拒绝了高风险操作授权")
                print("Agent> 用户已拒绝此次高风险操作。")
                continue

        session.replace_messages(history)
        save_session(session)

        print(f"Agent> {step.content}")


if __name__ == "__main__":
    main()
