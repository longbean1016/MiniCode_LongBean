from __future__ import annotations

import argparse

from app.agent_loop import continue_agent_from_history, run_agent_once
from app.config import load_config
from app.memory_extractor import LongTermMemoryExtractor
from app.memory_store import JsonMemoryStore, MemoryEntry
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
from app.turn_history import get_last_round_messages
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


def _should_extract_long_term_memory(step_content: str) -> bool:
    """判断这次最终回复是否适合触发长期记忆抽取。"""
    # 空回复不抽。
    text = step_content.strip()
    if not text:
        return False

    # 这些是内部兜底回复，不适合作为长期记忆抽取的“最终结果”。
    blocked_prefixes = (
        "模型调用失败:",
        "已达到最大循环步数",
        "未识别的模型返回类型",
        "模型返回了空的工具调用",
    )
    return not text.startswith(blocked_prefixes)


def _persist_extracted_memories(
    memory_store: JsonMemoryStore,
    entries: list[MemoryEntry],
) -> int:
    """
    把抽取出来的长期记忆写入 memory store。

    写入前再做一层本地去重，
    避免相同 category + content 被重复写入。
    """
    # 先读取已有长期记忆，构造一个稳定的去重键集合。
    existing_keys = {
        (
            entry.category.strip().lower(),
            " ".join(entry.content.strip().lower().split()),
        )
        for entry in memory_store.load_memories()
        if entry.content.strip()
    }

    stored_count = 0

    for entry in entries:
        # 当前候选记忆的去重键。
        dedupe_key = (
            entry.category.strip().lower(),
            " ".join(entry.content.strip().lower().split()),
        )

        # 已存在就跳过，不重复写。
        if dedupe_key in existing_keys:
            continue

        memory_store.add_memory(entry)
        existing_keys.add(dedupe_key)
        stored_count += 1

    return stored_count


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

    # 长期记忆先落到本地 JSON。
    # 后面如果要换成向量库，只需要替换这一行的具体实现。
    memory_store = JsonMemoryStore(config.workspace_root)

    # 当前会话的短期工作记忆
    working_memory = WorkingMemory()

    # 长期记忆抽取器：负责从“当前一轮完整消息”里提炼可复用记忆。
    memory_extractor = LongTermMemoryExtractor(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model,
    )

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
            session=session,
            working_memory=working_memory,
            memory_store=memory_store,
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
                    session=session,
                    working_memory=working_memory,
                    memory_store=memory_store,
                    session_id=session.session_id,
                )
            else:
                # 记录一次授权拒绝，提醒后续模型不要默认继续危险操作
                working_memory.add_failure("用户拒绝了高风险操作授权")
                print("Agent> 用户已拒绝此次高风险操作。")
                continue

        # 只有真正得到最终 assistant 回复时，才尝试做长期记忆抽取。
        if step.type == "assistant" and _should_extract_long_term_memory(step.content):
            try:
                # 从完整 history 里取出最后一轮消息，避免拿全量历史去抽长期记忆。
                last_round_messages = get_last_round_messages(history)

                # 基于“当前用户输入 + 本轮最终结果 + 本轮完整消息链”抽取长期记忆。
                extracted_memories = memory_extractor.extract_from_turn(
                    user_input=user_input,
                    final_step=step,
                    turn_messages=last_round_messages,
                    session_id=session.session_id,
                )

                # 写入前再做一层去重保护。
                _persist_extracted_memories(memory_store, extracted_memories)
            except Exception:
                # 长期记忆抽取失败不能影响主流程回答。
                pass

        session.replace_messages(history)
        save_session(session)

        print(f"Agent> {step.content}")


if __name__ == "__main__":
    main()
