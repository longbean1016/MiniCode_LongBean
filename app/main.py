from __future__ import annotations

import argparse
import copy
import time

"""命令行入口，负责启动会话、组装依赖并驱动整轮交互。"""

from app.agent_loop import continue_agent_from_history, run_agent_once
from app.background_worker import submit_background, wait_for_background_tasks
from app.config import load_config
from app.memory_decay import MemoryDecay
from app.history_summarizer import OlderHistorySummarizer
from app.logger import log_event
from app.memory_curator import MemoryCurator
from app.memory_extractor import LongTermMemoryExtractor
from app.memory_feedback import MemoryFeedbackStore
from app.memory_guard import MemoryWriteGuard
from app.memory_pipeline import MemoryPipeline
from app.memory_read_pipeline import MemoryReadPipeline
from app.memory_store import JsonMemoryStore
from app.memory_vector_index import MemoryVectorIndex
from app.memory_verifier import MemoryVerifier
from app.memory_write_pipeline import MemoryWritePipeline
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
from app.types import AgentStep, ChatMessage, ToolContext
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
    # 优先级是：
    # 1. 显式指定 session_id
    # 2. --resume latest
    # 3. 新建
    # 这样命令行行为始终可预测，不会因为“最近会话存在”就偷偷覆盖用户显式选择。
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
        # 只替换第一条匹配占位，避免历史里若存在重复残留时把多条结果一起污染。
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

    # 配置先加载，再组装所有依赖。
    # main 的职责不是执行业务逻辑，而是把“模型 / 工具 / 记忆 / 会话”这些基础设施接起来。
    config = load_config()

    if args.resume == "list":
        metas = list_sessions(config.workspace_root)
        print(format_session_list(metas))
        return

    # 工具注册表和模型适配器都在启动时构建，
    # 后续整轮会话里复用同一套实例，避免每次用户输入都重新初始化依赖。
    tool_registry = build_tool_registry()

    model = OpenAIModelAdapter(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model,
        tool_registry=tool_registry,
        retry_max_attempts=config.model_retry_max_attempts,
        retry_base_delay_seconds=config.model_retry_base_delay_seconds,
        retry_backoff_multiplier=config.model_retry_backoff_multiplier,
        retry_max_delay_seconds=config.model_retry_max_delay_seconds,
        circuit_failure_threshold=config.model_circuit_failure_threshold,
        circuit_recovery_timeout_seconds=config.model_circuit_recovery_timeout_seconds,
    )  # type: ignore[arg-type]

    # ToolContext 保存的是“工具运行时共享状态”，
    # 例如当前工作目录、已批准动作；它不是一次性参数，而是整场会话的执行上下文。
    tool_context = ToolContext(
        cwd=config.workspace_root,
    )

    # 只有显式开启 Qdrant 时，才初始化服务端向量索引。
    # 这样默认开发模式仍然可以只用本地 JSON 跑起来。
    vector_index: MemoryVectorIndex | None = None
    if config.qdrant_enabled:
        vector_index = MemoryVectorIndex(
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
            embedding_model=config.embedding_model,
            embedding_dimensions=config.embedding_dimensions,
            qdrant_url=config.qdrant_url,
            qdrant_path=config.qdrant_path,
            qdrant_api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection,
            retry_max_attempts=config.vector_retry_max_attempts,
            retry_base_delay_seconds=config.vector_retry_base_delay_seconds,
            retry_backoff_multiplier=config.vector_retry_backoff_multiplier,
            retry_max_delay_seconds=config.vector_retry_max_delay_seconds,
            circuit_failure_threshold=config.vector_circuit_failure_threshold,
            circuit_recovery_timeout_seconds=config.vector_circuit_recovery_timeout_seconds,
        )

    # 长期记忆的权威存储仍然是本地 JSON。
    # 如果配置了 Qdrant，则在写入 JSON 后同步建立语义向量索引。
    memory_store = JsonMemoryStore(
        config.workspace_root,
        vector_index=vector_index,
    )

    # 启动时先做一次全量收敛：
    # - 让 Qdrant payload 跟上本地最新 metadata
    # - 删除历史测试或异常遗留的孤儿点
    # 本地 JSON 仍然是权威来源，向量库只做语义检索和重复判断。
    if vector_index is not None:
        try:
            memory_store.reconcile_vector_index()
        except Exception as error:
            # 向量索引初始化收敛属于非主链路能力。
            # 即使这里失败，也应该允许主 agent 继续启动和回答。
            log_event(
                f"[session=startup] 向量索引初始化收敛失败，已降级继续运行: {error}"
            )

    # WorkingMemory 只承担“本次会话里近期要持续遵守/记住的事实”，
    # 长期知识仍然交给 memory store / memory pipeline。
    working_memory = WorkingMemory()

    # 长期记忆抽取器：负责把本轮任务整理成 task reflection，并产出候选记忆。
    memory_extractor = LongTermMemoryExtractor(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model,
        retry_max_attempts=config.aux_model_retry_max_attempts,
        retry_base_delay_seconds=config.aux_model_retry_base_delay_seconds,
        retry_backoff_multiplier=config.aux_model_retry_backoff_multiplier,
        retry_max_delay_seconds=config.aux_model_retry_max_delay_seconds,
        circuit_failure_threshold=config.aux_model_circuit_failure_threshold,
        circuit_recovery_timeout_seconds=config.aux_model_circuit_recovery_timeout_seconds,
    )
    memory_guard = MemoryWriteGuard()
    memory_verifier = MemoryVerifier(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model,
        vector_index=vector_index,
        retry_max_attempts=config.aux_model_retry_max_attempts,
        retry_base_delay_seconds=config.aux_model_retry_base_delay_seconds,
        retry_backoff_multiplier=config.aux_model_retry_backoff_multiplier,
        retry_max_delay_seconds=config.aux_model_retry_max_delay_seconds,
        circuit_failure_threshold=config.aux_model_circuit_failure_threshold,
        circuit_recovery_timeout_seconds=config.aux_model_circuit_recovery_timeout_seconds,
    )
    memory_curator = MemoryCurator(memory_store)
    memory_decay = MemoryDecay(
        memory_store,
        full_scan_trigger_count=config.decay_full_scan_trigger_count,
        min_decay_score=config.decay_min_score,
        archive_decay_threshold=config.decay_archive_threshold,
        archive_age_days=config.decay_archive_age_days,
        archive_confidence_threshold=config.decay_archive_confidence_threshold,
        archive_usage_threshold=config.decay_archive_usage_threshold,
    )
    memory_pipeline = MemoryPipeline(
        read_pipeline=MemoryReadPipeline(memory_store),
        write_pipeline=MemoryWritePipeline(
            memory_store=memory_store,
            memory_extractor=memory_extractor,
            memory_verifier=memory_verifier,
            memory_curator=memory_curator,
            memory_decay=memory_decay,
            memory_guard=memory_guard,
        ),
        feedback_store=MemoryFeedbackStore(memory_store),
    )

    # 旧历史摘要器只服务于 prompt 构造时的 older history summary。
    # 它带运行期缓存，避免每轮循环都重复调用模型做摘要。
    history_summarizer = OlderHistorySummarizer(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model,
        retry_max_attempts=config.aux_model_retry_max_attempts,
        retry_base_delay_seconds=config.aux_model_retry_base_delay_seconds,
        retry_backoff_multiplier=config.aux_model_retry_backoff_multiplier,
        retry_max_delay_seconds=config.aux_model_retry_max_delay_seconds,
        circuit_failure_threshold=config.aux_model_circuit_failure_threshold,
        circuit_recovery_timeout_seconds=config.aux_model_circuit_recovery_timeout_seconds,
    )
    # assistant 回复和 microcompact 的轻量语义抽取复用同一个摘要器配置，
    # 因此会使用当前运行时配置的 flash / 辅助模型，而不是在写链路里写死模型名。
    memory_pipeline.write_pipeline.history_summarizer = history_summarizer

    # 所有基础设施都就绪后再恢复/创建会话，
    # 这样一进主循环就能直接处理 explicit memory 命令、工具调用和长期记忆读写。
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

        # 每轮开始先清空“只属于上一轮”的运行时痕迹，再写入本轮用户意图。
        # 否则反思链路、失败上下文、短期任务状态会在多轮后互相污染。
        memory_pipeline.reset_turn_runtime(working_memory)
        memory_pipeline.remember_user_intent(working_memory, user_input)

        if user_input.lower() in {"quit", "exit"}:
            session.replace_messages(history)
            save_session(session)
            wait_for_background_tasks()
            print("Bye!")
            break

        # 先处理显式命令型输入，如 /memory add。
        # 这类输入本质上不是“问模型”，而是直接操作本地记忆系统，应该短路主 agent loop。
        explicit_result = memory_pipeline.handle_explicit_input(
            user_input=user_input,
            session_id=session.session_id,
            history=history,
            decay_log_enabled=config.decay_log_enabled,
            decay_log_echo=config.decay_log_echo,
        )
        if explicit_result.handled:
            history = explicit_result.history
            session.replace_messages(history)
            save_session(session)
            print(f"Agent> {explicit_result.assistant_text}")
            continue

        # 常规自然语言输入才进入主 agent loop。
        # run_agent_once 会负责拼装 prompt、调用模型、执行低风险工具，并返回新的 history。
        step, history = run_agent_once(
            user_input=user_input,
            model=model,
            tool_registry=tool_registry,
            tool_context=tool_context,
            session=session,
            working_memory=working_memory,
            memory_pipeline=memory_pipeline,
            history_summarizer=history_summarizer,
            history=history,
            session_id=session.session_id,
        )

        if step.type == "approval" and step.approval is not None:
            # 高风险工具不会直接执行，而是先把“待批准占位结果”写进历史，交给用户确认。
            print(step.approval.message)
            answer = input("是否允许本次执行？(y/n)> ").strip().lower()

            if answer == "y":
                # 同一会话里同一条高风险动作后续可直接放行
                tool_context.approved_actions.add(step.approval.action_key)

                # 批准后直接执行待授权工具，不重新问模型。
                # 否则模型可能在批准前后生成不同 tool_call，导致用户批准的并不是最终执行的那次动作。
                result = tool_registry.execute_tool(
                    tool_name=step.approval.tool_name,
                    input_data=step.approval.input_data,
                    context=tool_context,
                )

                memory_pipeline.record_tool_call(
                    working_memory,
                    tool_name=step.approval.tool_name,
                    tool_input=step.approval.input_data,
                )
                if not result.ok:
                    memory_pipeline.record_tool_failure(
                        working_memory,
                        tool_name=step.approval.tool_name,
                        result=result,
                    )

                approved_history = _replace_pending_tool_result(
                    history=history,
                    tool_use_id=step.approval.tool_use_id,
                    tool_name=step.approval.tool_name,
                    content=result.output,
                    is_error=not result.ok,
                )

                # 把真实 tool_result 写回历史后，再继续主循环。
                # 这样模型下一步看到的是“已经执行完的事实”，而不是一条脱节的批准结果。
                step, history = continue_agent_from_history(
                    history=approved_history,
                    model=model,
                    tool_registry=tool_registry,
                    tool_context=tool_context,
                    session=session,
                    working_memory=working_memory,
                    memory_pipeline=memory_pipeline,
                    history_summarizer=history_summarizer,
                    session_id=session.session_id,
                )
            else:
                # 记录一次授权拒绝，提醒后续模型不要默认继续危险操作
                working_memory.protect(
                    "用户拒绝了高风险操作授权",
                    entry_type="error_context",
                    ttl_seconds=1800,
                    importance=0.9,
                )
                print("Agent> 用户已拒绝此次高风险操作。")
                continue

        if step.type == "assistant":
            # finalize_turn 可能触发长期记忆抽取、验证和持久化，耗时会直接阻塞
            # Agent> 输出。这里先拷贝本轮快照，再交给后台顺序执行。
            finalize_task_description = user_input
            finalize_step = copy.deepcopy(step)
            finalize_turn_messages = copy.deepcopy(get_last_round_messages(history))
            finalize_session_id = session.session_id
            finalize_working_memory = copy.deepcopy(working_memory)
            finalize_decay_log_enabled = config.decay_log_enabled
            finalize_decay_log_echo = config.decay_log_echo

            def _finalize_turn_background() -> None:
                started_at = time.perf_counter()
                try:
                    memory_pipeline.finalize_turn(
                        task_description=finalize_task_description,
                        final_step=finalize_step,
                        turn_messages=finalize_turn_messages,
                        session_id=finalize_session_id,
                        working_memory=finalize_working_memory,
                        decay_log_enabled=finalize_decay_log_enabled,
                        decay_log_echo=finalize_decay_log_echo,
                    )
                    elapsed = time.perf_counter() - started_at
                    log_event(
                        f"[session={finalize_session_id}] 长期记忆后台写入完成 耗时={elapsed:.3f}s",
                        echo=False,
                    )
                except Exception as error:
                    log_event(
                        f"[session={finalize_session_id}] 长期记忆后台写入失败: {error}",
                        echo=False,
                    )

            submit_background(
                _finalize_turn_background,
                name="finalize_turn",
            )

        # 每轮末尾都把最新 history 落回 session，再持久化到磁盘。
        # 这样即使下一轮或下次启动中断，也能从最近一次完成态恢复。
        session.replace_messages(history)
        save_session(session)

        print(f"Agent> {step.content}")


if __name__ == "__main__":
    main()
