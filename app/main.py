from __future__ import annotations

import argparse

from app.agent_loop import continue_agent_from_history, run_agent_once
from app.config import load_config
from app.memory_decay import DecayRunResult, MemoryDecay
from app.history_summarizer import OlderHistorySummarizer
from app.logger import log_event
from app.memory_curator import MemoryCurator
from app.memory_extractor import LongTermMemoryExtractor
from app.memory_guard import MemoryWriteGuard
from app.memory_reflection_policy import (
    decide_project_reflection,
    should_reflect_long_term_memory,
)
from app.memory_store import JsonMemoryStore, MemoryEntry
from app.memory_vector_index import MemoryVectorIndex
from app.memory_verifier import MemoryVerifier
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
from app.working_memory_updater import extract_active_paths, summarize_failure


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


def _clear_reflection_runtime_context(working_memory: WorkingMemory) -> None:
    """
    清理上一轮留下的 reflection 运行时条目。

    这些条目只服务当前一轮的 task reflection，
    不应该泄漏到下一轮继续参与反思。
    """
    working_memory.clear_entries_by_type(
        "reflection_decision",
        "reflection_failure",
        "reflection_file",
    )


def _collect_reflection_entries(
    working_memory: WorkingMemory,
    entry_type: str,
    *,
    limit: int,
) -> list[str]:
    """
    从运行时工作记忆里取出当前轮的反思辅助条目，并做轻量去重。

    参数说明：
    - `entry_type`: 需要读取的运行时条目类型
    - `limit`: 最多返回多少条，避免单轮反思输入失控
    """
    result: list[str] = []
    seen: set[str] = set()

    for entry in working_memory.get_entries_by_type(entry_type):
        content = entry.content.strip()
        if not content or content in seen:
            continue
        seen.add(content)
        result.append(content)
        if len(result) >= limit:
            break

    return result


def _persist_extracted_memories(
    memory_store: JsonMemoryStore,
    memory_guard: MemoryWriteGuard,
    memory_verifier: MemoryVerifier,
    entries: list[MemoryEntry],
) -> list[MemoryEntry]:
    """
    把抽取出来的长期记忆写入 memory store。

    当前阶段的写入流程是：
    1. 先读取已有长期记忆
    2. 逐条走 `MemoryWriteGuard` 做快速门禁
    3. 再走 `MemoryVerifier` 做 duplicate / conflict / reject / supersede_store 判断
    4. 只有 verifier 判定为 `store` 或 `supersede_store` 才真正落盘

    `supersede_store` 是 minicode 风格更新链路的关键：
    - 新记忆先允许入库
    - 再把“它替代的是哪条旧记忆”一起记到 extra 里
    - 后面的 curator 会根据这条显式关系，把旧记忆降级成 superseded
    """
    existing_entries = memory_store.load_memories()
    stored_entries: list[MemoryEntry] = []

    for entry in entries:
        guard_decision = memory_guard.should_store(entry)
        if not guard_decision.should_store:
            continue

        similar_entries = memory_verifier.find_similar_entries(entry, existing_entries)
        verify_decision = memory_verifier.verify(entry, similar_entries)
        if verify_decision.action not in {"store", "supersede_store"}:
            continue

        # 如果 verifier 已经确认“这是一条替代更新”，
        # 就把被替代的旧记忆 id 显式写入 extra。
        # 这样 curator 后面不必完全依赖启发式猜测关系，
        # 能更稳定地把旧版本归档出主检索面。
        if (
            verify_decision.action == "supersede_store"
            and verify_decision.matched_memory_id.strip()
        ):
            entry.extra["supersedes_memory_id"] = verify_decision.matched_memory_id.strip()
            entry.extra["write_action"] = "supersede_store"

        stored_entry = memory_store.add_memory(entry)
        existing_entries.append(stored_entry)
        stored_entries.append(stored_entry)

    return stored_entries


def _log_decay_run_result(
    *,
    session_id: str,
    stage: str,
    result: DecayRunResult,
    enabled: bool,
    echo: bool,
) -> None:
    """
    记录一轮 decay 的摘要日志。

    这里只记录汇总信息：
    - 扫描了多少条
    - 发生了多少次分数刷新或归档
    - 其中有多少条被归档
    避免把细节日志直接打满正常对话输出。
    """
    if not enabled or result.changed_count <= 0:
        return

    log_event(
        (
            f"[session={session_id}] decay[{stage}] "
            f"scanned={result.scanned_count} "
            f"changed={result.changed_count} "
            f"archived={result.archived_count()}"
        ),
        echo=echo,
    )


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
        retry_max_attempts=config.model_retry_max_attempts,
        retry_base_delay_seconds=config.model_retry_base_delay_seconds,
        retry_backoff_multiplier=config.model_retry_backoff_multiplier,
        retry_max_delay_seconds=config.model_retry_max_delay_seconds,
        circuit_failure_threshold=config.model_circuit_failure_threshold,
        circuit_recovery_timeout_seconds=config.model_circuit_recovery_timeout_seconds,
    )  # type: ignore[arg-type]

    # 当前工具默认在工作区根目录下执行
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

    # 当前会话的短期工作记忆
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

        # 每一轮开始前先清掉上一轮的反思辅助条目。
        _clear_reflection_runtime_context(working_memory)

        # 先把本轮用户输入记为当前主要目标
        working_memory.protect(
            user_input,
            entry_type="user_intent",
            ttl_seconds=3600,
            importance=1.0,
            replace_latest_of_type=True,
        )

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
            history_summarizer=history_summarizer,
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

                # 授权后的工具执行不会经过 agent_loop 的常规工具分支，
                # 所以这里手动补齐反思输入采集，避免漏掉路径触点和失败摘要。
                for path in extract_active_paths(
                    step.approval.tool_name,
                    step.approval.input_data,
                ):
                    working_memory.protect(
                        path,
                        entry_type="active_task",
                        ttl_seconds=1800,
                        importance=0.8,
                    )
                    working_memory.protect(
                        path,
                        entry_type="reflection_file",
                        ttl_seconds=1800,
                        importance=0.7,
                    )

                if not result.ok:
                    failure_summary = summarize_failure(step.approval.tool_name, result)
                    working_memory.protect(
                        failure_summary,
                        entry_type="error_context",
                        ttl_seconds=1800,
                        importance=0.9,
                    )
                    working_memory.protect(
                        failure_summary,
                        entry_type="reflection_failure",
                        ttl_seconds=1800,
                        importance=0.9,
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

        # 只有命中“任务完成 / 阶段完成 / 稳定结论”时，才尝试做长期记忆反思。
        if should_reflect_long_term_memory(step):
            try:
                # 只用当前最后一轮消息做反思，不拿整段全量历史直接抽记忆。
                last_round_messages = get_last_round_messages(history)
                key_decisions = _collect_reflection_entries(
                    working_memory,
                    "reflection_decision",
                    limit=6,
                )
                failures = _collect_reflection_entries(
                    working_memory,
                    "reflection_failure",
                    limit=6,
                )
                files_touched = _collect_reflection_entries(
                    working_memory,
                    "reflection_file",
                    limit=10,
                )
                reflection_decision = decide_project_reflection(
                    files_touched=files_touched,
                    failures=failures,
                )

                if not reflection_decision.should_reflect:
                    log_event(
                        (
                            f"[session={session.session_id}] "
                            f"跳过自动长期记忆反思: {reflection_decision.reason}"
                        ),
                        echo=False,
                    )
                    print(f"Agent> {step.content}")
                    continue

                # 基于“任务描述 + 最终结果 + 当前轮完整消息链”做 task reflection。
                extracted_memories = memory_extractor.extract_from_task(
                    task_description=user_input,
                    final_step=step,
                    turn_messages=last_round_messages,
                    session_id=session.session_id,
                    key_decisions=key_decisions,
                    failures=failures,
                    files_touched=reflection_decision.project_files_touched,
                )

                # 候选记忆先过 guard，再允许落盘。
                stored_entries = _persist_extracted_memories(
                    memory_store,
                    memory_guard,
                    memory_verifier,
                    extracted_memories,
                )

                # 只围绕本次新写入的 project 记忆做增量整理。
                # curator 的职责是让长期记忆“逐步收敛”，
                # 避免系统长期运行后只增不减、重复堆积。
                if stored_entries:
                    memory_curator.curate_new_entries(stored_entries)
                    try:
                        incremental_decay_result = memory_decay.refresh_new_entries(
                            stored_entries
                        )
                    except Exception as error:
                        log_event(
                            f"[session={session.session_id}] decay[incremental] 执行失败: {error}",
                            echo=config.decay_log_echo,
                        )
                    else:
                        _log_decay_run_result(
                            session_id=session.session_id,
                            stage="incremental",
                            result=incremental_decay_result,
                            enabled=config.decay_log_enabled,
                            echo=config.decay_log_echo,
                        )

                    # 当 active project 记忆增长到一定规模后，
                    # 再额外触发一次低频全量整理，避免只做增量时留下历史脏数据。
                    if memory_curator.should_run_full_scan():
                        memory_curator.curate_project_memories()
                    if memory_decay.should_run_full_refresh():
                        try:
                            full_decay_result = memory_decay.refresh_project_memories()
                        except Exception as error:
                            log_event(
                                f"[session={session.session_id}] decay[full] 执行失败: {error}",
                                echo=config.decay_log_echo,
                            )
                        else:
                            _log_decay_run_result(
                                session_id=session.session_id,
                                stage="full",
                                result=full_decay_result,
                                enabled=config.decay_log_enabled,
                                echo=config.decay_log_echo,
                            )
            except Exception as error:
                # 长期记忆反思失败不能影响主流程回答，
                # 但必须把原因打印并写日志，否则会变成“静默丢记忆”，很难排查。
                log_event(
                    f"[session={session.session_id}] 长期记忆写入失败: {error}"
                )
                print(f"[memory-reflection-error] {error}")

        session.replace_messages(history)
        save_session(session)

        print(f"Agent> {step.content}")


if __name__ == "__main__":
    main()
