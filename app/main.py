from __future__ import annotations

import argparse

"""命令行入口，负责启动会话、组装依赖并驱动整轮交互。"""

from app.agent.loop import continue_agent_from_history, run_agent_once
from app.infra.background_worker import wait_for_background_tasks
from app.config import load_config
from app.memory.decay import MemoryDecay
from app.context.history_summarizer import OlderHistorySummarizer
from app.logger import log_event
from app.memory.curator import MemoryCurator
from app.memory.extractor import LongTermMemoryExtractor
from app.memory.pipeline import MemoryFeedbackStore
from app.memory.guard import MemoryWriteGuard
from app.memory.pipeline import MemoryPipeline
from app.memory.read_pipeline import MemoryReadPipeline
from app.memory.store import JsonMemoryStore
from app.memory.vector_index import MemoryVectorIndex
from app.memory.verifier import MemoryVerifier
from app.memory.write_pipeline import MemoryWritePipeline
from app.infra.model_registry import OpenAIModelAdapter
from app.state.session import (
    SessionData,
    create_new_session,
    format_session_list,
    get_latest_session,
    list_sessions,
    load_session,
    save_session,
)
from app.tools import build_tool_registry
from app.tui.app import MiniCodeApp
from app.types import AgentStep, ChatMessage, ToolContext
from app.state.working_memory import WorkingMemory


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
    # 这样命令行行为始终可预测，不会因为"最近会话存在"就偷偷覆盖用户显式选择。
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


def main() -> None:
    """程序入口：加载配置、恢复会话、启动 TUI 终端交互。"""
    import time

    _t_startup = time.time()
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 配置先加载，再组装所有依赖。
    config = load_config()
    print(f"[STARTUP] config load: {time.time() - _t_startup:.1f}s")

    if args.resume == "list":
        metas = list_sessions(config.workspace_root)
        print(format_session_list(metas))
        return

    from app.mcp.config import load_mcp_config

    _t = time.time()
    mcp_config = load_mcp_config(config.workspace_root)
    tool_registry, mcp_manager = build_tool_registry(
        cwd=config.workspace_root,
        mcp_config=mcp_config,
        start_mcp=False,  # 不阻塞，MCP 在 TUI 启动后异步连接
    )
    print(f"[STARTUP] tool registry: {time.time() - _t:.1f}s (MCP deferred, {len(mcp_config)} servers)")

    _t = time.time()
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
    print(f"[STARTUP] model adapter: {time.time() - _t:.1f}s")

    tool_context = ToolContext(cwd=config.workspace_root)

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

    # WorkingMemory 只承担"本次会话里近期要持续遵守/记住的事实"，
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

    print(f"[STARTUP] memory pipeline + summarizer: {time.time() - _t:.1f}s")

    _t = time.time()
    history: list[ChatMessage] = list(session.messages)
    print(f"[STARTUP] session load: {time.time() - _t:.1f}s (messages: {len(history)})")

    print(f"[STARTUP] total startup: {time.time() - _t_startup:.1f}s")
    # MCP Server 在后台异步连接，不阻塞 TUI 启动
    if mcp_config:
        mcp_manager.bootstrap_async(mcp_config)

    # ----
    tui_app = MiniCodeApp(
        model=model,
        tool_registry=tool_registry,
        tool_context=tool_context,
        session=session,
        working_memory=working_memory,
        memory_pipeline=memory_pipeline,
        history_summarizer=history_summarizer,
        mcp_manager=mcp_manager,  # 用于 /mcp 命令和退出清理
    )

    tui_app.run()

    # TUI 退出后会回到这里（exit/quit 命令或 Ctrl+C）
    wait_for_background_tasks()
    print("Bye!")


if __name__ == "__main__":
    main()
