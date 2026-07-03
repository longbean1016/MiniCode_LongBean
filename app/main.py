from __future__ import annotations

import argparse

"""命令行入口，负责启动会话、组装依赖并驱动整轮交互。"""

from app.agent.loop import continue_agent_from_history, run_agent_once
from app.infra.background_worker import wait_for_background_tasks
from app.config import load_config
from app.context.history_summarizer import OlderHistorySummarizer
from app.logger import log_event
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
    import asyncio

    _t_startup = time.time()
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ── 检查配置：没有 api_key → 首次启动配置向导 ──
    # 对标 Claude Code 首次启动时创建 ~/.claude/settings.json 的流程
    from app.infra.user_config import has_api_key, ensure_user_config

    if not has_api_key():
        # 确保配置文件骨架存在（api_key 为空）
        ensure_user_config()
        # 启动 TUI 配置向导收集 api_key / base_url / model
        from app.tui.setup_screen import SetupScreen
        # 用 Textual 独立启动配置界面
        from textual.app import App as TextualApp

        class _SetupApp(TextualApp):
            def on_mount(self) -> None:
                self.push_screen(SetupScreen())

        setup_app = _SetupApp()
        setup_app.run()
        # 配置向导完成后重新检查
        if not has_api_key():
            print("未配置 API Key，退出。")
            return

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

    tool_context = ToolContext(
        cwd=config.workspace_root,
        permanent_workspaces=set(config.workspace_additional_dirs),
    )

    # 新版 Hermes 风格 MemoryStore：MEMORY.md + USER.md，Markdown 列表格式
    from app.memory.memory_store import MemoryStore
    from app.memory.memory_tool import configure_memory_tool
    from app.memory.review import build_review_runner

    memory_store = MemoryStore(config.workspace_root)

    # 配置 memory_tool 的全局 MemoryStore
    configure_memory_tool(memory_store)

    # 后台反思 runner
    review_runner = build_review_runner(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model,
        memory_store=memory_store,
        retry_max_attempts=config.aux_model_retry_max_attempts,
        retry_base_delay_seconds=config.aux_model_retry_base_delay_seconds,
        retry_backoff_multiplier=config.aux_model_retry_backoff_multiplier,
        retry_max_delay_seconds=config.aux_model_retry_max_delay_seconds,
        circuit_failure_threshold=config.aux_model_circuit_failure_threshold,
        circuit_recovery_timeout_seconds=config.aux_model_circuit_recovery_timeout_seconds,
    )

    # 注入 review runner 到 loop 模块，自动触发后台反思
    from app.agent.loop import configure_review_runner
    configure_review_runner(review_runner)

    # 旧历史摘要器
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
        history_summarizer=history_summarizer,
        mcp_manager=mcp_manager,  # 用于 /mcp 命令和退出清理
    )

    tui_app.run()

    # TUI 退出后会回到这里（exit/quit 命令或 Ctrl+C）
    wait_for_background_tasks()
    print("Bye!")


if __name__ == "__main__":
    main()
