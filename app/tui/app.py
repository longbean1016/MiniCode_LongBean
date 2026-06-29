"""MiniCode TUI 主 App — 组装 Widget、连接 Agent 事件流、管理工作线程。

MiniCodeApp 是三部分的中枢：
1. compose() 组装 Header / Conversation / InputArea 三个 Widget
2. on_input_area_input_submitted() 处理用户输入（普通消息 / 审批 / 显式命令）
3. _run_agent_stream() 在 worker 线程中执行 stream_agent，通过 call_from_thread
   把事件推送到 UI 线程实时渲染

布局（Textual CSS Dock）：
- Header:        dock=top, height=1
- InputArea:     dock=bottom, height=auto
- Conversation:  填充中间剩余空间，可滚动
"""

import copy
import time

from textual.app import App, ComposeResult

from app.agent.loop import stream_agent
from app.tui.events import (
    ApprovalEvent,
    DoneEvent,
    ErrorEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.tui.widgets.conversation import ConversationWidget
from app.tui.widgets.header import HeaderWidget
from app.tui.widgets.input_area import InputArea


class MiniCodeApp(App):
    """MiniCode Agent TUI 主应用。

    使用方式：
        app = MiniCodeApp(
            model=model,
            tool_registry=tool_registry,
            tool_context=tool_context,
            session=session,
            
            history_summarizer=history_summarizer,
        )
        app.run()  # 接管终端，启动 TUI
    """

    # Ctrl+C: 有选中文本时复制，未选中时退出（终端标准行为）
    # Ctrl+Y: 纯文本复制最后回答到剪贴板（不乱码兜底）
    BINDINGS = [
        ("ctrl+c", "quit", "退出"),
        ("ctrl+q", "quit", "退出"),
        ("ctrl+y", "copy_last_response", "复制回答"),
    ]

    # ================================================================
    # CSS 样式：定义三区布局和各消息类型的视觉样式
    # ================================================================
    CSS = """
    HeaderWidget {
        dock: top;
        height: 1;
        background: $panel;
        color: $text-muted;
        text-style: bold;
    }

    InputArea {
        dock: bottom;
        height: auto;
        border-top: solid $primary-darken-1;
    }

    InputArea > #input-prompt {
        width: 3;
        color: $accent;
        content-align: center middle;
    }

    InputArea > #input-hint {
        height: 1;
        color: $text-disabled;
        text-style: italic;
    }

    ConversationWidget {
        scrollbar-size: 1 1;
    }
    """

    def __init__(
        self,
        model=None,
        tool_registry=None,
        tool_context=None,
        session=None,
        
        history_summarizer=None,
        mcp_manager=None,
    ) -> None:
        """初始化 MiniCodeApp。

        所有业务依赖通过构造函数注入，App 不负责创建这些实例。

        Args:
            model: OpenAIModelAdapter 实例
            tool_registry: ToolRegistry 实例
            tool_context: ToolContext 实例（含 cwd、approved_actions）
            session: SessionData 实例
            记忆已改用 MemoryStore 快照注入
            history_summarizer: OlderHistorySummarizer 实例
            mcp_manager: McpManager 实例（可选，用于 /mcp 命令）
        """
        super().__init__()
        self._model = model
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._session = session
        self._working_memory = None
        self._memory_pipeline = None
        self._history_summarizer = history_summarizer
        self._mcp_manager = mcp_manager  # MCP 生命周期管理器
        # 注入异步结果回调: MCP 后台线程完成后通过 call_from_thread 安全更新 UI
        if mcp_manager is not None:
            mcp_manager.set_result_callback(self._on_mcp_async_result)
        # 从 session 初始化消息历史
        self._history = list(session.messages) if session else []
        # 待处理的审批请求（ApprovalEvent 到达后暂存，等待用户 y/n）
        self._pending_approval = None
        # Ctrl+C 退出防误触：记录上一次按下的时间戳
        self._last_quit_attempt = 0.0

    # ================================================================
    # Widget 装配
    # ================================================================

    def compose(self) -> ComposeResult:
        """组装三个主 Widget：Header → Conversation → InputArea。"""
        yield HeaderWidget(
            session_id=self._session.session_id if self._session else "",
            model_name=getattr(self._model, "model_name", "unknown"),
        )
        yield ConversationWidget()
        yield InputArea()

    def action_copy_last_response(self) -> None:
        """Ctrl+Y: 静默复制最后回答到系统剪贴板。"""
        text = self.conversation.get_last_response()
        if not text:
            return

        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["clip"], input=text, text=True, encoding="utf-8", timeout=5
                )
            elif sys.platform == "darwin":
                subprocess.run(["pbcopy"], input=text, text=True, timeout=5)
            else:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text,
                    text=True,
                    timeout=5,
                )
        except Exception:
            pass

    def action_quit(self) -> None:
        """Ctrl+C: 首次提示，2 秒内再次按下才真正退出，防止误触。"""
        now = time.monotonic()
        double_press_window = 2.0  # 两次按键间隔不超过 2 秒才退出

        if (
            self._last_quit_attempt > 0
            and (now - self._last_quit_attempt) < double_press_window
        ):
            # 二次确认通过 → 真正退出
            from app.state.session import save_session

            self.conversation.add_system_info("正在保存会话并退出...")
            if self._session is not None:
                self._session.replace_messages(self._history)
                save_session(self._session)
            from app.infra.background_worker import wait_for_background_tasks

            wait_for_background_tasks()
            self.exit()
            return

        # 首次按下 → 提示用户再按一次
        self._last_quit_attempt = now
        self.conversation.add_system_info(
            "再按一次 Ctrl+C 确认退出（2 秒内有效）"
        )

    # ================================================================
    # 便捷属性，避免每次都写 query_one
    # ================================================================

    @property
    def header(self) -> HeaderWidget:
        return self.query_one(HeaderWidget)

    @property
    def conversation(self) -> ConversationWidget:
        return self.query_one(ConversationWidget)

    @property
    def input_area(self) -> InputArea:
        return self.query_one(InputArea)

    # ================================================================
    # 输入事件处理
    # ================================================================

    def on_input_area_input_submitted(self, event: InputArea.InputSubmitted) -> None:
        """处理用户的输入提交。

        根据输入类型分五种路径：
        1. 审批回复 → _handle_approval_reply()
        2. 退出命令（exit/quit）→ _do_exit()
        3. 显式记忆命令 → 已废弃
        4. /mcp 命令 → mcp_manager.handle_command()
        5. 普通对话 → _run_agent_stream()
        """
        user_text = event.text

        if event.is_approval:
            self._handle_approval_reply(user_text)
            return

        # 退出命令
        if user_text.lower() in ("quit", "exit"):
            self._do_exit()
            return

        # 显式记忆命令已废弃 — memory pipeline 已移除，跳过

        # /mcp 命令（管理 MCP Server），不走模型
        if user_text.strip().startswith("/mcp") and self._mcp_manager is not None:
            result = self._mcp_manager.handle_command(user_text)
            self.conversation.add_user_message(user_text)
            self.conversation.add_system_info(result)
            return

        # 普通对话输入
        self.conversation.add_user_message(user_text)
        self.input_area.disable_input()
        self._run_agent_stream(user_text)

    def _handle_approval_reply(self, answer: str) -> None:
        """处理审批回复（输入框内输入 y/n 或 s/p/n）。

        命令审批: y → 执行, n → 拒绝
        工作目录审批: s → 会话级加入, p → 永久加入, n → 拒绝

        Args:
            answer: "y" | "n" | "s" | "p"
        """
        approval = self._pending_approval
        self.input_area.exit_approval_mode()

        if approval is None:
            self.input_area.enable_input()
            return

        # ── 拒绝处理 ──
        if answer == "n":
            label = "工作目录加入" if approval.approval_type == "workspace_access" else "高风险操作"
            self.conversation.add_system_info(f"用户已拒绝此次{label}请求。")
            self._pending_approval = None
            self.input_area.enable_input()
            return

        # ── 工作目录授权处理 ──
        if approval.approval_type == "workspace_access":
            permanent = answer == "p"
            workspace_path = approval.workspace_path

            if permanent:
                self._tool_context.permanent_workspaces.add(workspace_path)
                self._save_permanent_workspace(workspace_path)
            else:
                self._tool_context.additional_workspaces.add(workspace_path)

            label = "永久" if permanent else "会话级"
            self.conversation.add_system_info(
                f"已将 {workspace_path} 加入工作目录（{label}）。"
            )
        else:
            self._tool_context.approved_actions.add(approval.action_key)

        # 重新执行工具
        self._tool_context.approved_actions.add(approval.action_key)
        result = self._tool_registry.execute_tool(
            tool_name=approval.tool_name,
            input_data=approval.input_data,
            context=self._tool_context,
        )

        self.conversation.add_tool_result(
            name=approval.tool_name,
            summary=f"已执行" if result.ok else f"失败: {result.error}",
            ok=result.ok,
        )

        self._history = self._replace_pending_tool_result(
            history=self._history,
            tool_use_id=approval.tool_use_id,
            tool_name=approval.tool_name,
            content=result.output,
            is_error=not result.ok,
        )
        self._pending_approval = None
        self._run_agent_stream(None)

    @staticmethod
    def _save_permanent_workspace(workspace_path: str) -> None:
        """将工作目录路径持久化追加到 .env 文件的 WORKSPACE_ADDITIONAL_DIRS。

        Args:
            workspace_path: 要持久化的绝对路径
        """
        from pathlib import Path as _Path

        env_path = _Path(".env")
        if not env_path.exists():
            return

        existing = ""
        try:
            existing = env_path.read_text(encoding="utf-8")
        except Exception:
            return

        # 检查是否已存在
        line_prefix = "WORKSPACE_ADDITIONAL_DIRS="
        new_value = workspace_path
        updated = False
        lines = existing.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(line_prefix):
                current = line[len(line_prefix):].strip().strip('"').strip("'")
                existing_paths = [p.strip() for p in current.split(";") if p.strip()]
                if workspace_path not in existing_paths:
                    existing_paths.append(workspace_path)
                    new_value = ";".join(existing_paths)
                else:
                    return  # 已经存在，无需重复添加
                lines[i] = f"{line_prefix}{new_value}"
                updated = True
                break

        if not updated:
            # 没有现有行，追加
            lines.append(f"{line_prefix}{new_value}")

        try:
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass  # 持久化失败不阻塞主流程

    def _on_mcp_async_result(self, result: str) -> None:
        """MCP 异步操作结果回调（由后台线程触发）。

        通过 Textual 的 call_from_thread 安全地将结果写入 Conversation Widget。
        """
        self.call_from_thread(self.conversation.add_system_info, result)

    def _do_exit(self) -> None:
        """安全退出 TUI。

        退出前：
        1. 先清理 MCP 子进程，避免残留进程
        2. 把最新 history 写回 session
        3. 持久化 session 到磁盘
        4. 等待后台任务（长期记忆写入等）完成
        5. 调用 self.exit() 退出 Textual App
        """
        # 先清理 MCP 子进程，避免残留进程
        if self._mcp_manager is not None:
            self._mcp_manager.dispose()

        from app.state.session import save_session

        if self._session is not None:
            self._session.replace_messages(self._history)
            save_session(self._session)
        from app.infra.background_worker import wait_for_background_tasks

        wait_for_background_tasks()
        self.exit()

    # ================================================================
    # Agent 流式执行
    # ================================================================

    @staticmethod
    def _replace_pending_tool_result(
        history: list,
        tool_use_id: str,
        tool_name: str,
        content: str,
        is_error: bool,
    ) -> list:
        """用真实工具结果替换审批前的占位 tool_result。

        这个方法的逻辑与 main.py 中原来的 _replace_pending_tool_result 一致。
        """
        updated = []
        replaced = False
        for msg in history:
            if (
                msg.get("role") == "tool_result"
                and msg.get("tool_use_id") == tool_use_id
                and not replaced
            ):
                updated.append({
                    "role": "tool_result",
                    "tool_use_id": tool_use_id,
                    "tool_name": tool_name,
                    "content": content,
                    "is_error": is_error,
                })
                replaced = True
                continue
            updated.append(msg)
        if not replaced:
            updated.append({
                "role": "tool_result",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "content": content,
                "is_error": is_error,
            })
        return updated

    @staticmethod
    def _update_token_display(header: HeaderWidget, total: int, budget: int) -> None:
        """更新 Header 的 token 显示。

        Args:
            header: HeaderWidget 实例
            total: 当前估算的 token 总数
            budget: 可用上下文预算（来自 DEFAULT_USABLE_CONTEXT_BUDGET）
        """
        header.token_info = f"{total / 1000:.1f}k / {budget / 1000:.0f}k"

    def _run_agent_stream(self, user_input: str | None) -> None:
        """在 worker 线程中执行 stream_agent，实时推送事件到 UI。

        Args:
            user_input: 用户输入文本。
                        为 None 时表示审批后的继续执行（不追加 user 消息，
                        也不重置 turn runtime）。
        """

        def run_in_worker() -> None:
            """worker 线程入口。

            在独立线程中同步调用 stream_agent() 生成器，
            每收到一个事件就通过 self.call_from_thread() 安全推送到 UI 线程。
            """
            # 通知 UI 开始新回复
            self.call_from_thread(self.conversation.begin_agent_response)

            if user_input is not None:
                pass  # memory pipeline removed

            # 捕获最终 AgentStep，用于后续 finalize_turn
            final_step = None

            try:
                # 遍历 stream_agent 生成的每一条事件
                for event in stream_agent(
                    user_input=user_input or "",
                    model=self._model,
                    tool_registry=self._tool_registry,
                    tool_context=self._tool_context,
                    session=self._session,
                    
                    history_summarizer=self._history_summarizer,
                    history=self._history,
                    max_steps=20,
                    session_id=self._session.session_id if self._session else "",
                ):
                    # ---- 根据事件类型分发到对应 UI 更新 ----

                    if isinstance(event, ThinkingEvent):
                        self.call_from_thread(
                            self.conversation.add_thinking, event.text
                        )

                    elif isinstance(event, TextEvent):
                        self.call_from_thread(
                            self.conversation.add_text_chunk, event.text
                        )

                    elif isinstance(event, ToolCallEvent):
                        self.call_from_thread(
                            self.conversation.add_tool_call,
                            event.name,
                            event.args,
                        )

                    elif isinstance(event, ToolResultEvent):
                        self.call_from_thread(
                            self.conversation.add_tool_result,
                            event.name,
                            event.summary,
                            event.ok,
                        )

                    elif isinstance(event, ApprovalEvent):
                        # 审批事件：对话区显示提示，输入区切换审批模式
                        self._pending_approval = event.approval
                        self.call_from_thread(
                            self.conversation.add_approval_prompt,
                            event.approval.message,
                        )
                        if event.approval.approval_type == "workspace_access":
                            self.call_from_thread(
                                self.input_area.enter_workspace_approval_mode,
                            )
                        else:
                            self.call_from_thread(
                                self.input_area.enter_approval_mode,
                            )

                    elif isinstance(event, DoneEvent):
                        # 本轮完成：更新历史、结束流式构建、刷新 token 显示
                        self._history = event.history
                        final_step = event.step  # 保存 step 用于后台记忆写入
                        self.call_from_thread(
                            self.conversation.end_agent_response,
                        )

                        # 更新 Header token 显示
                        try:
                            from app.context.manager import (
                                DEFAULT_USABLE_CONTEXT_BUDGET,
                                estimate_messages_tokens,
                            )
                            total = estimate_messages_tokens(self._history)
                            self.call_from_thread(
                                self._update_token_display,
                                self.header,
                                total,
                                DEFAULT_USABLE_CONTEXT_BUDGET,
                            )
                        except Exception:
                            # token 估算失败不影响主链路
                            pass

                    elif isinstance(event, ErrorEvent):
                        self.call_from_thread(
                            self.conversation.add_error, event.message
                        )

            except Exception as error:
                self.call_from_thread(
                    self.conversation.add_error,
                    f"Agent 执行异常: {error}",
                )

            finally:
                # 无论成功或失败，都恢复输入区
                self.call_from_thread(self.input_area.enable_input)

                # ---- 持久化会话 ----
                from app.state.session import save_session
                if self._session is not None:
                    self._session.replace_messages(self._history)
                    save_session(self._session)

        # 启动 worker 线程，thread=True 表示在独立线程中执行
        self.run_worker(run_in_worker, thread=True, exclusive=True)
