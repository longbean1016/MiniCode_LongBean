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

from app.agent_loop import stream_agent
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
            working_memory=working_memory,
            memory_pipeline=memory_pipeline,
            history_summarizer=history_summarizer,
        )
        app.run()  # 接管终端，启动 TUI
    """

    # 键盘绑定：Ctrl+C 和 Ctrl+Q 都触发退出，保存会话后离开
    BINDINGS = [
        ("ctrl+c", "quit", "退出"),
        ("ctrl+q", "quit", "退出"),
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

    .user-message {
        margin: 1 1 0 1;
    }

    .agent-label {
        color: $secondary;
        text-style: bold;
        margin: 1 1 0 1;
    }

    .agent-response {
        margin: 0 1 1 1;
    }

    .thinking-message {
        color: $text-disabled;
        text-style: italic;
        margin: 0 1;
    }

    .tool-call-message {
        color: $warning;
        margin: 0 1;
        padding-left: 1;
        border-left: solid $warning;
    }

    .tool-result-message {
        color: $success;
        margin: 0 1;
        padding-left: 1;
        border-left: solid $success;
    }

    .error-message {
        color: $error;
        margin: 0 1;
        padding-left: 1;
        border-left: solid $error;
    }

    .approval-message {
        color: $warning;
        margin: 1 1;
        padding: 0 1;
        background: $warning 15%;
    }

    .system-message {
        color: $text-disabled;
        margin: 0 1;
    }

    .turn-separator {
        color: $text-disabled;
        margin: 1 1 0 1;
    }
    """

    def __init__(
        self,
        model=None,
        tool_registry=None,
        tool_context=None,
        session=None,
        working_memory=None,
        memory_pipeline=None,
        history_summarizer=None,
    ) -> None:
        """初始化 MiniCodeApp。

        所有业务依赖通过构造函数注入，App 不负责创建这些实例。

        Args:
            model: OpenAIModelAdapter 实例
            tool_registry: ToolRegistry 实例
            tool_context: ToolContext 实例（含 cwd、approved_actions）
            session: SessionData 实例
            working_memory: WorkingMemory 实例
            memory_pipeline: MemoryPipeline 实例
            history_summarizer: OlderHistorySummarizer 实例
        """
        super().__init__()
        self._model = model
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._session = session
        self._working_memory = working_memory
        self._memory_pipeline = memory_pipeline
        self._history_summarizer = history_summarizer
        # 从 session 初始化消息历史
        self._history = list(session.messages) if session else []
        # 待处理的审批请求（ApprovalEvent 到达后暂存，等待用户 y/n）
        self._pending_approval = None

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

    def action_quit(self) -> None:
        """Ctrl+C 或 quit 命令触发的退出动作。

        覆盖 Textual 默认 action_quit，确保退出前：
        1. 把最新 history 写回 session
        2. 持久化到磁盘
        3. 等待后台任务完成
        """
        from app.session import save_session

        if self._session is not None:
            self._session.replace_messages(self._history)
            save_session(self._session)
        from app.background_worker import wait_for_background_tasks

        wait_for_background_tasks()
        self.exit()

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

        根据输入类型分三种路径：
        1. 审批回复（y/n） → _handle_approval_reply()
        2. 退出命令（exit/quit）→ _do_exit()
        3. 显式记忆命令 → memory_pipeline.handle_explicit_input()
        4. 普通对话 → _run_agent_stream()
        """
        user_text = event.text

        if event.is_approval:
            self._handle_approval_reply(user_text)
            return

        # 退出命令
        if user_text.lower() in ("quit", "exit"):
            self._do_exit()
            return

        # 显式记忆命令（/user add、/memory add 等），不走模型
        if self._memory_pipeline is not None:
            explicit_result = self._memory_pipeline.handle_explicit_input(
                user_input=user_text,
                session_id=self._session.session_id,
                history=self._history,
                decay_log_enabled=True,
                decay_log_echo=False,
            )
            if explicit_result.handled:
                self._history = explicit_result.history
                self.conversation.add_user_message(user_text)
                self.conversation.add_system_info(explicit_result.assistant_text)
                return

        # 普通对话输入
        self.conversation.add_user_message(user_text)
        self.input_area.disable_input()
        self._run_agent_stream(user_text)

    def _handle_approval_reply(self, answer: str) -> None:
        """处理审批回复。

        y → 执行已批准的工具，把结果写入 history，继续 agent
        n → 拒绝操作，恢复输入区

        Args:
            answer: "y" 或 "n"
        """
        self.input_area.exit_approval_mode()

        if answer == "y" and self._pending_approval is not None:
            approval = self._pending_approval
            # 先把 action_key 加入已批准集合，再执行工具。
            # 否则工具在执行时还会命中权限检查，返回 PERMISSION_REQUIRED。
            self._tool_context.approved_actions.add(approval.action_key)
            # 执行已批准的工具
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

            # 用真实工具结果替换占位 tool_result
            self._history = self._replace_pending_tool_result(
                history=self._history,
                tool_use_id=approval.tool_use_id,
                tool_name=approval.tool_name,
                content=result.output,
                is_error=not result.ok,
            )
            self._pending_approval = None
            # 继续剩余的 agent 流程（不追加新用户消息）
            self._run_agent_stream(None)
        else:
            # 拒绝高风险操作
            self.conversation.add_system_info("用户已拒绝此次高风险操作。")
            self._pending_approval = None
            self.input_area.enable_input()

    def _do_exit(self) -> None:
        """安全退出 TUI。

        退出前：
        1. 把最新 history 写回 session
        2. 持久化 session 到磁盘
        3. 等待后台任务（长期记忆写入等）完成
        4. 调用 self.exit() 退出 Textual App
        """
        from app.session import save_session

        if self._session is not None:
            self._session.replace_messages(self._history)
            save_session(self._session)
        from app.background_worker import wait_for_background_tasks

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
                # 新请求：重置 turn runtime 并记录用户意图到工作记忆
                if self._memory_pipeline is not None:
                    self._memory_pipeline.reset_turn_runtime(self._working_memory)
                    self._memory_pipeline.remember_user_intent(
                        self._working_memory, user_input
                    )

            try:
                # 遍历 stream_agent 生成的每一条事件
                for event in stream_agent(
                    user_input=user_input or "",
                    model=self._model,
                    tool_registry=self._tool_registry,
                    tool_context=self._tool_context,
                    session=self._session,
                    working_memory=self._working_memory,
                    memory_pipeline=self._memory_pipeline,
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
                        # 审批事件：暂存审批请求，UI 显示确认提示，切审批模式
                        self._pending_approval = event.approval
                        self.call_from_thread(
                            self.conversation.add_approval_prompt,
                            event.approval.message,
                        )
                        self.call_from_thread(
                            self.input_area.enter_approval_mode,
                        )

                    elif isinstance(event, DoneEvent):
                        # 本轮完成：更新历史、结束流式构建、刷新 token 显示
                        self._history = event.history
                        self.call_from_thread(
                            self.conversation.end_agent_response,
                        )

                        # 更新 Header token 显示
                        try:
                            from app.context_manager import (
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

                # ---- 后台写入长期记忆 ----
                from app.background_worker import submit_background

                finalize_task_description = user_input or ""
                finalize_session_id = (
                    self._session.session_id if self._session else ""
                )
                finalize_working_memory = copy.deepcopy(self._working_memory)
                finalize_mp = self._memory_pipeline

                def _finalize_turn_background() -> None:
                    """后台任务：抽取并持久化长期记忆。"""
                    started_at = time.perf_counter()
                    try:
                        if finalize_mp is not None:
                            finalize_mp.finalize_turn(
                                task_description=finalize_task_description,
                                final_step=None,
                                turn_messages=[],
                                session_id=finalize_session_id,
                                working_memory=finalize_working_memory,
                                decay_log_enabled=True,
                                decay_log_echo=False,
                            )
                        elapsed = time.perf_counter() - started_at
                        from app.logger import log_event
                        log_event(
                            f"[session={finalize_session_id}] 长期记忆后台写入完成 "
                            f"耗时={elapsed:.3f}s",
                            echo=False,
                        )
                    except Exception as error:
                        from app.logger import log_event
                        log_event(
                            f"[session={finalize_session_id}] 长期记忆后台写入失败: {error}",
                            echo=False,
                        )

                submit_background(_finalize_turn_background, name="finalize_turn")

                # ---- 持久化会话 ----
                from app.session import save_session
                if self._session is not None:
                    self._session.replace_messages(self._history)
                    save_session(self._session)

        # 启动 worker 线程，thread=True 表示在独立线程中执行
        self.run_worker(run_in_worker, thread=True, exclusive=True)
