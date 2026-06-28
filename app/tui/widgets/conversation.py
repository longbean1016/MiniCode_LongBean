"""对话渲染区 Widget — 流式 Markdown 渲染、代码高亮、工具状态显示。

ConversationWidget 占据终端中间所有可用空间，
负责渲染用户消息、Agent 流式回复、工具调用状态条、
思考过程提示、审批确认信息和错误提示。

所有内容通过 add_* 方法追加，自动滚动到底部。
"""

from textual.containers import VerticalScroll
from textual.widgets import Static
from rich.markdown import Markdown
from rich.text import Text


class ConversationWidget(VerticalScroll):
    """对话渲染区，核心展示组件。

    关键方法：
    - add_user_message(text)        → 用户消息 "▸ ..."
    - begin_agent_response()        → 开始一段 Agent 流式回复
    - add_text_chunk(text)          → 追加流式文本片段（Markdown 渲染）
    - end_agent_response()          → 结束当前流式回复
    - add_thinking(text)            → 灰色思考提示
    - add_tool_call(name, args)     → 黄色工具调用条
    - add_tool_result(name, sum, ok) → 绿色/红色工具结果条
    - add_error(message)            → 红色错误提示
    - add_approval_prompt(message)  → 审批确认提示
    - add_system_info(text)         → 灰色系统信息
    """

    def __init__(self) -> None:
        super().__init__(id="conversation")
        # 当前流式构建中的 Agent 回复内容
        self._current_response = ""
        # 当前流式回复对应的 Static widget 引用
        self._current_static: Static | None = None
        # 已发送消息轮数，用于判断是否需要添加分隔线
        self._turn_count = 0
        # 本轮是否已有过工具调用（用于判断是否需要收起中间消息）
        self._had_tool_calls = False
        # 中间消息是否已被收起
        self._intermediate_collapsed = False

    def on_mount(self) -> None:
        """组件挂载后显示欢迎信息。"""
        self.add_system_info("欢迎使用 MiniCode Agent！输入问题开始对话。")

    def _smart_scroll(self) -> None:
        """智能滚动：仅在用户未手动上翻时才自动滚到底部。

        如果用户正在向上回看历史，就不强制拉回底部。
        """
        # 用户当前离底部不超过 3 行时，才自动跟随到底部
        if self.max_scroll_y - self.scroll_y <= 3:
            self.scroll_end(animate=False)

    # ============================================================
    # 公开方法，由 MiniCodeApp 通过 call_from_thread 调用
    # ============================================================

    def add_user_message(self, text: str) -> None:
        """添加用户消息，在每轮开始前加分隔线以区分不同问题。

        Args:
            text: 用户输入的问题文本
        """
        # 不是第一轮时，在用户消息前加分隔线
        if self._turn_count > 0:
            separator = Text()
            separator.append("─" * 60, style="dim")
            self.mount(Static(separator, classes="turn-separator"))
        self._turn_count += 1

        user_text = Text()
        user_text.append("▸ ", style="bold cyan")
        user_text.append(text, style="white")
        self.mount(Static(user_text, classes="user-message"))
        self._smart_scroll()

    def begin_agent_response(self) -> None:
        """开始一段新的 Agent 回复。

        创建空的 Static 容器，后续流式文本通过 add_text_chunk
        增量追加到这个容器中，实现逐 token 渲染效果。
        """
        self._current_response = ""
        self._current_static = Static("", classes="agent-response")
        self._had_tool_calls = False
        self._intermediate_collapsed = False
        # 先添加 Agent 标签行
        self.mount(Static("Agent", classes="agent-label"))
        self.mount(self._current_static)
        self._smart_scroll()

    def add_text_chunk(self, text: str) -> None:
        """流式追加文本 fragment。

        把新到的文本片段拼接到累积内容中，用 Rich Markdown 重新渲染整体，
        实现"边出字边显示"的效果。

        当模型开始输出最终答案（首个 TextEvent）且本轮已有工具调用时，
        自动收起所有中间消息（thinking、工具调用、工具结果），
        让终端只展示纯文字流式输出。

        Args:
            text: 模型流式输出的一个文本片段
        """
        # 首次文本到达 + 本轮用过工具 → 立刻收起中间消息
        if self._had_tool_calls and not self._intermediate_collapsed:
            self._collapse_intermediate_messages()

        self._current_response += text
        if self._current_static is not None:
            try:
                rendered = Markdown(self._current_response)
                self._current_static.update(rendered)
            except Exception:
                # Markdown 渲染失败时退回纯文本
                self._current_static.update(self._current_response)
        self._smart_scroll()

    def _collapse_intermediate_messages(self) -> None:
        """收起本轮所有中间消息（thinking、工具调用、工具结果）。

        在两种时机触发：
        1. 模型开始输出最终答案时（首个 TextEvent）
        2. Agent 回复完全结束时（end_agent_response）
        """
        for cls in (
            ".thinking-message",
            ".tool-call-message",
            ".tool-result-message",
        ):
            for widget in self.query(cls):
                widget.remove()
        self._intermediate_collapsed = True

    def end_agent_response(self) -> None:
        """结束当前 Agent 回复的流式构建。

        清空内部状态，同时移除本轮所有临时消息（thinking、工具调用状态条），
        只保留 Agent 最终回答文本。
        """
        self._current_response = ""
        self._current_static = None
        self._collapse_intermediate_messages()

    def add_thinking(self, text: str) -> None:
        """添加思考过程提示。

        灰色斜体，简短显示 Agent 当前在做什么（如"正在准备上下文..."）。

        Args:
            text: 处理状态描述文本
        """
        thinking = Text()
        thinking.append("  ", style="")
        thinking.append(text, style="dim italic")
        self.mount(Static(thinking, classes="thinking-message"))
        self._smart_scroll()

    def add_tool_call(self, name: str, args: dict | None = None) -> None:
        """添加工具调用提示条。

        黄色左边框，表示 Agent 准备调用某个工具。

        Args:
            name: 工具名称
            args: 工具调用参数（可选，最多显示前 3 个键值对）
        """
        args_str = ""
        if args:
            flat_items = []
            for i, (k, v) in enumerate(args.items()):
                if i >= 3:
                    flat_items.append("...")
                    break
                flat_items.append(f"{k}={v}")
            args_str = f" ({', '.join(flat_items)})"

        self._had_tool_calls = True
        text = Text()
        text.append(f"  ⚡ 正在调用工具: {name}", style="bold yellow")
        if args_str:
            text.append(args_str, style="yellow")
        self.mount(Static(text, classes="tool-call-message"))
        self._smart_scroll()

    def add_tool_result(self, name: str, summary: str, ok: bool = True) -> None:
        """添加工具执行结果条。

        成功时绿色左边框，失败时红色左边框。

        Args:
            name: 工具名称
            summary: 结果摘要（如 "完成 (1.2KB)"）
            ok: 是否成功
        """
        style = "bold green" if ok else "bold red"
        icon = "✓" if ok else "✗"
        text = Text()
        text.append(f"  {icon} {name}: {summary}", style=style)
        self.mount(Static(text, classes="tool-result-message"))
        self._smart_scroll()

    def add_error(self, message: str) -> None:
        """添加错误提示，红色文字。

        Args:
            message: 错误描述文本
        """
        text = Text()
        text.append(f"  ⚠ {message}", style="bold red")
        self.mount(Static(text, classes="error-message"))
        self._smart_scroll()

    def add_approval_prompt(self, message: str) -> None:
        """显示审批确认提示。

        黄色背景区域，提示用户需要确认高风险操作。

        Args:
            message: 审批提示文本（工具名 + 命令 + 原因）
        """
        text = Text()
        text.append("  ⚠ ", style="bold yellow")
        text.append(message, style="yellow")
        self.mount(Static(text, classes="approval-message"))
        self._smart_scroll()

    def add_system_info(self, text: str) -> None:
        """添加系统信息。

        灰色文字，用于显式命令结果、欢迎信息等非对话内容。

        Args:
            text: 系统信息文本
        """
        info = Text()
        info.append("  ℹ ", style="dim")
        info.append(text, style="dim")
        self.mount(Static(info, classes="system-message"))
        self._smart_scroll()
