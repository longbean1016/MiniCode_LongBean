"""对话渲染区 Widget — 基于 RichLog，彩色 Markdown 渲染 + 工具状态展示。

Rich 级别的行内样式（颜色、粗体、斜体、代码块），
工具调用/结果/思考各有独立颜色，模型输出答案时自动收起中间消息。

复制：Ctrl+Y 静默复制最后回答到剪贴板。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.widgets import RichLog
from rich.markdown import Markdown
from rich.text import Text


@dataclass(slots=True)
class _Entry:
    renderable: Any
    kind: str


class ConversationWidget(RichLog):
    """对话渲染区。

    - 工具调用/结果彩色展示（黄/绿/红/灰）
    - Markdown 渲染（代码块、粗体、斜体）
    - 模型开始回答时自动收起中间消息
    """

    _INTERMEDIATE = frozenset({"thinking", "tool_call", "tool_result"})

    def __init__(self) -> None:
        super().__init__(id="conversation", wrap=True)
        self._entries: list[_Entry] = []
        self._current_response = ""
        self._turn_count = 0
        self._intermediate_collapsed = False
        self._last_agent_response = ""

    def on_mount(self) -> None:
        self.add_system_info("欢迎使用 MiniCode Agent！输入问题开始对话。")

    # ── 渲染 ──────────────────────────────────────────────

    def _render_all(self) -> None:
        self.clear()
        for e in self._entries:
            self.write(e.renderable)
        if self._current_response:
            self.write(Markdown(self._current_response))

    # ── 用户消息 ──────────────────────────────────────────

    def add_user_message(self, text: str) -> None:
        if self._turn_count > 0:
            self._entries.append(
                _Entry(renderable=Text("─" * 60, style="dim"), kind="system")
            )
        self._turn_count += 1
        user = Text()
        user.append("▸ ", style="bold cyan")
        user.append(text, style="white")
        self._entries.append(_Entry(renderable=user, kind="user"))
        self._render_all()

    # ── Agent 回复 ────────────────────────────────────────

    def begin_agent_response(self) -> None:
        self._current_response = ""
        self._intermediate_collapsed = False
        self._entries.append(
            _Entry(renderable=Text("Agent", style="bold $secondary"), kind="agent")
        )
        self._render_all()

    def add_text_chunk(self, text: str) -> None:
        if not self._intermediate_collapsed:
            self._collapse()
        self._current_response += text
        self._render_all()

    def end_agent_response(self) -> None:
        self._last_agent_response = self._current_response
        if self._current_response:
            self._entries.append(
                _Entry(renderable=Markdown(self._current_response), kind="agent")
            )
        self._current_response = ""
        self._render_all()

    def get_last_response(self) -> str:
        return self._last_agent_response

    # ── 收起中间消息 ─────────────────────────────────────

    def _collapse(self) -> None:
        if self._intermediate_collapsed:
            return
        self._entries = [e for e in self._entries if e.kind not in self._INTERMEDIATE]
        self._intermediate_collapsed = True
        self._render_all()

    # ── 工具 / 思考 ───────────────────────────────────────

    def add_thinking(self, text: str) -> None:
        t = Text()
        t.append("  ", style="")
        t.append(text, style="dim italic")
        self._entries.append(_Entry(renderable=t, kind="thinking"))
        self._render_all()

    def add_tool_call(self, name: str, args: dict | None = None) -> None:
        t = Text()
        t.append(f"  ⚡ {name}", style="bold yellow")
        if args:
            flat = []
            for i, (k, v) in enumerate(args.items()):
                if i >= 3:
                    flat.append("...")
                    break
                flat.append(f"{k}={v}")
            t.append(f" ({', '.join(flat)})", style="yellow")
        self._entries.append(_Entry(renderable=t, kind="tool_call"))
        self._render_all()

    def add_tool_result(self, name: str, summary: str, ok: bool = True) -> None:
        style = "bold green" if ok else "bold red"
        icon = "✓" if ok else "✗"
        t = Text()
        t.append(f"  {icon} {name}: {summary}", style=style)
        self._entries.append(_Entry(renderable=t, kind="tool_result"))
        self._render_all()

    # ── 错误 / 审批 / 系统 ────────────────────────────────

    def add_error(self, message: str) -> None:
        t = Text()
        t.append(f"  ⚠ {message}", style="bold red")
        self._entries.append(_Entry(renderable=t, kind="system"))
        self._render_all()

    def add_approval_prompt(self, message: str) -> None:
        t = Text()
        t.append("  ⚠ ", style="bold yellow")
        t.append(message, style="yellow")
        self._entries.append(_Entry(renderable=t, kind="system"))
        self._render_all()

    def add_system_info(self, text: str) -> None:
        t = Text()
        t.append("  ℹ ", style="dim")
        t.append(text, style="dim")
        self._entries.append(_Entry(renderable=t, kind="system"))
        self._render_all()
