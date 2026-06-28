"""底部输入区 Widget — 多行文本输入、发送消息、审批模式切换、命令历史。

InputArea 固定 dock=bottom，两种工作模式：
- normal:   接受任意文本输入，Enter 发送消息，↑↓ 浏览历史
- approval: 只接受 y/n 输入，用于高风险操作确认

发送消息时派发 InputSubmitted 自定义事件给父组件 MiniCodeApp。
"""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, Static
from textual.events import Key


class InputArea(Horizontal):
    """底部输入区组件。

    包含三个子组件：
    - 提示符 "▸ "
    - 文本输入框
    - 快捷键提示行

    Reactive 属性：
    - is_approval_mode: True 时只接受 y/n，False 时正常文本输入

    命令历史：
    - 每次发送的消息自动存入历史
    - ↑ 键回退到上一条历史，↓ 键前进到下一条
    - 最多保留 100 条历史
    """

    is_approval_mode: reactive[bool] = reactive(False)  # type: ignore[assignment]
    approval_mode_type: reactive[str] = reactive("command")  # type: ignore[assignment]

    class InputSubmitted(Message):
        """用户提交了输入的自定义事件。

        Attributes:
            text: 用户输入的文本内容
            is_approval: 是否为审批模式回复
        """

        def __init__(self, text: str, is_approval: bool = False) -> None:
            super().__init__()
            self.text = text
            self.is_approval = is_approval

    def __init__(self) -> None:
        super().__init__()
        # 命令历史列表
        self._history: list[str] = []
        # 当前历史位置，-1 表示在最新的（空）输入行
        self._history_index: int = -1
        # 用户还没按回车时的临时输入，浏览历史时保留
        self._draft_input: str = ""

    def compose(self) -> ComposeResult:
        """布局：提示符 + 输入框 + 快捷键提示行。"""
        yield Static("▸ ", id="input-prompt")
        yield Input(
            placeholder="输入你的问题...",
            id="user-input",
        )
        yield Static(
            "Ctrl+Enter 发送 | ↑↓ 历史",
            id="input-hint",
        )

    def on_mount(self) -> None:
        """组件挂载后自动聚焦输入框。"""
        self.query_one("#user-input", Input).focus()

    def watch_is_approval_mode(self, value: bool) -> None:
        """审批模式切换时更新输入框的外观和提示文本。

        Args:
            value: True 进入审批模式，False 退出
        """
        input_widget = self.query_one("#user-input", Input)
        hint = self.query_one("#input-hint", Static)
        if value:
            if self.approval_mode_type == "workspace_access":
                input_widget.placeholder = "s=仅本次会话 p=永久加入 n=拒绝"
                hint.update("s/p/n 工作目录授权")
            else:
                input_widget.placeholder = "输入 y 执行 或 n 拒绝"
                hint.update("y/n 确认")
            input_widget.value = ""
            self._history_index = -1
            self._draft_input = ""
        else:
            input_widget.placeholder = "输入你的问题..."
            hint.update("Ctrl+Enter 发送 | ↑↓ 历史")
            self.approval_mode_type = "command"

    def on_key(self, event: Key) -> None:
        """拦截 ↑↓ 键实现命令历史浏览。

        只有在 Input 获得焦点时才处理，
        且审批模式下不启用历史浏览。
        """
        input_widget = self.query_one("#user-input", Input)
        if not input_widget.has_focus:
            return
        if self.is_approval_mode:
            return

        if event.key == "up":
            self._navigate_history(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            self._navigate_history(1)
            event.prevent_default()
            event.stop()

    def _navigate_history(self, direction: int) -> None:
        """在历史命令中导航。

        Args:
            direction: -1 表示向上（更旧），1 表示向下（更新）
        """
        input_widget = self.query_one("#user-input", Input)

        # 第一次离开当前输入行时保存草稿
        if self._history_index == -1 and direction < 0:
            self._draft_input = input_widget.value

        # 计算新位置
        new_index = self._history_index + direction

        # 越界检查
        if new_index < -1:
            return
        if new_index >= len(self._history):
            return

        self._history_index = new_index

        # 应用历史
        if self._history_index == -1:
            # 回到草稿
            input_widget.value = self._draft_input
            self._draft_input = ""
            # 光标移到末尾
            input_widget.action_end()
        else:
            # 显示历史条目（从最新到最旧）
            history_entry = self._history[
                -(self._history_index + 1)
            ]
            input_widget.value = history_entry
            # 光标移到末尾
            input_widget.action_end()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """用户按下 Enter 发送消息。

        同时在普通模式下将文本存入命令历史。

        Args:
            event: Textual Input.Submitted 事件
        """
        text = event.value.strip()
        if not text:
            return

        if self.is_approval_mode:
            # 审批模式：根据类型接受不同输入
            valid_responses = (
                ("s", "p", "n")
                if self.approval_mode_type == "workspace_access"
                else ("y", "n")
            )
            if text.lower() in valid_responses:
                self.post_message(
                    self.InputSubmitted(text=text.lower(), is_approval=True)
                )
                self.query_one("#user-input", Input).value = ""
            else:
                # 非法输入，提示用户
                self.query_one("#user-input", Input).value = ""
                hint = "/".join(valid_responses)
                self.query_one("#user-input", Input).placeholder = f"请输入 {hint}"
        else:
            # 普通模式：存入历史，发送消息
            self._add_to_history(text)
            self.post_message(
                self.InputSubmitted(text=text, is_approval=False)
            )
            self.query_one("#user-input", Input).value = ""

        # 重置历史导航状态
        self._history_index = -1
        self._draft_input = ""

    def _add_to_history(self, text: str) -> None:
        """将文本存入命令历史，去重、限制数量。

        Args:
            text: 要存入的命令文本
        """
        # 连续重复的命令不重复存
        if self._history and self._history[-1] == text:
            return
        self._history.append(text)
        # 最多保留 100 条
        if len(self._history) > 100:
            self._history = self._history[-100:]

    def disable_input(self) -> None:
        """禁用输入框（Agent 处理中时调用），防止用户重复提交。"""
        self.query_one("#user-input", Input).disabled = True

    def enable_input(self) -> None:
        """启用输入框（Agent 处理完成后调用），恢复用户交互。"""
        input_widget = self.query_one("#user-input", Input)
        input_widget.disabled = False
        input_widget.focus()

    def enter_approval_mode(self) -> None:
        """进入审批模式（由 MiniCodeApp 在收到 ApprovalEvent 时调用）。"""
        self.is_approval_mode = True

    def enter_workspace_approval_mode(self) -> None:
        """进入工作目录授权审批模式（三选项：s/p/n）。"""
        self.approval_mode_type = "workspace_access"
        self.is_approval_mode = True

    def exit_approval_mode(self) -> None:
        """退出审批模式（审批完成或取消后调用）。"""
        self.is_approval_mode = False
        self.approval_mode_type = "command"
