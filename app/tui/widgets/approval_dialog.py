"""审批弹窗 — ModalScreen 实现，用按钮选择代替打字输入。

支持两种审批类型：
- command: 高风险命令确认 (y/n)
- workspace_access: 工作目录加入授权 (s/p/n)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ApprovalDialog(ModalScreen[str]):
    """审批确认弹窗。

    使用 ModalScreen 在对话区上方弹出，用户点击按钮或按快捷键选择。

    返回字符串: "y" | "n" | "s" | "p"
    """

    DEFAULT_CSS = """
    ApprovalDialog {
        align: center middle;
    }

    ApprovalDialog > Container {
        width: 56;
        max-height: 20;
        background: $panel;
        border: thick $primary;
        padding: 1 2;
    }

    ApprovalDialog .dialog-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
        content-align: center middle;
    }

    ApprovalDialog .dialog-message {
        color: $text-muted;
        margin-bottom: 2;
    }

    ApprovalDialog .dialog-buttons {
        width: 100%;
        align-horizontal: center;
    }

    ApprovalDialog .dialog-info {
        color: $warning;
        margin-top: 1;
        content-align: center middle;
    }
    """

    def __init__(
        self,
        message: str,
        approval_type: str = "command",
    ) -> None:
        super().__init__()
        self._message = message
        self._approval_type = approval_type

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("⚠ 权限确认", classes="dialog-title")

            # 消息内容
            for line in self._message.split("\n"):
                if line.strip():
                    yield Static(f"  {line.strip()}", classes="dialog-message")

            # 分隔
            yield Static("")

            # 按钮区
            with Horizontal(classes="dialog-buttons"):
                if self._approval_type == "workspace_access":
                    yield Button("仅本次会话 (S)", id="btn-s", variant="primary")
                    yield Button("永久加入 (P)", id="btn-p", variant="primary")
                    yield Button("拒绝 (N)", id="btn-n", variant="error")
                else:
                    yield Button("执行 (Y)", id="btn-y", variant="primary")
                    yield Button("拒绝 (N)", id="btn-n", variant="error")

            with Container(classes="dialog-info"):
                yield Static("按对应字母键或点击按钮选择", classes="dialog-info")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """按钮点击时关闭弹窗并返回对应值。"""
        if event.button.id == "btn-y":
            self.dismiss("y")
        elif event.button.id == "btn-n":
            self.dismiss("n")
        elif event.button.id == "btn-s":
            self.dismiss("s")
        elif event.button.id == "btn-p":
            self.dismiss("p")

    def on_key(self, event) -> None:
        """键盘快捷键直接选择。"""
        if self._approval_type == "workspace_access":
            key_map = {"s": "s", "p": "p", "n": "n"}
        else:
            key_map = {"y": "y", "n": "n"}

        key = event.key.lower()
        if key in key_map:
            self.dismiss(key_map[key])
            event.stop()
            event.prevent_default()
