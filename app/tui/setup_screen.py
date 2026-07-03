"""首次启动配置向导 — 收集 api_key / base_url 并选择模型后保存到 ~/.bean/settings.json。

   对标 Claude Code 首次启动时的配置流程。
"""

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static


class SetupScreen(Screen):
    """首次配置界面：API Key → Base URL → 模型选择 → 保存。

       用户填写完成后点"开始使用"，配置写入 ~/.bean/settings.json。
    """

    CSS = """
    SetupScreen {
        align: center middle;
    }
    #setup-container {
        width: 55;
        height: auto;
        border: solid $primary;
        padding: 2 3;
        background: $surface;
    }
    #setup-title {
        content-align: center middle;
        padding: 1;
        text-style: bold;
    }
    #setup-form {
        padding: 1 0;
    }
    Label {
        padding: 1 0 0 0;
    }
    Input {
        width: 100%;
        margin: 1 0;
    }
    Select {
        width: 100%;
        margin: 1 0;
    }
    #setup-error {
        color: red;
        padding: 1 0;
        height: 1;
    }
    #setup-btn {
        width: 100%;
        margin: 1 0;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._available_models = self._load_model_list()

    def _load_model_list(self) -> list[tuple[str, str]]:
        """从 model_capabilities 加载可选模型列表。"""
        from app.infra.model_capabilities import _MODEL_CAPABILITIES
        models = []
        for model_id in _MODEL_CAPABILITIES:
            models.append((model_id, model_id))
        return models

    def compose(self) -> ComposeResult:
        """构建配置表单界面。"""
        with Container(id="setup-container"):
            yield Static("MiniCode 首次配置", id="setup-title")

            with VerticalScroll(id="setup-form"):
                yield Label("API Key:")
                yield Input(
                    placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx",
                    id="input-key",
                    password=True,
                )

                yield Label("Base URL:")
                yield Input(
                    placeholder="https://api.deepseek.com",
                    id="input-url",
                    value="https://api.deepseek.com",
                )

                yield Label("选择模型:")
                yield Select(
                    self._available_models,
                    id="select-model",
                    value=self._available_models[0][0] if self._available_models else "",
                )

                yield Static("", id="setup-error")

                yield Button("开始使用", id="setup-btn", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """处理"开始使用"按钮点击。"""
        if event.button.id != "setup-btn":
            return

        api_key = self.query_one("#input-key", Input).value.strip()
        base_url = self.query_one("#input-url", Input).value.strip()
        model = self.query_one("#select-model", Select).value

        # 校验必填项
        if not api_key:
            self.query_one("#setup-error", Static).update("请输入 API Key")
            return

        if not base_url:
            base_url = "https://api.deepseek.com"

        # 保存到 ~/.bean/settings.json
        from app.infra.user_config import ensure_user_config, save_user_config

        config = ensure_user_config()
        config.api_key = api_key
        config.base_url = base_url
        config.model = str(model) if model else "deepseek-v4-flash"
        save_user_config(config)

        # 配置完成，跳转到主界面
        self.dismiss(config)
