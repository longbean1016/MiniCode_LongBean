"""首次启动配置向导 — 收集 api_key / base_url 并选择模型后保存到 ~/.bean/settings.json。

   对标 Hermes-agent 的交互式配置流程 + Claude Code 的配置存储方式。
   base_url 输入后自动识别厂商，只展示该厂商的可用模型。
"""

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static


class SetupScreen(Screen):
    """首次配置界面。

       用户输入 base_url → 自动检测厂商 → 展示该厂商模型 → 勾选 → 保存。
    """

    CSS = """
    SetupScreen {
        align: center middle;
        background: $surface-darken-1;
    }
    #setup-container {
        width: 58;
        max-height: 90%;
        border: thick $primary;
        padding: 2 3;
        background: $surface;
    }
    #setup-title {
        content-align: center middle;
        padding: 1 2;
        text-style: bold;
        color: $text;
    }
    #setup-subtitle {
        content-align: center middle;
        padding: 0 2 1 2;
        color: $text-disabled;
    }
    Label {
        padding: 1 0 0 0;
        text-style: bold;
    }
    Input {
        width: 100%;
        margin: 1 0;
    }
    #model-list {
        height: auto;
        max-height: 14;
        border: solid $primary-darken-1;
        padding: 0 1;
        margin: 1 0;
    }
    #setup-error {
        color: $error;
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
        self._provider = "deepseek"

    def compose(self) -> ComposeResult:
        """构建配置表单。"""
        with Container(id="setup-container"):
            yield Static("MiniCode 首次配置", id="setup-title")
            yield Static("填写 API Key 和 Base URL，选择模型后开始使用", id="setup-subtitle")

            with VerticalScroll():
                yield Label("API Key")
                yield Input(
                    placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx",
                    id="input-key",
                    password=True,
                )

                yield Label("Base URL（输入后自动识别厂商和模型）")
                yield Input(
                    placeholder="https://api.deepseek.com",
                    id="input-url",
                    value="https://api.deepseek.com",
                )

                yield Label("可用模型（点击选择，默认勾选第一个）")
                yield Static("正在加载...", id="model-status")

                with VerticalScroll(id="model-list"):
                    yield RadioSet(
                        RadioButton("deepseek-v4-flash", id="setup-model-0"),
                        id="model-radios",
                    )

                yield Static("", id="setup-error")
                yield Button("开始使用", id="setup-btn", variant="primary")

    def on_mount(self) -> None:
        """挂载后加载默认模型列表。"""
        self._refresh_models()

    def on_input_changed(self, event: Input.Changed) -> None:
        """base_url 输入变化时刷新模型列表。"""
        if event.input.id == "input-url":
            self._refresh_models()

    def _refresh_models(self) -> None:
        """根据当前 base_url 检测厂商，刷新模型选择列表。"""
        from app.infra.model_capabilities import detect_provider, get_models_for_provider

        url = self.query_one("#input-url", Input).value.strip()
        if not url:
            url = "https://api.deepseek.com"

        self._provider = detect_provider(url)
        models = get_models_for_provider(self._provider)

        # 清除旧选项并挂载新选项
        radio_set = self.query_one("#model-radios", RadioSet)
        for child in list(radio_set.children):
            child.remove()

        for i, model in enumerate(models):
            radio_set.mount(RadioButton(model, id=f"setup-model-{i}", value=(i == 0)))

        # 更新状态提示
        provider_label = self._provider.upper() if self._provider != "custom" else "自定义厂商（展示全部模型）"
        self.query_one("#model-status", Static).update(
            f"厂商: {provider_label}  |  可选模型: {len(models)} 个"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """处理"开始使用"按钮点击。"""
        if event.button.id != "setup-btn":
            return

        api_key = self.query_one("#input-key", Input).value.strip()
        base_url = self.query_one("#input-url", Input).value.strip()

        # 获取选中的模型
        radio_set = self.query_one("#model-radios", RadioSet)
        selected = radio_set.pressed_button
        if selected is None:
            self.query_one("#setup-error", Static).update("请选择一个模型")
            return

        model = str(selected.label) if hasattr(selected, "label") else str(selected)

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
        config.model = model
        save_user_config(config)

        # 配置完成，跳转到主界面
        self.dismiss(config)
