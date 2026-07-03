"""顶部状态栏 Widget — 显示会话 ID、模型名称、token 用量。

Header 固定 dock=top，高度 1 行，始终可见。
token 信息通过 reactive 属性驱动，外部直接赋值即可触发 UI 刷新。
"""

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static


class HeaderWidget(Static):
    """顶部状态栏组件。

    左侧显示 Agent 名称和会话 ID 前缀，
    右侧显示当前模型和 token 用量。

    Reactive 属性：
    - token_info: 变更时自动触发 watch_token_info 刷新显示
    """

    token_info: reactive[str] = reactive("-- / --")  # type: ignore[assignment]

    def __init__(
        self,
        session_id: str = "",
        model_name: str = "",
    ) -> None:
        """初始化 Header。

        Args:
            session_id: 当前会话 ID，显示前 8 位
            model_name: 模型名称，来自 .env 的 OPENAI_MODEL 配置
        """
        super().__init__()
        # 截取会话 ID 前 8 位，简洁明了
        self._short_session = session_id[:8] if session_id else "new"
        self._model_name = model_name or "unknown"

    def compose(self) -> ComposeResult:
        """布局：单行 Static 文本。"""
        yield Static(
            self._build_text(),
            id="header-content",
        )

    def _build_text(self) -> str:
        """构建 Header 显示文本。

        格式: " MiniCode Agent · session: xxxxxxxx  |  模型: xxx  |  tokens: xxx"
        """
        return (
            f" MiniCode Agent · session: {self._short_session}"
            f"  |  模型: {self._model_name}"
            f"  |  tokens: {self.token_info}"
        )

    def update_model(self, model_name: str) -> None:
        """切换模型后刷新 Header 显示的模型名。

           /model 命令选中新模型后调用此方法更新顶部状态栏。
        """
        self._model_name = model_name or "unknown"
        try:
            content = self.query_one("#header-content", Static)
            content.update(self._build_text())
        except Exception:
            pass

    def watch_token_info(self, value: str) -> None:
        """token_info 变化时自动刷新 Header 显示。

        Textual 的 reactive 机制会在属性赋值后自动调用此方法。
        注意：首次触发可能在 compose() 完成之前（子 Widget 尚未挂载），
        此时 query_one 会失败，需要静默跳过。
        """
        try:
            content = self.query_one("#header-content", Static)
            content.update(self._build_text())
        except Exception:
            # compose 尚未完成，跳过即可；初始值会在 compose 时通过 _build_text 渲染
            pass
