"""TUI 事件类型定义 — Agent 向 UI 推送的流式事件。

数据流中，stream_agent() 生成器逐条 yield 这些事件，
MiniCodeApp 的 worker 线程接收后通过 call_from_thread() 推送到对应 Widget 渲染。
"""

from dataclasses import dataclass, field
from typing import Any

from app.types import AgentStep, ApprovalRequest, ChatMessage


@dataclass(slots=True)
class ThinkingEvent:
    """Agent 内部处理步骤，对话区灰色斜体显示。

    例如："正在分析请求..."、"正在准备上下文窗口..."、
    "正在等待模型响应..." 等非对话内容的处理状态。
    """

    text: str


@dataclass(slots=True)
class TextEvent:
    """流式文本 fragment，逐 token 追加到对话区。

    对话区用 Rich Markdown 实时渲染累积后的完整文本，
    每次收到 TextEvent 就增量更新。
    """

    text: str


@dataclass(slots=True)
class ToolCallEvent:
    """模型决定调用工具时的提示。

    对话区显示黄色提示条 "⚡ 正在调用 {name}..."。
    此时工具尚未执行，只是模型表达了调用意图。
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolRunningEvent:
    """工具开始执行。

    对话区更新为 "⏳ {name} 执行中..."，
    表示工具注册表已接收到执行请求。
    """

    name: str


@dataclass(slots=True)
class ToolResultEvent:
    """工具执行完成。

    对话区显示绿色完成条 "✓ {name} 完成 ({summary})"，
    成功时 ok=True 绿色，失败时 ok=False 红色。
    """

    name: str
    summary: str
    ok: bool = True


@dataclass(slots=True)
class UsageEvent:
    """API 返回的 token 用量统计。

    对标准 Claude Code 的 usage 信息展示。
    total_tokens 是本次 API 调用的总 token 消耗。
    """

    total_tokens: int = 0


@dataclass(slots=True)
class ApprovalEvent:
    """高风险操作需要用户确认。

    对话区内嵌审批提示，输入区自动切换为 y/n 模式。
    用户回复 y 后继续执行工具，n 则拒绝。
    """

    approval: ApprovalRequest


@dataclass(slots=True)
class DoneEvent:
    """本轮 Agent 执行完成。

    携带最终 AgentStep 和完整的消息历史。
    输入区从禁用恢复为可用。
    """

    step: AgentStep
    history: list[ChatMessage]


@dataclass(slots=True)
class ErrorEvent:
    """Agent 执行过程中发生错误。

    对话区红色错误提示，TUI 不崩溃。
    详细错误栈写入 debug.log。
    """

    message: str


# 联合类型别名，方便 consumer 做 isinstance 检查后穷尽分支
AgentEvent = (
    ThinkingEvent
    | TextEvent
    | ToolCallEvent
    | ToolRunningEvent
    | ToolResultEvent
    | UsageEvent
    | ApprovalEvent
    | DoneEvent
    | ErrorEvent
)
