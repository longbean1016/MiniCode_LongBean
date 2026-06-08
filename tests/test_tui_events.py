"""TUI 事件类型测试 — 验证 AgentEvent 各类型的构造和行为。"""

import pytest
from app.tui.events import (
    ApprovalEvent,
    DoneEvent,
    ErrorEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolRunningEvent,
)
from app.types import AgentStep, ApprovalRequest, ChatMessage


class TestAgentEvents:
    """验证 TUI 事件类型的字段和行为。"""

    def test_thinking_event_creation(self) -> None:
        """ThinkingEvent 应该能正确保存文本内容。"""
        event = ThinkingEvent(text="正在分析请求...")
        assert event.text == "正在分析请求..."

    def test_text_event_creation(self) -> None:
        """TextEvent 应该能正确保存流式文本片段。"""
        event = TextEvent(text="agent_loop.py")
        assert event.text == "agent_loop.py"

    def test_text_event_empty(self) -> None:
        """TextEvent 的空文本也可以创建。"""
        event = TextEvent(text="")
        assert event.text == ""

    def test_tool_call_event_creation(self) -> None:
        """ToolCallEvent 应该保存工具名和参数。"""
        event = ToolCallEvent(
            name="read_file",
            args={"path": "app/main.py"},
        )
        assert event.name == "read_file"
        assert event.args == {"path": "app/main.py"}

    def test_tool_call_event_empty_args(self) -> None:
        """不传 args 时默认为空字典。"""
        event = ToolCallEvent(name="list_files")
        assert event.args == {}

    def test_tool_running_event_creation(self) -> None:
        """ToolRunningEvent 保存工具名。"""
        event = ToolRunningEvent(name="grep_files")
        assert event.name == "grep_files"

    def test_tool_result_event_success(self) -> None:
        """成功的 ToolResultEvent。"""
        event = ToolResultEvent(
            name="read_file",
            summary="完成 (1.2KB)",
            ok=True,
        )
        assert event.name == "read_file"
        assert event.summary == "完成 (1.2KB)"
        assert event.ok is True

    def test_tool_result_event_failure(self) -> None:
        """失败的工具结果 ok=False，red text。"""
        event = ToolResultEvent(
            name="run_command",
            summary="命令执行失败",
            ok=False,
        )
        assert event.ok is False

    def test_tool_result_event_default_ok(self) -> None:
        """不传 ok 参数时默认为 True。"""
        event = ToolResultEvent(name="list_files", summary="done")
        assert event.ok is True

    def test_approval_event_creation(self) -> None:
        """审批事件包含完整的 ApprovalRequest 对象。"""
        approval = ApprovalRequest(
            tool_name="write_file",
            tool_use_id="call_abc123",
            action_key="write_file_/tmp/test.py",
            message="确认写入 /tmp/test.py？",
            input_data={"path": "/tmp/test.py", "content": "hello"},
        )
        event = ApprovalEvent(approval=approval)
        assert event.approval.tool_name == "write_file"
        assert event.approval.action_key == "write_file_/tmp/test.py"

    def test_done_event_creation(self) -> None:
        """DoneEvent 包含 AgentStep 和历史消息列表。"""
        step = AgentStep(type="assistant", content="分析完成", kind="final")
        history: list[ChatMessage] = [
            {"role": "user", "content": "帮我分析"},
            {"role": "assistant", "content": "分析完成"},
        ]
        event = DoneEvent(step=step, history=history)
        assert event.step.type == "assistant"
        assert len(event.history) == 2

    def test_error_event_creation(self) -> None:
        """ErrorEvent 保存错误消息文本。"""
        event = ErrorEvent(message="模型调用超时")
        assert event.message == "模型调用超时"

    def test_multiple_text_events_accumulate(self) -> None:
        """模拟流式场景：多个 TextEvent 拼接后应等于完整内容。"""
        chunks = ["agent", "_loop", ".py", " 是核心模块"]
        events = [TextEvent(text=c) for c in chunks]
        combined = "".join(e.text for e in events)
        assert combined == "agent_loop.py 是核心模块"
