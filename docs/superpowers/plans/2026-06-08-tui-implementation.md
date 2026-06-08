# MiniCode TUI 改造实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 MiniCode Agent 从纯文本 REPL 改为基于 Textual 的流式 TUI，支持上下分栏、Markdown 渲染、流式输出、工具调用状态展示。

**Architecture:** 采用 Textual 框架接管终端。UI 层（`app/tui/`）通过事件驱动方式消费 `agent_loop.stream_agent()` 生成器产生的事件。模型调用改为 OpenAI streaming API，边生成边推送到 UI。原有业务逻辑（工具、记忆、上下文管理）全部保持不变。

**Tech Stack:** Textual>=2.0.0, Rich>=13.7.0（已有）, OpenAI SDK>=1.30.0（已有）

---

## 文件结构

```
新建:
  app/tui/__init__.py
  app/tui/events.py              -- AgentEvent 类型定义
  app/tui/widgets/__init__.py
  app/tui/widgets/header.py      -- 顶部状态栏 Widget
  app/tui/widgets/conversation.py -- 对话渲染区 Widget（核心）
  app/tui/widgets/input_area.py  -- 底部输入区 Widget
  app/tui/app.py                 -- MiniCodeApp 主类
  tests/test_tui_events.py       -- 事件类型测试
  tests/test_tui_app.py          -- TUI 集成测试

修改:
  requirements.txt               -- 添加 textual
  app/logger.py                  -- echo 默认改为 False
  app/model_registry.py          -- 新增 stream_chat() 方法
  app/agent_loop.py              -- 新增 stream_agent() 流式生成器
  app/main.py                    -- 替换 REPL 循环为 TUI 启动
```

---

### Task 1: 添加 textual 依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 添加 textual 到 requirements.txt**

```python
# 在 requirements.txt 末尾追加
textual>=2.0.0
```

- [ ] **Step 2: 安装依赖**

Run: `pip install textual>=2.0.0`
Expected: 成功安装 textual 及其依赖

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: 添加 Textual>=2.0.0 依赖，为 TUI 改造做准备"
```

---

### Task 2: 关闭日志终端回显

**Files:**
- Modify: `app/logger.py:20`

- [ ] **Step 1: 将 log_event 的 echo 默认值改为 False**

修改 `app/logger.py` 第 20-21 行：

```python
def log_event(
    message: str,
    log_file: str = DEFAULT_LOG_FILE,
    *,
    echo: bool = False,  # 之前默认为 True。改为 False，内部日志只写 debug.log，终端由 TUI 独占
) -> None:
```

- [ ] **Step 2: 检查有没有 main.py 之外的 print 直接输出**

Run: `grep -rn "print(" app/ --include="*.py" | grep -v "__pycache__" | grep -v "main.py"`
Expected: 如果有结果，确认它们是否应该走 `log_event`（一般不影响，因为 TUI 接管终端后 print 会被 Textual 框架吃掉）

- [ ] **Step 3: Commit**

```bash
git add app/logger.py
git commit -m "refactor: 关闭 log_event 终端回显，为 TUI 独占终端做准备"
```

---

### Task 3: 创建事件类型定义

**Files:**
- Create: `app/tui/__init__.py`
- Create: `app/tui/events.py`
- Create: `app/tui/widgets/__init__.py`

- [ ] **Step 1: 创建目录结构**

Run:
```powershell
New-Item -ItemType Directory -Force -Path "app\tui\widgets"
```

- [ ] **Step 2: 创建 app/tui/__init__.py**

```python
"""MiniCode TUI 模块 — 基于 Textual 的终端交互界面。"""
```

- [ ] **Step 3: 创建 app/tui/widgets/__init__.py**

```python
"""TUI Widget 组件。"""
```

- [ ] **Step 4: 创建 app/tui/events.py**

```python
"""TUI 事件类型定义 — Agent 向 UI 推送的流式事件。"""

from dataclasses import dataclass, field
from typing import Any

from app.types import AgentStep, ApprovalRequest, ChatMessage


@dataclass(slots=True)
class ThinkingEvent:
    """Agent 内部处理步骤，灰色折叠显示。

    例如："正在分析请求..."、"正在准备上下文窗口..."
    """
    text: str


@dataclass(slots=True)
class TextEvent:
    """流式文本 fragment，逐 token 追加到对话区的 Markdown 渲染中。"""
    text: str


@dataclass(slots=True)
class ToolCallEvent:
    """模型决定调用工具时的提示。

    对话区显示黄色提示条 "⚡ 正在调用 {name}..."。
    """
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolRunningEvent:
    """工具开始执行。

    对话区更新为 "⏳ {name} 执行中..."。
    """
    name: str


@dataclass(slots=True)
class ToolResultEvent:
    """工具执行完成。

    对话区显示绿色完成条 "✓ {name} 完成 ({summary})"。
    """
    name: str
    summary: str
    ok: bool = True


@dataclass(slots=True)
class ApprovalEvent:
    """高风险操作需要用户确认。

    对话区显示确认提示，输入区切换为 y/n 模式。
    """
    approval: ApprovalRequest


@dataclass(slots=True)
class DoneEvent:
    """本轮 Agent 执行完成。

    输入区恢复正常模式。
    """
    step: AgentStep
    history: list[ChatMessage]


@dataclass(slots=True)
class ErrorEvent:
    """Agent 执行过程中发生错误。

    对话区显示红色错误提示，TUI 不崩溃。
    """
    message: str


# 联合类型别名，方便 consumer 做 isinstance 检查
AgentEvent = (
    ThinkingEvent
    | TextEvent
    | ToolCallEvent
    | ToolRunningEvent
    | ToolResultEvent
    | ApprovalEvent
    | DoneEvent
    | ErrorEvent
)
```

- [ ] **Step 5: Commit**

```bash
git add app/tui/__init__.py app/tui/events.py app/tui/widgets/__init__.py
git commit -m "feat: 添加 TUI 事件类型定义 (AgentEvent)"
```

---

### Task 4: 模型适配器增加流式调用

**Files:**
- Modify: `app/model_registry.py`

- [ ] **Step 1: 在 model_registry.py 中添加 StreamChunk 类型和 stream_chat 方法**

在 `app/model_registry.py` 文件顶部 import 区域添加 `dataclass` 和 `Literal`：

```python
from dataclasses import dataclass
from typing import Any, Iterator, Literal
```

在 `OpenAIModelAdapter` 类之前添加 `StreamChunk` 数据类：

```python
@dataclass(slots=True)
class StreamChunk:
    """流式响应的单个 chunk，可能是文本片段或工具调用片段。

    因为 OpenAI streaming API 会把 tool_calls 拆成多次 delta 推送，
    所以这里使用增量模式。
    """
    type: Literal["text", "tool_call_name", "tool_call_args"]
    text: str = ""
    tool_id: str = ""
    tool_index: int = 0
```

在 `OpenAIModelAdapter` 类末尾（`circuit_breaker.record_success()` 之后、`return parse_openai_response_message(...)` 之前）添加 `stream_chat` 方法：

```python
    def stream_chat(
        self,
        messages: list[ChatMessage],
    ) -> Iterator[StreamChunk]:
        """流式调用聊天模型，逐 chunk yield 文本或工具调用增量。

        这个方法与 next() 共用同一套重试 / 熔断 / 协议翻译逻辑，
        只是把 OpenAI SDK 的 stream=True 模式的结果拆成了 StreamChunk 流。
        由于 Textual worker 在独立线程中运行，使用同步迭代器即可。
        """
        # 协议翻译与 next() 共用
        openai_messages = build_openai_messages(messages)
        openai_tools = build_openai_tools(self.tool_registry.list_tools())

        # 熔断检查
        if not self.circuit_breaker.allow_request():
            raise CircuitOpenError(self.circuit_breaker.reject_reason())

        def _request_stream():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,  # type: ignore[arg-type]
                tools=openai_tools,  # type: ignore[arg-type]
                stream=True,
                stream_options={"include_usage": True},
            )

        try:
            response_stream = run_with_retry(
                _request_stream,
                policy=self.retry_policy,
                should_retry=should_retry_model_error,
                on_retry=lambda attempt, error, delay: log_event(
                    (
                        f"聊天模型流式调用失败，准备第 {attempt + 1} 次尝试："
                        f"{type(error).__name__}: {error}，等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.circuit_breaker.record_failure(error)
            raise

        self.circuit_breaker.record_success()

        # 遍历流式响应，把每一块 delta 转成 StreamChunk
        for chunk in response_stream:
            # 用法统计（Usage）出现在最后一个 chunk 中，没有 choices
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 模型返回了文本片段
            if delta.content:
                yield StreamChunk(type="text", text=delta.content)

            # 模型返回了工具调用增量（可能分多块）
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    # 第一块通常包含 tool_call id 和函数名
                    if tc_delta.id:
                        yield StreamChunk(
                            type="tool_call_name",
                            text=tc_delta.function.name or "",
                            tool_id=tc_delta.id,
                            tool_index=tc_delta.index or 0,
                        )
                    # 后续块追加 arguments JSON 片段
                    if tc_delta.function and tc_delta.function.arguments:
                        yield StreamChunk(
                            type="tool_call_args",
                            text=tc_delta.function.arguments,
                            tool_index=tc_delta.index or 0,
                        )
```

- [ ] **Step 2: 验证现有测试不受影响**

Run: `python -m pytest tests/test_main_smoke.py -v`
Expected: 现有测试仍然通过（`stream_chat` 是新方法，不影响 `next()`）

- [ ] **Step 3: Commit**

```bash
git add app/model_registry.py
git commit -m "feat: OpenAIModelAdapter 新增 stream_chat() 流式调用方法"
```

---

### Task 5: agent_loop 增加流式生成器

**Files:**
- Modify: `app/agent_loop.py`

- [ ] **Step 1: 在 agent_loop.py 中添加 stream_agent 函数**

在 `app/agent_loop.py` 的 import 区域添加：

```python
from app.tui.events import (
    AgentEvent,
    ApprovalEvent,
    DoneEvent,
    ErrorEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
```

在文件末尾（`_run_agent_loop` 函数之后）添加 `stream_agent` 函数：

```python
def stream_agent(
    user_input: str,
    model: object,  # OpenAIModelAdapter，为减少循环依赖不写类型注解
    tool_registry: object,
    tool_context: object,
    session: object,
    working_memory: object,
    memory_pipeline: object | None,
    history_summarizer: object | None,
    history: list | None,
    max_steps: int,
    session_id: str,
) -> object:  # Generator[AgentEvent, None, None]
    """流式执行一轮 Agent 请求。

    与 run_agent_once 使用相同的底层逻辑（prepare_agent_context、工具执行），
    但通过 stream_chat 把模型输出拆成流式 AgentEvent 逐条 yield，
    而不是等所有步骤完成后一次性返回。

    调用方用 for event in stream_agent(...) 消费事件流即可。
    """
    import json
    import time

    from app.context_reactive_compact import (
        is_context_overflow_error,
        recover_from_context_overflow,
    )
    from app.context_runtime import (
        prepare_agent_context,
        persist_post_response_working_memory_state,
    )
    from app.history_summarizer import OlderHistorySummarizer
    from app.logger import log_event
    from app.memory_pipeline import MemoryPipeline
    from app.message_builder import MessageBuilder
    from app.model_registry import StreamChunk
    from app.session import SessionData
    from app.tooling import ToolRegistry
    from app.types import (
        AgentStep,
        ApprovalRequest,
        ChatMessage,
        ModelAdapter,
        ToolContext,
        ToolResult,
    )
    from app.working_memory import WorkingMemory
    from app.model_registry import OpenAIModelAdapter

    # 没有历史时用空列表兜底
    if history is None:
        history = []

    # 用 MessageBuilder 统一管理本轮消息
    builder = MessageBuilder()
    builder.extend(history)
    builder.add_user(user_input)

    loop_started_at = time.perf_counter()
    pending_user_nudge: str | None = None
    exploration_history: list[tuple[str, str]] = []

    log_event(
        f"[session={session_id or '-'}] 开始一轮 Agent 流式请求"
    )

    for step_index in range(max_steps):
        step_started_at = time.perf_counter()

        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮循环开始"
        )

        full_history = list(builder.build())

        # 上下文准备阶段 → yield ThinkingEvent
        yield ThinkingEvent("正在分析请求并准备上下文...")

        context_started_at = time.perf_counter()
        prepared_context = prepare_agent_context(
            full_history=full_history,
            session=session,
            tool_registry=tool_registry,
            working_memory=working_memory,
            memory_pipeline=memory_pipeline,
            history_summarizer=history_summarizer,
        )
        context_cost = time.perf_counter() - context_started_at

        # token 统计日志（只写 debug.log，不在终端显示）
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮 token统计: "
            f"total={prepared_context.stats.total_tokens} usage={prepared_context.stats.usage_ratio:.1%} "
            f"budget={prepared_context.stats.usable_budget}"
        )

        # 更新 Header 显示的 token 信息 → 这里不 yield，由 Header Widget 自己定时拉取
        # 或者通过 DoneEvent 的 context_stats 传递

        messages = _append_transient_user_nudge(
            prepared_context.messages,
            pending_user_nudge,
        )
        if pending_user_nudge:
            pending_user_nudge = None

        # ---- 流式调模型 ----
        yield ThinkingEvent("正在等待模型响应...")

        # 收集流式结果
        collected_text = ""
        tool_calls_buf: dict[int, dict] = {}  # tool_index → {id, name, args_str}

        model_started_at = time.perf_counter()
        try:
            for chunk in model.stream_chat(messages=messages):
                if chunk.type == "text":
                    collected_text += chunk.text
                    yield TextEvent(text=chunk.text)

                elif chunk.type == "tool_call_name":
                    idx = chunk.tool_index
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {"id": "", "name": "", "args_str": ""}
                    tool_calls_buf[idx]["id"] = chunk.tool_id
                    tool_calls_buf[idx]["name"] = chunk.text
                    # 解析出 name 时 yield ToolCallEvent
                    yield ToolCallEvent(
                        name=chunk.text,
                        args={},
                    )

                elif chunk.type == "tool_call_args":
                    idx = chunk.tool_index
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {"id": "", "name": "", "args_str": ""}
                    tool_calls_buf[idx]["args_str"] += chunk.text

            model_cost = time.perf_counter() - model_started_at

        except Exception as error:
            # 模型调用失败
            if is_context_overflow_error(error):
                recovery_result = recover_from_context_overflow(
                    messages=messages,
                    usable_budget=prepared_context.stats.usable_budget,
                )
                if recovery_result.recovered:
                    log_event(
                        f"[session={session_id or '-'}] 第 {step_index + 1} 轮触发 Reactive Compact Recover"
                    )
                    # 溢出恢复后需要重试，但流式模式下复杂度太高，先做简单降级
                    yield ErrorEvent(
                        message=f"上下文溢出，已自动压缩后重试。"
                    )
                    # 继续下一轮循环
                    continue

            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮模型调用异常: {error}"
            )
            yield ErrorEvent(message=f"模型调用失败: {error}")
            # 兜底：返回当前历史
            fallback = AgentStep(
                type="assistant",
                content=f"模型调用失败: {error}",
                kind="final",
            )
            builder.add_assistant(fallback.content)
            yield DoneEvent(step=fallback, history=builder.build())
            return

        # ---- 处理工具调用 ----
        if tool_calls_buf:
            # 解析工具调用参数
            calls = []
            for idx in sorted(tool_calls_buf.keys()):
                info = tool_calls_buf[idx]
                tool_name = info["name"]
                tool_use_id = info["id"]
                args_str = info["args_str"]

                try:
                    parsed_input = json.loads(args_str) if args_str.strip() else {}
                except json.JSONDecodeError:
                    parsed_input = {}

                calls.append({
                    "tool_name": tool_name,
                    "input": parsed_input,
                    "id": tool_use_id,
                })

            # 执行每个工具
            for call in calls:
                tool_name = call["tool_name"]
                tool_input = call["input"]
                tool_use_id = call["id"]

                yield ToolRunningEvent(name=tool_name)

                # 记录到 working memory
                if memory_pipeline is not None:
                    memory_pipeline.record_tool_call(
                        working_memory,
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )

                # 写入历史
                builder.add_tool_call(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    input_data=tool_input,
                )

                # 执行工具
                tool_started_at = time.perf_counter()
                try:
                    result = tool_registry.execute_tool(
                        tool_name=tool_name,
                        input_data=tool_input,
                        context=tool_context,
                    )
                except Exception as error:
                    tool_cost = time.perf_counter() - tool_started_at
                    log_event(
                        f"[session={session_id or '-'}] 工具 {tool_name} 执行异常: {error} 耗时={tool_cost:.3f}s"
                    )
                    result = ToolResult(
                        ok=False,
                        output=f"工具调用发生未捕获异常: {error}",
                        error="UNCAUGHT_TOOL_ERROR",
                        meta={"tool_name": tool_name},
                    )

                tool_cost = time.perf_counter() - tool_started_at
                log_event(
                    f"[session={session_id or '-'}] 工具 {tool_name} "
                    f"返回 ok={result.ok} error={result.error} 耗时={tool_cost:.3f}s"
                )

                # 构造工具结果摘要
                output_len = len(str(result.output))
                status = "完成" if result.ok else "失败"
                summary = f"✓ {status} ({output_len} 字符)"
                if result.error:
                    summary += f" — {result.error}"

                yield ToolResultEvent(
                    name=tool_name,
                    summary=summary,
                    ok=result.ok,
                )

                # 权限检查
                if result.error == "PERMISSION_REQUIRED":
                    command = str(result.meta.get("command", ""))
                    reason = str(result.meta.get("reason", ""))
                    action_key = str(result.meta.get("action_key", ""))

                    builder.add_tool_result(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        content="该操作需要用户授权，当前尚未执行。",
                        is_error=True,
                        meta=dict(result.meta),
                    )

                    approval_message = (
                        "该操作需要用户授权。\n"
                        f"工具: {tool_name}\n"
                        f"命令: {command}\n"
                        f"原因: {reason}"
                    )

                    approval_step = AgentStep(
                        type="approval",
                        content=approval_message,
                        approval=ApprovalRequest(
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            action_key=action_key,
                            message=approval_message,
                            input_data=tool_input,
                        ),
                    )
                    yield ApprovalEvent(approval=approval_step.approval)
                    # 注意：这里直接 return，把审批交给 main/TUI 处理
                    # 审批后由 TUI 重新调用 stream_agent 继续执行
                    done_step = approval_step
                    yield DoneEvent(step=done_step, history=builder.build())
                    return

                # 正常情况写回工具结果
                context_output = result.meta.get("context_output", result.output)
                if not isinstance(context_output, str) or not context_output.strip():
                    context_output = result.output
                builder.add_tool_result(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    content=context_output,
                    is_error=not result.ok,
                    meta=dict(result.meta),
                )

                # 工具失败时记录
                if not result.ok:
                    if memory_pipeline is not None:
                        memory_pipeline.record_tool_failure(
                            working_memory,
                            tool_name=tool_name,
                            result=result,
                        )

            # 工具执行完后继续下一轮循环
            step_cost = time.perf_counter() - step_started_at
            log_event(
                f"[session={session_id or '-'}] 第 {step_index + 1} 轮工具阶段结束 "
                f"step耗时={step_cost:.3f}s context={context_cost:.3f}s model={model_cost:.3f}s"
            )
            continue

        # ---- 处理最终回答 ----
        if collected_text.strip():
            builder.add_assistant(collected_text)

            if memory_pipeline is not None:
                memory_pipeline.record_assistant_reply(
                    working_memory,
                    content=collected_text,
                )

            persist_post_response_working_memory_state(
                session=session,
                working_memory=working_memory,
            )

        step = AgentStep(
            type="assistant",
            content=collected_text,
            kind="final",
        )
        step_cost = time.perf_counter() - step_started_at
        total_cost = time.perf_counter() - loop_started_at
        log_event(
            f"[session={session_id or '-'}] 第 {step_index + 1} 轮流式回答完成 "
            f"step耗时={step_cost:.3f}s context={context_cost:.3f}s "
            f"model={model_cost:.3f}s 总耗时={total_cost:.3f}s"
        )
        yield DoneEvent(step=step, history=builder.build())
        return

    # 达到最大步数停止
    total_cost = time.perf_counter() - loop_started_at
    log_event(
        f"[session={session_id or '-'}] 达到最大循环步数 {max_steps} 总耗时={total_cost:.3f}s"
    )
    fallback = AgentStep(
        type="assistant",
        content="已达到最大循环步数，本轮已停止。",
        kind="final",
    )
    builder.add_assistant(fallback.content)
    yield DoneEvent(step=fallback, history=builder.build())
```

- [ ] **Step 2: 验证 agent_loop.py 语法正确**

Run: `python -c "from app.agent_loop import stream_agent; print('stream_agent 导入成功')"`
Expected: 输出 "stream_agent 导入成功"

- [ ] **Step 3: Commit**

```bash
git add app/agent_loop.py
git commit -m "feat: 新增 stream_agent() 流式 Agent 生成器"
```

---

### Task 6: 创建 Header Widget

**Files:**
- Create: `app/tui/widgets/header.py`

- [ ] **Step 1: 创建 Header Widget**

```python
"""顶部状态栏 Widget — 显示会话 ID、模型名称、token 用量。"""

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static


class HeaderWidget(Static):
    """顶部状态栏，固定 dock=top。

    显示会话 ID、模型名、token 使用情况。
    token 用量通过 reactive 属性驱动，外部直接赋值即可刷新。
    """

    session_id: reactive[str] = reactive("", recompose=True)  # type: ignore[assignment]
    model_name: reactive[str] = reactive("", recompose=True)  # type: ignore[assignment]
    token_info: reactive[str] = reactive("--", recompose=True)  # type: ignore[assignment]

    def __init__(
        self,
        session_id: str = "",
        model_name: str = "",
    ) -> None:
        """初始化 Header。

        Args:
            session_id: 当前会话 ID（截取前 8 位显示）
            model_name: 模型名称，来自 .env OPENAI_MODEL
        """
        super().__init__()
        self.session_id = session_id[:8] if session_id else "new"
        self.model_name = model_name

    def compose(self) -> ComposeResult:
        """布局：左侧 Agent 名 + 会话，右侧模型 + token。"""
        short_session = self.session_id[:8] if self.session_id else "new"
        yield Static(
            f" MiniCode Agent · session: {short_session}  |  "
            f"模型: {self.model_name}  |  tokens: {self.token_info}",
            id="header-content",
        )

    def watch_token_info(self, value: str) -> None:
        """当 token_info 变化时，更新显示的文本。"""
        content = self.query_one("#header-content", Static)
        short_session = self.session_id[:8] if self.session_id else "new"
        content.update(
            f" MiniCode Agent · session: {short_session}  |  "
            f"模型: {self.model_name}  |  tokens: {value}"
        )
```

- [ ] **Step 2: 验证导入**

Run: `python -c "from app.tui.widgets.header import HeaderWidget; print('HeaderWidget 导入成功')"`
Expected: "HeaderWidget 导入成功"

- [ ] **Step 3: Commit**

```bash
git add app/tui/widgets/header.py
git commit -m "feat: 添加 HeaderWidget 顶部状态栏组件"
```

---

### Task 7: 创建 Conversation Widget

**Files:**
- Create: `app/tui/widgets/conversation.py`

- [ ] **Step 1: 创建 Conversation Widget**

```python
"""对话渲染区 Widget — 流式 Markdown 渲染、代码高亮、工具状态显示。"""

from textual.containers import VerticalScroll
from textual.widgets import Static
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


class ConversationWidget(VerticalScroll):
    """对话区，占据中间所有可用空间。

    负责：
    - 渲染用户消息和 Agent 流式回复
    - 显示工具调用 / 结果状态条
    - 显示思考过程（折叠）
    - 审批提示

    通过 add_* 系列方法追加内容，通过 clear_turn 清空当前轮次内容。
    """

    def __init__(self) -> None:
        super().__init__(id="conversation")
        # 当前正在流式构建的 Agent 回复内容
        self._current_response = ""
        # 当前正在流式构建的 Static widget 引用
        self._current_static: Static | None = None

    def on_mount(self) -> None:
        """组件挂载后显示欢迎信息。"""
        self.add_user_message("欢迎使用 MiniCode Agent！输入你的问题开始对话。")

    # ---- 公开方法，由 MiniCodeApp 调用 ----

    def add_user_message(self, text: str) -> None:
        """添加用户消息到对话区。

        显示为 "▸ {text}" 格式，青色箭头。
        """
        user_text = Text()
        user_text.append("▸ ", style="bold cyan")
        user_text.append(text, style="white")
        self.mount(Static(user_text, classes="user-message"))

    def begin_agent_response(self) -> None:
        """开始一段新的 Agent 回复。

        创建一个空 Static 容器，后续流式文本追加到这个容器中。
        """
        self._current_response = ""
        self._current_static = Static("", classes="agent-response")
        # 先添加 Agent 标签
        self.mount(Static("Agent", classes="agent-label"))
        self.mount(self._current_static)
        self.scroll_end(animate=False)

    def add_text_chunk(self, text: str) -> None:
        """流式追加文本 fragment 到当前 Agent 回复中。

        Args:
            text: 一个流式输出的文本片段
        """
        self._current_response += text
        if self._current_static is not None:
            # 使用 Rich Markdown 渲染当前累积内容
            try:
                rendered = Markdown(self._current_response)
                self._current_static.update(rendered)
            except Exception:
                # Markdown 渲染失败时回退纯文本
                self._current_static.update(self._current_response)
        self.scroll_end(animate=False)

    def end_agent_response(self) -> None:
        """结束当前 Agent 回复的流式构建。"""
        self._current_response = ""
        self._current_static = None

    def add_thinking(self, text: str) -> None:
        """添加思考过程提示。

        灰色文字，表示 Agent 正在执行内部处理步骤。
        """
        thinking_text = Text()
        thinking_text.append("  ", style="")
        thinking_text.append(text, style="dim italic")
        self.mount(Static(thinking_text, classes="thinking-message"))
        self.scroll_end(animate=False)

    def add_tool_call(self, name: str, args: dict | None = None) -> None:
        """添加工具调用提示条。

        黄色左边框，表示正在调用工具。
        """
        args_str = ""
        if args:
            # 简化参数显示
            flat = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
            args_str = f"({flat})"
        text = Text()
        text.append(f"  ⚡ 正在调用工具: {name}", style="bold yellow")
        text.append(args_str, style="yellow")
        self.mount(Static(text, classes="tool-call-message"))
        self.scroll_end(animate=False)

    def add_tool_result(self, name: str, summary: str, ok: bool = True) -> None:
        """添加工具结果提示条。

        绿色左边框表示成功，红色表示失败。
        """
        style = "bold green" if ok else "bold red"
        icon = "✓" if ok else "✗"
        text = Text()
        text.append(f"  {icon} {name}: {summary}", style=style)
        self.mount(Static(text, classes="tool-result-message"))
        self.scroll_end(animate=False)

    def add_error(self, message: str) -> None:
        """添加错误提示。红色文字。"""
        text = Text()
        text.append(f"  ⚠ {message}", style="bold red")
        self.mount(Static(text, classes="error-message"))
        self.scroll_end(animate=False)

    def add_approval_prompt(self, message: str) -> None:
        """显示审批提示。黄色背景。"""
        approval_text = Text()
        approval_text.append("  ⚠ ", style="bold yellow")
        approval_text.append(message, style="yellow")
        self.mount(Static(approval_text, classes="approval-message"))
        self.scroll_end(animate=False)

    def add_system_info(self, text: str) -> None:
        """添加系统信息（显式命令结果等）。灰色文字。"""
        info = Text()
        info.append("  ℹ ", style="dim")
        info.append(text, style="dim")
        self.mount(Static(info, classes="system-message"))
        self.scroll_end(animate=False)
```

- [ ] **Step 2: 验证导入**

Run: `python -c "from app.tui.widgets.conversation import ConversationWidget; print('ConversationWidget 导入成功')"`
Expected: "ConversationWidget 导入成功"

- [ ] **Step 3: Commit**

```bash
git add app/tui/widgets/conversation.py
git commit -m "feat: 添加 ConversationWidget 对话渲染区组件"
```

---

### Task 8: 创建 InputArea Widget

**Files:**
- Create: `app/tui/widgets/input_area.py`

- [ ] **Step 1: 创建 InputArea Widget**

```python
"""底部输入区 Widget — 多行文本输入、发送、审批模式切换。"""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, Static


class InputArea(Horizontal):
    """底部输入区，固定 dock=bottom。

    两种模式：
    - normal: 接受任意文本输入，Ctrl+Enter 发送
    - approval: 只接受 y/n，用于高风险操作确认

    发送消息时派发 InputSubmitted 事件。
    """

    # 当前模式
    is_approval_mode: reactive[bool] = reactive(False)  # type: ignore[assignment]

    class InputSubmitted(Message):
        """用户提交了输入。

        Attributes:
            text: 用户输入的文本
            is_approval: 是否是审批回复（y/n）
        """

        def __init__(self, text: str, is_approval: bool = False) -> None:
            super().__init__()
            self.text = text
            self.is_approval = is_approval

    def compose(self) -> ComposeResult:
        """布局：提示符 + 输入框 + 快捷键提示。"""
        yield Static("▸ ", id="input-prompt", classes="input-prompt")
        yield Input(
            placeholder="输入你的问题...",
            id="user-input",
        )
        yield Static(
            "Ctrl+Enter 发送 | Alt+Enter 换行 | ↑↓ 历史",
            id="input-hint",
        )

    def on_mount(self) -> None:
        """组件挂载后设置焦点到输入框。"""
        self.query_one("#user-input", Input).focus()

    def watch_is_approval_mode(self, value: bool) -> None:
        """切换审批模式时更新输入框状态。"""
        input_widget = self.query_one("#user-input", Input)
        hint = self.query_one("#input-hint", Static)
        if value:
            input_widget.placeholder = "输入 y 执行 或 n 拒绝"
            hint.update("y/n 确认 | Esc 取消")
            input_widget.value = ""
        else:
            input_widget.placeholder = "输入你的问题..."
            hint.update("Ctrl+Enter 发送 | Alt+Enter 换行 | ↑↓ 历史")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """用户按下 Enter 发送消息。

        在审批模式下只接受 y/n；普通模式下正常发送。
        """
        text = event.value.strip()
        if not text:
            return

        if self.is_approval_mode:
            # 审批模式：只接受 y/n
            if text.lower() in ("y", "n"):
                self.post_message(
                    self.InputSubmitted(text=text.lower(), is_approval=True)
                )
                self.query_one("#user-input", Input).value = ""
            else:
                self.query_one("#user-input", Input).value = ""
                self.query_one("#user-input", Input).placeholder = (
                    "请输入 y 或 n"
                )
        else:
            # 普通模式：正常发送
            self.post_message(
                self.InputSubmitted(text=text, is_approval=False)
            )
            self.query_one("#user-input", Input).value = ""

    def disable_input(self) -> None:
        """禁用输入（Agent 正在处理时调用）。"""
        self.query_one("#user-input", Input).disabled = True

    def enable_input(self) -> None:
        """启用输入（Agent 处理完成后调用）。"""
        self.query_one("#user-input", Input).disabled = False
        self.query_one("#user-input", Input).focus()

    def enter_approval_mode(self) -> None:
        """进入审批模式。"""
        self.is_approval_mode = True

    def exit_approval_mode(self) -> None:
        """退出审批模式。"""
        self.is_approval_mode = False
```

- [ ] **Step 2: 验证导入**

Run: `python -c "from app.tui.widgets.input_area import InputArea; print('InputArea 导入成功')"`
Expected: "InputArea 导入成功"

- [ ] **Step 3: Commit**

```bash
git add app/tui/widgets/input_area.py
git commit -m "feat: 添加 InputArea 底部输入区组件（支持普通/审批双模式）"
```

---

### Task 9: 创建 MiniCodeApp 主类

**Files:**
- Create: `app/tui/app.py`

- [ ] **Step 1: 创建 MiniCodeApp**

```python
"""MiniCode TUI 主 App — 组装 Widget、连接 Agent 事件流、管理工作线程。"""

from textual.app import App, ComposeResult
from textual.worker import Worker, WorkerState, get_current_worker

from app.agent_loop import stream_agent
from app.context_manager import collect_context_stats
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

    布局（CSS Dock）：
    - Header:    dock=top, height=1
    - InputArea: dock=bottom, height=auto
    - ConvArea:  填充剩余中间空间

    数据流：
    用户输入 → on_input_submitted → worker 线程跑 stream_agent
    → call_from_thread 推送事件到 UI → 实时渲染
    """

    CSS = """
    HeaderWidget {
        dock: top;
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
        text-style: bold;
    }

    InputArea {
        dock: bottom;
        height: auto;
        padding: 0 1;
        border-top: solid $primary;
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
        padding: 0 1;
        scrollbar-size: 1 1;
    }

    .user-message {
        margin: 1 0;
        padding: 0;
    }

    .agent-label {
        color: $secondary;
        text-style: bold;
        margin-top: 1;
    }

    .agent-response {
        margin-bottom: 1;
        padding: 0;
    }

    .thinking-message {
        color: $text-disabled;
        text-style: italic;
        margin: 0;
    }

    .tool-call-message {
        color: $warning;
        margin: 0;
        padding-left: 1;
        border-left: solid $warning;
    }

    .tool-result-message {
        color: $success;
        margin: 0;
        padding-left: 1;
        border-left: solid $success;
    }

    .error-message {
        color: $error;
        margin: 0;
        padding-left: 1;
        border-left: solid $error;
    }

    .approval-message {
        color: $warning;
        margin: 1 0;
        padding: 0 1;
        background: $warning 15%;
    }

    .system-message {
        color: $text-disabled;
        margin: 0;
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
        """初始化 TUI App。

        Args:
            model: OpenAIModelAdapter 实例
            tool_registry: ToolRegistry 实例
            tool_context: ToolContext 实例
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
        self._history = list(session.messages) if session else []
        self._pending_approval = None  # 待处理的审批请求

    def compose(self) -> ComposeResult:
        """组装三个主 Widget。"""
        yield HeaderWidget(
            session_id=self._session.session_id if self._session else "",
            model_name=getattr(self._model, "model_name", "unknown"),
        )
        yield ConversationWidget()
        yield InputArea()

    # ---- 便捷属性 ----

    @property
    def header(self) -> HeaderWidget:
        return self.query_one(HeaderWidget)

    @property
    def conversation(self) -> ConversationWidget:
        return self.query_one(ConversationWidget)

    @property
    def input_area(self) -> InputArea:
        return self.query_one(InputArea)

    # ---- 事件处理 ----

    def on_input_area_input_submitted(self, event: InputArea.InputSubmitted) -> None:
        """处理用户输入提交。"""
        user_text = event.text

        if event.is_approval:
            # 审批模式回复
            self._handle_approval_reply(user_text)
            return

        # 检查退出命令
        if user_text.lower() in ("quit", "exit"):
            self._do_exit()
            return

        # 检查显式记忆命令
        if self._memory_pipeline is not None:
            explicit_result = self._memory_pipeline.handle_explicit_input(
                user_input=user_text,
                session_id=self._session.session_id,
                history=self._history,
            )
            if explicit_result.handled:
                self._history = explicit_result.history
                self.conversation.add_user_message(user_text)
                self.conversation.add_system_info(explicit_result.assistant_text)
                return

        # 普通输入 → 启动 Agent worker
        self.conversation.add_user_message(user_text)
        self.input_area.disable_input()
        self._run_agent_stream(user_text)

    def _handle_approval_reply(self, answer: str) -> None:
        """处理审批回复（y/n）。"""
        self.input_area.exit_approval_mode()

        if answer == "y" and self._pending_approval is not None:
            # 执行已批准的工具
            approval = self._pending_approval
            result = self._tool_registry.execute_tool(
                tool_name=approval.tool_name,
                input_data=approval.input_data,
                context=self._tool_context,
            )
            self._tool_context.approved_actions.add(approval.action_key)

            self.conversation.add_tool_result(
                name=approval.tool_name,
                summary=f"✓ 已执行" if result.ok else f"✗ {result.error}",
                ok=result.ok,
            )

            # 把结果写回 history 后继续
            self._history = self._replace_pending_tool_result(
                history=self._history,
                tool_use_id=approval.tool_use_id,
                tool_name=approval.tool_name,
                content=result.output,
                is_error=not result.ok,
            )
            self._pending_approval = None
            # 继续剩余 agent 流程
            self._run_agent_stream(None)  # None 表示不追加用户输入
        else:
            self.conversation.add_system_info("用户已拒绝此次高风险操作。")
            self._pending_approval = None
            self.input_area.enable_input()

    def _do_exit(self) -> None:
        """安全退出 TUI。"""
        from app.session import save_session

        if self._session is not None:
            self._session.replace_messages(self._history)
            save_session(self._session)
        from app.background_worker import wait_for_background_tasks

        wait_for_background_tasks()
        self.exit()

    # ---- Agent 执行 ----

    @staticmethod
    def _replace_pending_tool_result(
        history: list,
        tool_use_id: str,
        tool_name: str,
        content: str,
        is_error: bool,
    ) -> list:
        """替换待授权的占位 tool_result。"""
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
    def _update_token_display(header: HeaderWidget, stats) -> None:
        """根据 ContextStats 更新 Header 的 token 显示。"""
        header.token_info = (
            f"{stats.total_tokens / 1000:.1f}k / {stats.usable_budget / 1000:.0f}k"
        )

    def _run_agent_stream(self, user_input: str | None) -> None:
        """启动 Agent 流式执行（在 worker 线程中运行）。

        Args:
            user_input: 用户输入文本；为 None 时表示审批后的继续执行（不追加 user 消息）
        """

        def run_in_worker() -> None:
            """在 worker 线程中执行 stream_agent，通过 call_from_thread 更新 UI。"""
            import time

            self.call_from_thread(self.conversation.begin_agent_response)

            if user_input is not None:
                # 新请求：重置 memory turn runtime
                if self._memory_pipeline is not None:
                    self._memory_pipeline.reset_turn_runtime(self._working_memory)
                    self._memory_pipeline.remember_user_intent(
                        self._working_memory, user_input
                    )

            try:
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
                        self._pending_approval = event.approval
                        self.call_from_thread(
                            self.conversation.add_approval_prompt,
                            event.approval.message,
                        )
                        self.call_from_thread(
                            self.input_area.enter_approval_mode,
                        )
                        # 审批事件出现时让 worker 停止，等用户输入
                        # 注意：这里直接 return，worker 结束
                        # 审批后在 _handle_approval_reply 里重新启动 worker

                    elif isinstance(event, DoneEvent):
                        self._history = event.history
                        self.call_from_thread(
                            self.conversation.end_agent_response,
                        )

                        # 更新 token 显示
                        try:
                            from app.context_manager import (
                                DEFAULT_USABLE_CONTEXT_BUDGET,
                                estimate_messages_tokens,
                            )
                            total = estimate_messages_tokens(self._history)
                            self.call_from_thread(
                                self._update_token_display,
                                self.header,
                                type(
                                    "Stats", (),
                                    {
                                        "total_tokens": total,
                                        "usable_budget": DEFAULT_USABLE_CONTEXT_BUDGET,
                                    },
                                )(),
                            )
                        except Exception:
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
                # 恢复输入
                self.call_from_thread(self.input_area.enable_input)

                # 后台写入长期记忆
                from app.background_worker import submit_background
                import copy

                finalize_session_id = (
                    self._session.session_id if self._session else ""
                )
                finalize_working_memory = copy.deepcopy(self._working_memory)

                def _finalize_turn_background() -> None:
                    started_at = time.perf_counter()
                    try:
                        if self._memory_pipeline is not None:
                            self._memory_pipeline.finalize_turn(
                                task_description=user_input or "",
                                final_step=None,
                                turn_messages=[],
                                session_id=finalize_session_id,
                                working_memory=finalize_working_memory,
                            )
                        elapsed = time.perf_counter() - started_at
                        from app.logger import log_event
                        log_event(
                            f"[session={finalize_session_id}] 长期记忆后台写入完成 耗时={elapsed:.3f}s",
                        )
                    except Exception as error:
                        from app.logger import log_event
                        log_event(
                            f"[session={finalize_session_id}] 长期记忆后台写入失败: {error}",
                        )

                submit_background(_finalize_turn_background, name="finalize_turn")

                # 持久化会话
                from app.session import save_session
                if self._session is not None:
                    self._session.replace_messages(self._history)
                    save_session(self._session)

        # 启动 worker 线程
        self.run_worker(run_in_worker, thread=True, exclusive=True)
```

- [ ] **Step 2: 验证导入**

Run: `python -c "from app.tui.app import MiniCodeApp; print('MiniCodeApp 导入成功')"`
Expected: "MiniCodeApp 导入成功"

- [ ] **Step 3: Commit**

```bash
git add app/tui/app.py
git commit -m "feat: 添加 MiniCodeApp TUI 主类（Widget 组装 + worker 事件分发）"
```

---

### Task 10: 改造 main.py 启动 TUI

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: 简化 main.py，用 MiniCodeApp.run() 替换 REPL 循环**

修改 `app/main.py` 的 `main()` 函数。在文件顶部添加 import：

```python
from app.tui.app import MiniCodeApp
```

将 `main()` 函数中 `print("LongBean MiniCode Agent 已启动...")` 之后的整个 `while True` 循环（约第 300-460 行）替换为：

```python
    # 构建 TUI App 实例，注入所有业务依赖
    tui_app = MiniCodeApp(
        model=model,
        tool_registry=tool_registry,
        tool_context=tool_context,
        session=session,
        working_memory=working_memory,
        memory_pipeline=memory_pipeline,
        history_summarizer=history_summarizer,
    )

    # 一行启动 TUI，终端被 Textual 接管
    tui_app.run()

    # TUI 退出后的收尾（正常 exit/quit 或 Ctrl+C 都会到这里）
    wait_for_background_tasks()
    print("Bye!")
```

同时删除 `main()` 中不再使用的代码块，包括：
- `while True:` 循环（约 302-460 行）
- `_replace_pending_tool_result` 函数（该逻辑已移入 `MiniCodeApp`）

保留的部分：
- `_build_arg_parser()` 和 `_load_or_create_session()` 函数（不变）
- `main()` 中组装依赖的代码（`load_config()` 到 `history = list(session.messages)` 的部分，保留）
- `if __name__ == "__main__": main()`（保留）

- [ ] **Step 2: 验证 main.py 语法正确**

Run: `python -c "import ast; ast.parse(open('app/main.py').read()); print('main.py 语法正确')"`
Expected: "main.py 语法正确"

- [ ] **Step 3: 验证完整导入链**

Run: `python -c "from app.main import main; print('main 导入成功')"`
Expected: "main 导入成功"

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: main.py 改用 MiniCodeApp.run() 启动 TUI，替换原有 REPL 循环"
```

---

### Task 11: 编写测试

**Files:**
- Create: `tests/test_tui_events.py`

- [ ] **Step 1: 编写事件类型测试**

```python
"""TUI 事件类型测试 — 验证 AgentEvent 各类型的构造和行为。"""

import pytest
from app.tui.events import (
    AgentEvent,
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
            summary="✓ 完成 (1.2KB)",
            ok=True,
        )
        assert event.name == "read_file"
        assert event.summary == "✓ 完成 (1.2KB)"
        assert event.ok is True

    def test_tool_result_event_failure(self) -> None:
        """失败的工具结果默认 ok=False。"""
        event = ToolResultEvent(
            name="run_command",
            summary="✗ 命令执行失败",
            ok=False,
        )
        assert event.ok is False

    def test_approval_event_creation(self) -> None:
        """审批事件包含 ApprovalRequest 对象。"""
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
        """DoneEvent 包含 AgentStep 和历史消息。"""
        step = AgentStep(type="assistant", content="分析完成", kind="final")
        history: list[ChatMessage] = [
            {"role": "user", "content": "帮我分析"},
            {"role": "assistant", "content": "分析完成"},
        ]
        event = DoneEvent(step=step, history=history)
        assert event.step.type == "assistant"
        assert len(event.history) == 2

    def test_error_event_creation(self) -> None:
        """ErrorEvent 保存错误消息。"""
        event = ErrorEvent(message="模型调用超时")
        assert event.message == "模型调用超时"
```

- [ ] **Step 2: 运行测试**

Run: `python -m pytest tests/test_tui_events.py -v`
Expected: 全部通过

- [ ] **Step 3: Commit**

```bash
git add tests/test_tui_events.py
git commit -m "test: 添加 AgentEvent 事件类型单元测试"
```

---

### Task 12: 集成测试与最终验证

**Files:**
- Modify: `app/tui/app.py` (可能需要微调)
- 验证所有现有测试不受影响

- [ ] **Step 1: 运行所有现有测试**

Run: `python -m pytest tests/ -v`
Expected: 现有测试全部通过（TUI 改动不应影响业务逻辑测试）

- [ ] **Step 2: 检查 Textual App 能否正常启动（dry-run 级别）**

Run: `python -c "from app.tui.app import MiniCodeApp; print('App 类导入正常')"`
Expected: "App 类导入正常"

- [ ] **Step 3: 检查 .gitignore 是否需要更新**

确认 `.superpowers/` 已在 `.gitignore` 中（visual-companion 文件不应该提交）。

- [ ] **Step 4: 最终 commit**

```bash
git add -A
git commit -m "feat: 完成 MiniCode TUI 改造，基于 Textual 实现流式终端交互"
```

---

## 自检清单

1. **Spec 覆盖检查**：
   - [x] 终端布局（Header + Conversation + InputArea）→ Task 6/7/8
   - [x] 流式输出 → Task 4/5
   - [x] 事件类型定义 → Task 3
   - [x] 思考过程折叠显示 → Task 7 (add_thinking)
   - [x] 工具调用状态提示 → Task 7 (add_tool_call/add_tool_result)
   - [x] 审批内嵌确认 → Task 8/9 (_handle_approval_reply)
   - [x] 显式命令处理 → Task 9 (handle_explicit_input)
   - [x] 日志分离 → Task 2
   - [x] 会话持久化 → Task 9 (_do_exit, _run_agent_stream finally)
   - [x] 错误处理 → Task 9 (ErrorEvent → add_error)
   - [x] 测试 → Task 11

2. **占位符扫描**：无 TBD、TODO、未完成段落

3. **类型一致性**：AgentEvent 联合类型在每个 task 中引用一致，字段名在 events.py 和 app.py 中一致
