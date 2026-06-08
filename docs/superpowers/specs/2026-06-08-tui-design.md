# MiniCode TUI 改造设计

## 目标

将当前 `input()`/`print()` 纯文本 REPL 改为 Claude Code 风格的终端交互体验，支持流式输出、Markdown 渲染、代码高亮、上下分栏布局。

## 技术选型

使用 **Textual** 框架（基于 Rich 构建，与现有 `rich>=13.7.0` 依赖同源）。

## 终端布局

```
┌──────────────────────────────────────┐
│ MiniCode Agent · session: 81d81acc   │  ← Header 状态栏（顶部固定 1 行）
│ 模型: gpt-4o-mini  tokens: 1.2k/8k  │
├──────────────────────────────────────┤
│                                      │
│ ▸ 帮我分析 agent_loop.py 的主循环     │  ← 对话区（中间填充，可滚动）
│                                      │
│ Agent                                │
│ agent_loop.py 是核心模块...（流式     │
│ Markdown 渲染，代码块语法高亮）        │
│                                      │
│ 💭 思考过程 >>（折叠面板，默认收起）   │
│                                      │
│ ⚡ 正在调用工具: read_file(...)       │  ← 黄色提示条
│ ✓ read_file 完成 (1.2KB, 45行)      │  ← 绿色完成条
│                                      │
│ ⚠ 是否允许执行 write_file？(y/n)     │  ← 审批提示（内嵌，输入区切换 y/n）
│                                      │
├──────────────────────────────────────┤
│ ▸ _                                  │  ← 输入区（底部固定，多行编辑）
│ Ctrl+Enter 发送  Alt+Enter 换行       │
└──────────────────────────────────────┘
```

## 新增模块：`app/tui/`

```
app/tui/
├── __init__.py
├── app.py               # MiniCodeApp(Textual App) 主类
├── events.py            # AgentEvent 类型定义
└── widgets/
    ├── __init__.py
    ├── header.py        # Header Widget — 会话/模型/token 状态
    ├── conversation.py  # Conversation Widget — 流式渲染核心
    └── input_area.py    # InputArea Widget — 底部多行输入
```

### events.py — 事件类型

| 事件类型 | 字段 | 渲染方式 |
|----------|------|----------|
| `ThinkingEvent` | `text: str` | 灰色文字，折叠面板默认收起 |
| `TextEvent` | `text: str` | 流式追加到对话区，Markdown 渲染 |
| `ToolCallEvent` | `name: str, args: dict` | 黄色提示条 "⚡ 正在调用..." |
| `ToolRunningEvent` | `name: str` | 执行中状态提示 |
| `ToolResultEvent` | `name: str, summary: str` | 绿色完成条 "✓ 完成 (1.2KB)" |
| `ApprovalEvent` | `approval: ApprovalRequest` | 内嵌审批提示，输入区切 y/n |
| `DoneEvent` | `step: AgentStep, history: list` | 本轮结束，输入区恢复 |
| `ErrorEvent` | `message: str` | 红色错误提示 |

### app.py — MiniCodeApp

Textual App 主类：

- **compose()** — 组装 Header / Conversation / InputArea 三个 Widget
- **CSS** — `dock: top` 固定 Header，`dock: bottom` 固定 InputArea，Conversation 填充剩余空间
- **on_input_submit(text)** — 用户发送消息时触发，创建 worker 运行 `_run_agent()`
- **_run_agent(user_input)** — `async for event in stream_agent(...)`，根据事件类型分发到对应 Widget 渲染
- 显式命令（`/user add`、`/memory add` 等）在进 `stream_agent` 之前短路处理，结果直接显示在对话区

## 现有模块改动

### main.py — 大幅简化

当前 ~70 行 `while True: input()` 循环替换为：

```python
app = MiniCodeApp(
    model=model,
    tool_registry=tool_registry,
    session=session,
    memory_pipeline=memory_pipeline,
    working_memory=working_memory,
    history_summarizer=history_summarizer,
    ...
)
app.run()
```

依赖组装逻辑不变。

### agent_loop.py — 异步流式生成器

`run_agent_once()` 改为 `async def stream_agent()`：

```python
async def stream_agent(
    user_input: str,
    model: ModelAdapter,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    session: SessionData,
    working_memory: WorkingMemory,
    memory_pipeline: MemoryPipeline,
    history_summarizer: OlderHistorySummarizer,
    history: list[ChatMessage],
    session_id: str,
) -> AsyncIterator[AgentEvent]:
    # yield ThinkingEvent("正在分析请求...")
    # async for chunk in model.stream_chat(messages):
    #     if chunk == tool_call → yield ToolCallEvent
    #     elif chunk == text → yield TextEvent
    # 执行工具 → yield ToolRunningEvent / ToolResultEvent
    # 需要审批 → yield ApprovalEvent
    # 最终 → yield DoneEvent(step, history)
```

核心逻辑（prompt 拼装、工具执行、分析护栏）保持不变，只是从同步阻塞改为异步流式 yield。

### model_registry.py — 新增流式调用

`OpenAIModelAdapter` 新增 `stream_chat()` 方法：

```python
async def stream_chat(
    self, messages: list[ChatMessage]
) -> AsyncIterator[StreamChunk]:
    # 使用 openai SDK 的 stream=True，逐 chunk yield
```

### logger.py — 关闭终端回显

`log_event()` 的 `echo` 默认值从 `True` 改为 `False`。内部日志只写 `debug.log`，终端由 TUI 独占。

## 不改动的模块

- `app/tools/` — 工具注册与执行系统
- `app/memory_pipeline.py`、`app/memory_store.py` 等记忆管线
- `app/context_runtime.py`、`app/context_compactor.py` 等上下文管理
- `app/session.py` — 会话持久化
- `app/config.py` — 配置加载
- `app/message_builder.py`、`app/history_summarizer.py` — prompt 构造
- `app/analysis_guard.py` — 分析护栏
- `app/background_worker.py` — 后台任务（长期记忆写入等）

## 数据流

```
用户键盘输入
    ↓
InputArea Widget → 派发事件
    ↓
MiniCodeApp._run_agent() (worker)
    ↓ async for
agent_loop.stream_agent()
    → yield ThinkingEvent → Conversation 追加灰色思考内容
    → yield TextEvent      → Conversation 流式追加 Markdown
    → yield ToolCallEvent  → Conversation 显示黄色工具提示
    → yield ToolResultEvent→ Conversation 显示绿色完成提示
    → yield ApprovalEvent  → InputArea 切换 y/n 模式
    → yield DoneEvent      → InputArea 恢复正常
    → yield ErrorEvent     → Conversation 显示红色错误
    ↓
Header Widget 实时更新 token 用量 / 状态
```

## 显式命令处理

`/user add`、`/memory add`、`/user set` 等显式命令在 `_run_agent()` 中先调 `memory_pipeline.handle_explicit_input()` 短路处理：

- 是显式命令 → 直接返回结果文本，显示在对话区，不走模型
- 不是显式命令 → 进入 `stream_agent()` 正常流式执行

## 高风险操作审批

采用**对话区内嵌**方式：

1. `stream_agent()` yield `ApprovalEvent`
2. 对话区显示审批提示："⚠ 是否允许执行 write_file？"
3. 输入区自动切换为 y/n 模式，只接受 y/n 输入
4. 用户输入 y → 继续执行工具，结果写回历史
5. 用户输入 n → working_memory 记录拒绝，Agent 继续但不执行该工具
6. 同一会话中同类型操作后续可直接放行（`tool_context.approved_actions` 机制不变）

## 错误处理

| 错误类型 | 终端表现 | 日志 |
|----------|----------|------|
| API / 模型错误 | 对话区红色提示，可继续对话 | debug.log |
| 工具执行失败 | 对话区显示错误摘要 | debug.log |
| 上下文溢出 | 自动压缩，Header 短暂提示 | debug.log |
| TUI 框架内部崩溃 | 安全退出，终端恢复，打印 traceback | debug.log |
| 启动配置错误 | 终端打印错误信息后退出 | - |

核心原则：只要不是启动级致命错误，TUI 保持存活，用户永远可以继续操作。

## 日志分离

- **终端** — 完全由 TUI 占据，只显示用户对话内容、思考过程、工具状态
- **debug.log** — 接收所有内部诊断日志，可另开终端 `tail -f debug.log` 查看

## 会话持久化与恢复

退出方式：

- 输入 `exit` / `quit` → 正常退出，save_session() 落盘
- `Ctrl+C` → Textual 信号处理，save_session() 落盘后退出

恢复方式（与现在一致）：

```bash
python app/main.py --resume latest   # 恢复最近会话
python app/main.py --session <id>     # 恢复指定会话
```

## 开发模式

开发调试时使用 Textual 内置热重载：

```bash
textual run --dev app/main.py
```

保存代码后 App 自动重启，省掉手动 Ctrl+C + 重新敲命令。日常使用仍用 `python app/main.py`。

## 依赖变更

- 已有：`rich>=13.7.0`
- 新增：`textual>=2.0.0`

## 测试策略

- 工具系统、记忆管线、上下文管理等不改模块的现有测试继续通过
- 新增 `tests/test_tui_events.py` — 验证事件类型定义和流式生成器逻辑
- 新增 `tests/test_tui_app.py` — Textual App 基础集成测试（使用 Textual 内置 `pilot` 工具）
