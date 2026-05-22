
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, TypedDict


class ChatMessage(TypedDict,total=False):
    """
    表示一条对话消息，是主循环里最核心的消息结构。
    不同 role 的消息字段可以不同，所以使用 total=False 的 TypedDict。
    """

    role:Literal[
        "system",  # 系统消息
        "user",    # 用户消息
        "assistant",# 模型的普通回答
        "assistant_progress", # 模型的中间进度说明
        "assistant_tool_call", # 模型发起工具调用时对应的消息
        "tool_result", # 工具执行后的结果消息
    ]
    content: str # 这条消息的文本内容。
    tool_use_id: str # 这次工具调用的唯一标识。
    tool_name: str # 这次工具调用的工具名称。
    input: Any # 工具调用时的输入参数。
    is_error: bool # 表示这条消息是不是错误结果。

class ToolCall(TypedDict):
    """
    表示一次工具调用请求。
    模型在某一步决定调用工具时，会产出这个结构，随后由 agent loop 执行。
    """

    id: str
    toolName: str
    input: Any # 调用该工具时传入的参数。

@dataclass(slots=True)
class StepDiagnostics:
    """
    保存当前 agent step 的诊断信息，
    用于说明模型为什么停下、是否遇到阻断类型、哪些阻断被忽略。
    """

    stopReason: str|None=None  # 记录这一步为什么停止。
    block_types: list[str]=field(default_factory=list) # 记录当前这一步遇到了哪些“阻断类型”。
    ignored_block_types: list[str]=field(default_factory=list) # 记录哪些阻断类型被系统忽略了，没有真的阻止执行。


@dataclass(slots=True)
class AgentStep:
    """
    表示模型执行一步后的统一结果。
    这一步可能是普通回答，也可能是发起工具调用。
    agent loop 会根据 type 决定下一步是继续对话还是执行工具。

    """
    type:Literal["assistant","tool_calls"]
    content: str=""
    kind: Literal["final","progress"]|None=None # 表示这条 assistant 内容属于哪种回答阶段。
    calls: list[ToolCall]=field(default_factory=list) # 本步需要执行的工具调用列表。当 type="tool_calls" 时，这里会有一个或多个 ToolCall。
    contentKind: Literal["progress"]|None=None # 表示内容类型的更细粒度标记。
    diagnostics: StepDiagnostics|None=None #附带的诊断信息。


class ModelAdapter(Protocol):

    """
    定义模型适配器的统一接口。
    不同模型提供商只要实现 next()，就能接入 agent loop。
    返回值统一为 AgentStep，便于主循环处理。
    """

    def next(
        self,
        messages: list[ChatMessage], #当前完整消息历史。
        on_stream_chunk: Callable[[str], None] | None = None, #流式输出回调函数。
        store: Any | None = None, #额外的全局状态容器。
    ) -> AgentStep: ...


@dataclass(slots=True)
class AppConfig:
    # 模型服务的 API Key
    api_key: str

    # 模型服务的基础地址
    # 例如 OpenAI: https://api.openai.com/v1
    # 例如 DeepSeek: https://api.deepseek.com
    base_url: str

    # 当前使用的模型名称
    model: str

    # agent 允许工作的根目录
    workspace_root: str
