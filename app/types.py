
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
    tool_name: str
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
class ApprovalRequest:
    """"
        表示一次待用户确认的授权请求
    """

    tool_name: str # 哪个工具触发了授予权限
    tool_use_id: str # 本次工具调用id
    action_key: str  # 本次授权的唯一键，用于批准后重试
    message:str # 给终端展示的提示文案
    input_data: Any  # 原始工具输入，批准后用于重试




@dataclass(slots=True)
class AgentStep:
    """
    表示模型执行一步后的统一结果。
    这一步可能是普通回答，也可能是发起工具调用。
    agent loop 会根据 type 决定下一步是继续对话还是执行工具。

    """
    type:Literal["assistant","tool_calls","approval"]
    content: str=""
    kind: Literal["final","progress"]|None=None # 表示这条 assistant 内容属于哪种回答阶段。
    calls: list[ToolCall]=field(default_factory=list) # 本步需要执行的工具调用列表。当 type="tool_calls" 时，这里会有一个或多个 ToolCall。
    content_kind: Literal["progress"]|None=None # 表示内容类型的更细粒度标记。
    diagnostics: StepDiagnostics|None=None #附带的诊断信息。
    approval: ApprovalRequest|None=None  # 只有 type="approval" 时使用


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

    # 生成长期记忆语义向量时使用的 embedding 模型
    embedding_model: str

    # embedding 服务专用 API Key
    embedding_api_key: str

    # embedding 服务专用基础地址
    embedding_base_url: str

    # embedding 向量维度；为 0 时表示不显式传 dimensions
    embedding_dimensions: int

    # agent 允许工作的根目录
    workspace_root: str

    # 是否启用 Qdrant 服务端向量索引
    qdrant_enabled: bool

    # Qdrant 服务端地址，例如 http://localhost:6333
    qdrant_url: str

    # Qdrant API Key，本地无鉴权时可为空
    qdrant_api_key: str

    # 长期记忆向量集合名
    qdrant_collection: str


@dataclass(slots=True)
class ToolResult:
    """
        表示一次工具执行后的结果
    """

    ok: bool  # 是否成功
    output: str  # 给模型看的文本输出（永远是字符串）
    error: str | None = None  # 错误摘要（成功时为 None）
    meta: dict[str, Any] = field(default_factory=dict)  # 附加信息（截断、耗时等）



@dataclass(slots=True)
class ToolContext:
    """
    表示一次工具调用的上下文信息
    """

    cwd: str # 当前工具执行时的工作目录
    approved_actions: set[str] = field(default_factory=set)  # 当前会话内已批准的动作键




