"""模型能力配置模块：硬编码上下文窗口大小、最大输出 token 数和压缩阈值计算。
   对标 Claude Code context.ts + configs.ts + autoCompact.ts 的模型能力查询逻辑。

   设计思路（对齐 Claude Code）：
   1. 硬编码已知模型的 context_window 和 max_output_tokens
   2. 模糊匹配：模型名包含 key 就算命中
   3. get_compact_threshold = context_window - max_output_tokens - buffer
   4. buffer 按 Claude Code 分档：<=128k→13k, <=400k→30k, >400k→50k
   5. 提供 estimate_max_turn_growth 做本轮增长预测
"""

from dataclasses import dataclass

# ── 模型能力字典（对标 Claude Code ALL_MODEL_CONFIGS + context.ts 硬编码）──
# 新增模型只需在这里加一行，包含 provider 字段用于按厂商过滤
_MODEL_CAPABILITIES: dict[str, dict[str, int | str]] = {
    # DeepSeek 系列（1M 上下文，API 默认 max_output=32K）
    "deepseek-v4-flash":     {"context_window": 1_048_576, "max_output": 32_768, "provider": "deepseek"},
    "deepseek-v4-pro":       {"context_window": 1_048_576, "max_output": 32_768, "provider": "deepseek"},
    # OpenAI 系列
    "gpt-4o-mini":           {"context_window": 128_000,   "max_output": 16_384, "provider": "openai"},
    "gpt-4o":                {"context_window": 128_000,   "max_output": 16_384, "provider": "openai"},
    "gpt-4-turbo":           {"context_window": 128_000,   "max_output": 4_096,  "provider": "openai"},
    # Claude 系列（通过 OpenAI 兼容接口使用）
    "claude-sonnet-4-6":     {"context_window": 200_000,   "max_output": 32_000, "provider": "anthropic"},
    "claude-opus-4-6":       {"context_window": 200_000,   "max_output": 64_000, "provider": "anthropic"},
    "claude-haiku-4-5":      {"context_window": 200_000,   "max_output": 16_384, "provider": "anthropic"},
}

# ── URL 到厂商映射（对标 Hermes-agent 的 base_url_hostname 检测）──
# 当用户输入 base_url 时，自动识别厂商，只展示该厂商的模型
_PROVIDER_PATTERNS: dict[str, str] = {
    "api.deepseek.com":     "deepseek",
    "deepseek.com":         "deepseek",
    "api.openai.com":       "openai",
    "openai.com":           "openai",
    "api.anthropic.com":    "anthropic",
    "anthropic.com":        "anthropic",
}


def detect_provider(base_url: str) -> str:
    """根据 base_url 自动识别模型厂商。

       匹配规则：URL 中包含已知域名则返回对应厂商名，否则返回 "custom"。
       对标 Hermes-agent 的 base_url_hostname 检测逻辑。
    """
    lowered = base_url.lower()
    for pattern, provider in _PROVIDER_PATTERNS.items():
        if pattern in lowered:
            return provider
    return "custom"


def get_models_for_provider(provider: str) -> list[str]:
    """返回指定厂商的可选模型列表。custom 时返回全部模型。"""
    models = []
    for model_id, caps in _MODEL_CAPABILITIES.items():
        if provider == "custom" or caps.get("provider") == provider:
            models.append(model_id)
    return models if models else list(_MODEL_CAPABILITIES.keys())

# ── 默认值（未知模型使用） ──
_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_MAX_OUTPUT = 8_000

# ── 缓冲分档（对标 Claude Code getAutocompactBufferTokens）──
# buffer 含义：本轮可能产生的工具结果 + 模型输出的预留空间
_BUFFER_SMALL = 13_000   # <= 128K 窗口
_BUFFER_MEDIUM = 30_000  # <= 400K 窗口
_BUFFER_LARGE = 50_000   # > 400K 窗口


@dataclass(slots=True)
class ModelCapability:
    """单个模型的能力配置。"""
    context_window: int      # 最大上下文窗口（token 数）
    max_output: int          # 单次最大输出 token 数
    compact_threshold: int   # 触发 auto compact 的阈值


def get_model_capability(model_name: str) -> ModelCapability:
    """查询模型能力配置。模糊匹配模型名，未知模型返回默认值。

       对标 Claude Code getContextWindowForModel() + getModelMaxOutputTokens()。
    """
    # 转小写后查找命中
    lowered = model_name.lower()
    cap = None
    for key, value in _MODEL_CAPABILITIES.items():
        if key in lowered:
            cap = value
            break

    context_window = cap["context_window"] if cap else _DEFAULT_CONTEXT_WINDOW
    max_output = cap["max_output"] if cap else _DEFAULT_MAX_OUTPUT
    threshold = _compute_compact_threshold(context_window, max_output)

    return ModelCapability(
        context_window=context_window,
        max_output=max_output,
        compact_threshold=threshold,
    )


def get_context_window(model_name: str) -> int:
    """获取模型的最大上下文窗口大小。"""
    return get_model_capability(model_name).context_window


def get_max_output_tokens(model_name: str) -> int:
    """获取模型单次最大输出 token 数。"""
    return get_model_capability(model_name).max_output


def get_compact_threshold(model_name: str) -> int:
    """获取触发 auto compact 的 token 阈值。

       对标 Claude Code getAutoCompactThreshold()：
       threshold = context_window - max_output - buffer。
    """
    return get_model_capability(model_name).compact_threshold


def estimate_max_turn_growth(model_name: str) -> int:
    """预估本轮对话最多能增长多少 token。

       对标 Claude Code estimateMaxTurnGrowth()：
       = max_output_tokens + 15000（工具结果平均增长的保守估计）。
       用于在 API 请求前做预测：如果当前用量 + 预估增长 > 阈值，先压缩再发请求。
    """
    max_output = get_max_output_tokens(model_name)
    tool_result_growth = 15_000  # 对标 Claude Code TOOL_RESULT_GROWTH_ESTIMATE
    return max_output + tool_result_growth


def _compute_compact_threshold(context_window: int, max_output: int) -> int:
    """计算触发阈值 = 上下文窗口 - 最大输出 - 缓冲。

       对标 Claude Code getAutocompactBufferTokens() 分档：
       - 128K 及以下 → 13K 缓冲
       - 400K 及以下 → 30K 缓冲
       - 400K 以上   → 50K 缓冲
    """
    if context_window <= 128_000:
        buffer = _BUFFER_SMALL
    elif context_window <= 400_000:
        buffer = _BUFFER_MEDIUM
    else:
        buffer = _BUFFER_LARGE
    return context_window - max_output - buffer
