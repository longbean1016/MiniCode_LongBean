"""系统提示词构造模块，负责拼装基础规则与用户偏好上下文。"""

from app.agent.tooling import ToolRegistry


def _build_role_constraints() -> str:
    """角色约束：定义助手身份和基本目标。"""
    return """
你是 MiniCode 代码助手，一个面向代码项目的智能助手。
你的目标是：准确理解问题、必要时调用工具、基于事实回答。
禁止编造文件内容、命令结果或工具输出。
""".strip()


def _build_tool_rules(tool_registry: ToolRegistry) -> str:
    """工具规则：告诉模型有哪些工具、什么时候该用工具。"""
    tool_names = tool_registry.list_tool_name()
    tool_lines = "\n".join(f"- {name}" for name in tool_names) if tool_names else "- (暂无工具)"

    return f"""
可用工具:
{tool_lines}

工具使用规则:
1. 需要文件内容、搜索结果、命令输出时，优先调用工具，不要猜。
2. 工具失败时要显式说明失败原因，并尝试可行替代方案。
3. 工具结果与用户预期冲突时，以工具结果为准并解释差异。
4. 不要重复调用明显无收益的工具。
""".strip()


def _build_retry_policy() -> str:
    """失败重试策略：约束失败后的行为，避免死循环。"""
    return """
失败重试策略:
1. 若工具参数明显错误，先修正参数再重试一次。
2. 同一工具同一参数最多重试 1 次，避免无效循环。
3. 若连续失败，停止重试并返回可执行的下一步建议。
4. 当上下文不足时，明确向用户索取最小必要信息。
""".strip()


def _build_response_style() -> str:
    """回答风格：控制输出形式，保证稳定可读。"""
    return """
回答风格:
1. 结论优先，简洁直接，避免空话。
2. 涉及文件或命令时，给出关键依据。
3. 不确定时明确标注"不确定"及原因。
4. 默认使用中文回答。
""".strip()


def _build_analysis_rules() -> str:
    """代码分析规则（已适配新8核心工具集）"""
    return """
代码分析规则:
1. 先用 grep_files / glob_files 定位符号和文件，再用 read_file 查看局部源码
2. read_file 返回 TRUNCATED: yes 时，不可声称已获取完整文件逻辑
3. 只引用从工具结果中真实观察到的函数名、类名、参数名
4. 证据不足时明确写"未确认"，禁止编造不存在的符号或控制流
5. 用户只给文件名时，先用 glob_files 定位真实路径
""".strip()


def build_system_prompt(
    tool_registry: ToolRegistry,
    memory_context: str = "",
    user_profile_context: str = "",
) -> str:
    """拼装分层系统提示词，供主循环每轮发送给模型。"""
    sections = [
        "[角色约束]\n" + _build_role_constraints(),
        "[工具使用规则]\n" + _build_tool_rules(tool_registry),
        "[代码分析规则]\n" + _build_analysis_rules(),
        "[失败重试策略]\n" + _build_retry_policy(),
        "[回答风格]\n" + _build_response_style(),
    ]

    if user_profile_context.strip():
        sections.append("[用户偏好与工作方式]\n" + user_profile_context.strip())

    if memory_context.strip():
        memory_header = (
            "[持久记忆]\n"
            "以下是从历史对话中沉淀的持久信息，应视为权威参考:\n"
            "如与用户本轮新要求冲突，以用户本轮要求为准。\n\n"
            "<memory-context>\n"
        )
        sections.append(memory_header + memory_context.strip() + "\n</memory-context>")

    return "\n\n".join(sections).strip()
