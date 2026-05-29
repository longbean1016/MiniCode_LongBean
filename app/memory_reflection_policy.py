from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.types import AgentStep


# 这些目录更像运行产物、会话数据或临时实验目录，
# 不应单独作为自动 project memory 的“项目证据”。
_NON_PROJECT_PATH_MARKERS = (
    "tmp\\",
    "tmp/",
    ".memory\\",
    ".memory/",
    ".sessions\\",
    ".sessions/",
    "__pycache__\\",
    "__pycache__/",
)

# 这些词更像“稳定项目经验”的语言信号，
# 常出现在约定、修复结论、风险规避、实现收敛里。
_PERSISTENT_PROJECT_MARKERS = (
    "以后",
    "统一",
    "改为",
    "保持",
    "必须",
    "默认",
    "只允许",
    "不再",
    "约定",
    "约束",
    "规范",
    "修复",
    "结论",
    "风险",
    "失败经验",
    "避免",
    "保留",
    "固定为",
    "switch to",
    "change to",
    "replace with",
    "use only",
    "no longer",
    "must",
    "should",
    "conclusion",
    "fix",
)

# 这些结构更像真实仓库执行痕迹，而不是普通闲聊或算法问答。
_CODE_OR_REPO_PATTERNS = (
    r"`[^`]+`",  # 反引号里的路径、函数名、配置项
    r"\b[a-zA-Z0-9_\-]+\.(py|md|toml|json|yaml|yml|txt|js|ts|tsx|jsx)\b",
    r"\b[a-zA-Z_][a-zA-Z0-9_]*\([^()\n]*\)",
    r"\b[a-zA-Z_][a-zA-Z0-9_]*:[0-9]+\b",
    r"(?:^|[\s(])(?:app|tests|docs|src|config|scripts)[/\\][^\s]+",
)

# 补充少量“工程上下文”词，避免只能靠路径命中。
_ENGINEERING_MARKERS = (
    "函数",
    "模块",
    "配置",
    "脚本",
    "测试",
    "接口",
    "仓库",
    "项目",
    "会话",
    "记忆",
    "检索",
    "向量",
    "权限",
    "工具",
    "注释",
    "prompt",
    "session",
    "memory",
    "embedding",
    "qdrant",
    "config",
    "tool",
    "test",
)


@dataclass(slots=True)
class ReflectionAdmissionDecision:
    """
    自动长期记忆准入决策结果。

    字段说明：
    - `should_reflect`: 这一轮是否允许继续进入自动 reflection
    - `reason`: 判定原因，便于日志追踪和后续调参
    - `project_files_touched`: 过滤后的真实项目文件触点
    - `admission_source`: 本次准入来源
      - `project_file_evidence`: 有明确项目文件证据
      - `execution_signal`: 没有文件触点，但有较强任务执行经验信号
      - `blocked`: 不允许自动 reflection
    """

    should_reflect: bool
    reason: str
    project_files_touched: list[str] = field(default_factory=list)
    admission_source: str = "blocked"


def should_reflect_long_term_memory(step: AgentStep) -> bool:
    """
    判断这次最终回复是否满足“进入自动长期记忆反思”的时机。

    这一层只判断回复阶段是否稳定：
    1. 必须是 assistant 最终回复
    2. 不能是异常兜底文案
    3. 回复里要出现“完成 / 修复 / 结论”这类稳定收尾信号
    """
    if step.type != "assistant":
        return False

    text = step.content.strip()
    if not text:
        return False

    blocked_prefixes = (
        "模型调用失败:",
        "已达到最大循环步数",
        "未识别的模型返回类型",
        "模型返回了空的工具调用",
    )
    if text.startswith(blocked_prefixes):
        return False

    stable_markers = (
        "已完成",
        "已经完成",
        "完成了",
        "阶段完成",
        "已实现",
        "实现了",
        "已修复",
        "修复了",
        "最终",
        "结论",
        "可以确定",
        "建议采用",
        "改为",
        "保留",
        "done",
        "completed",
        "implemented",
        "fixed",
        "conclusion",
        "final",
    )
    lowered = text.lower()
    return any(marker in text or marker in lowered for marker in stable_markers)


def normalize_reflection_path(path: str) -> str:
    """把反思阶段采集到的路径标准化，便于后续做目录规则判断。"""
    return path.strip().replace("/", "\\").lower()


def filter_project_reflection_files(files_touched: list[str]) -> list[str]:
    """
    只保留真正代表“主仓库执行证据”的文件触点。

    自动 project memory 仍然优先依赖真实文件证据，
    但不再把它当成唯一准入条件。
    """
    result: list[str] = []
    seen: set[str] = set()

    for raw_path in files_touched:
        normalized = normalize_reflection_path(raw_path)
        if not normalized:
            continue
        if any(marker in normalized for marker in _NON_PROJECT_PATH_MARKERS):
            continue
        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(raw_path.strip())

    return result


def decide_project_reflection(
    *,
    task_description: str,
    final_response: str,
    key_decisions: list[str],
    failures: list[str],
    files_touched: list[str],
) -> ReflectionAdmissionDecision:
    """
    决定这一轮是否值得进入自动 project memory reflection。

    当前策略刻意做成“更接近 minicode，但比它更保守”：
    1. 有真实项目文件证据时，直接允许进入 reflection
    2. 没有文件证据时，不再一刀切拒绝
       只要存在较强的任务执行经验信号，也允许先产出候选记忆
    3. 最终是否落盘，仍然要交给 guard / verifier 再做二次过滤
    """
    project_files_touched = filter_project_reflection_files(files_touched)
    if project_files_touched:
        return ReflectionAdmissionDecision(
            should_reflect=True,
            reason="检测到真实项目文件触点，允许自动反思",
            project_files_touched=project_files_touched,
            admission_source="project_file_evidence",
        )

    if _has_task_execution_signal(
        task_description=task_description,
        final_response=final_response,
        key_decisions=key_decisions,
        failures=failures,
    ):
        return ReflectionAdmissionDecision(
            should_reflect=True,
            reason="未命中文件触点，但检测到稳定任务执行经验信号，允许先产出候选记忆",
            project_files_touched=[],
            admission_source="execution_signal",
        )

    return ReflectionAdmissionDecision(
        should_reflect=False,
        reason="缺少真实项目证据，也没有足够稳定的任务执行经验信号，跳过自动长期记忆反思",
        project_files_touched=[],
        admission_source="blocked",
    )


def _has_task_execution_signal(
    *,
    task_description: str,
    final_response: str,
    key_decisions: list[str],
    failures: list[str],
) -> bool:
    """
    判断当前任务是否像“值得沉淀的项目执行经验”。

    这里不再硬编码某类算法题名称，而是看结构化信号：
    - 是否有工程/仓库痕迹
    - 是否出现稳定约定、修复结论、失败规避这类语言
    - 是否至少有关键决策或失败信息可供复用
    """
    normalized_decisions = [item.strip() for item in key_decisions if item.strip()]
    normalized_failures = [item.strip() for item in failures if item.strip()]
    if not normalized_decisions and not normalized_failures:
        return False

    signal_text = "\n".join(
        [
            task_description.strip(),
            final_response.strip(),
            *normalized_decisions,
            *normalized_failures,
        ]
    ).strip()
    if not signal_text:
        return False

    if not _contains_repo_or_engineering_context(signal_text):
        return False

    if normalized_failures:
        return True

    if _contains_persistent_project_signal(signal_text):
        return True

    # 没有失败、没有强稳定信号时，至少要有两条以上决策，
    # 说明这更像一次可复用的实现收敛，而不是一句随口回答。
    return len(normalized_decisions) >= 2


def _contains_repo_or_engineering_context(text: str) -> bool:
    """判断文本里是否带有仓库执行或工程实现上下文。"""
    lowered = text.lower()
    if any(marker in lowered for marker in _ENGINEERING_MARKERS):
        return True
    return any(re.search(pattern, text) for pattern in _CODE_OR_REPO_PATTERNS)


def _contains_persistent_project_signal(text: str) -> bool:
    """判断文本里是否出现了稳定约定、收敛结论或风险规避信号。"""
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in _PERSISTENT_PROJECT_MARKERS)
