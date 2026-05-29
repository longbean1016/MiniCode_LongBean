from __future__ import annotations

from dataclasses import dataclass, field

from app.types import AgentStep


# 这些目录更像运行产物、临时实验或会话数据，
# 不应该单独作为自动 project memory 的写入依据。
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


@dataclass(slots=True)
class ReflectionAdmissionDecision:
    """
    自动长期记忆准入决策结果。

    字段说明：
    - `should_reflect`: 这一轮是否允许继续走自动 reflection
    - `reason`: 判定原因，便于日志追踪和后续调参
    - `project_files_touched`: 过滤后的真实项目文件触点
    """

    should_reflect: bool
    reason: str
    project_files_touched: list[str] = field(default_factory=list)


def should_reflect_long_term_memory(step: AgentStep) -> bool:
    """
    判断这次最终回复是否满足“进入自动长期记忆反思”的时机。

    这一层只判断回复阶段是否稳定：
    - 必须是 assistant 最终回复
    - 必须不是模型异常兜底文案
    - 回复语气里要出现“完成 / 修复 / 结论”这类稳定结束信号
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

    设计意图：
    - 自动 project memory 更像“仓库执行经验记录”
    - 不是普通问答、算法题、tmp 实验脚本的内容池
    - 所以必须先把临时目录、会话目录、运行产物目录剔除掉
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
    files_touched: list[str],
    failures: list[str],
) -> ReflectionAdmissionDecision:
    """
    决定这一轮是否值得进入自动 project memory reflection。

    这里按更接近 minicode 的思路做限制：
    - 自动 reflection 的核心是“任务执行经验”
    - 是否写入 project memory，先看有没有真实仓库执行证据
    - 没有项目文件触点时，默认视为普通问答、临时实验或泛化知识，不自动写入

    注意：
    - `failures` 这里只作为日志上下文保留，不单独放宽准入条件
    - 这样可以避免“只是问了一个算法题，但中间报过一次错”也进入 project memory
    """
    project_files_touched = filter_project_reflection_files(files_touched)
    if project_files_touched:
        return ReflectionAdmissionDecision(
            should_reflect=True,
            reason="检测到真实项目文件触点，允许自动反思",
            project_files_touched=project_files_touched,
        )

    if failures:
        return ReflectionAdmissionDecision(
            should_reflect=False,
            reason="存在失败信息，但没有真实项目文件触点，拒绝自动写入 project memory",
            project_files_touched=[],
        )

    return ReflectionAdmissionDecision(
        should_reflect=False,
        reason="没有真实项目执行证据，跳过自动长期记忆反思",
        project_files_touched=[],
    )
