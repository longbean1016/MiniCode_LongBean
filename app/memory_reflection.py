from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from app.circuit_breaker import CircuitBreaker
from app.logger import log_event
from app.retry import RetryPolicy, run_with_retry, should_retry_model_error
from app.types import AgentStep, ChatMessage


# 自动 reflection 当前只保留三类“项目执行经验”。
ALLOWED_MEMORY_CATEGORIES = {
    "convention",
    "constraint",
    "failure",
}

# 明显属于过程性、临时性、礼貌性的表达，不应该进入长期记忆。
TEMPORARY_CONTENT_MARKERS = {
    "本轮",
    "这一轮",
    "刚刚",
    "临时",
    "暂时",
    "稍后",
    "马上",
    "待会",
    "这次先",
    "先这样",
    "this turn",
    "just now",
    "temporarily",
    "for now",
    "later",
}

LOW_VALUE_PHRASES = {
    "好的",
    "收到",
    "明白",
    "没问题",
    "已处理",
    "我来帮你",
    "我可以继续",
    "thanks",
    "thank you",
    "got it",
    "sounds good",
}


@dataclass(slots=True)
class TaskReflectionInput:
    """
    一次任务反思所需的结构化输入。

    不是拿整段聊天直接抽记忆，
    而是把当前任务整理成 task description + execution trace 风格的输入。
    """

    task_description: str
    final_step: AgentStep
    turn_messages: list[ChatMessage]
    key_decisions: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReflectionMemoryCandidate:
    """
    反思模型返回的一条候选长期记忆。

    这里只是候选结果，
    后面还要经过 guard 和 verifier 才会真正写入。
    """

    content: str
    category: str
    tags: list[str]
    confidence: float
    domains: list[str] = field(default_factory=list)


class TaskMemoryReflectionEngine:
    """
    基于任务执行经验的长期记忆反思引擎。

    对齐当前的 project-memory 设计后，
    它只负责从一次真实仓库任务执行中提炼：
    - `convention`: 项目约定、实现约束、工作方式
    - `constraint`: 更硬性的项目边界、禁止项、结构约束
    - `failure`: 可复用的失败经验、风险警告
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        *,
        max_context_chars: int = 7000,
        max_candidates: int = 4,
        retry_max_attempts: int = 3,
        retry_base_delay_seconds: float = 0.8,
        retry_backoff_multiplier: float = 2.0,
        retry_max_delay_seconds: float = 4.0,
        circuit_failure_threshold: int = 3,
        circuit_recovery_timeout_seconds: float = 45.0,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.max_candidates = max_candidates
        self.retry_policy = RetryPolicy(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay_seconds,
            backoff_multiplier=retry_backoff_multiplier,
            max_delay_seconds=retry_max_delay_seconds,
        )
        self.circuit_breaker = CircuitBreaker(
            name="memory_reflection",
            failure_threshold=circuit_failure_threshold,
            recovery_timeout_seconds=circuit_recovery_timeout_seconds,
        )

    def reflect(self, reflection_input: TaskReflectionInput) -> list[ReflectionMemoryCandidate]:
        """
        对当前任务做一次结构化反思。

        流程：
        1. 构造反思上下文
        2. 调模型生成候选记忆
        3. 对候选结果做本地过滤
        """
        context_text = self._build_reflection_context(reflection_input)
        if not context_text:
            return []

        raw_candidates = self._call_reflection_model(context_text)
        return self._post_filter_candidates(raw_candidates)

    def _build_reflection_context(self, reflection_input: TaskReflectionInput) -> str:
        """构造发给 reflection 模型的结构化上下文。"""
        parts: list[str] = []

        task_description = reflection_input.task_description.strip()
        if task_description:
            parts.append("## 当前任务")
            parts.append(task_description)

        final_text = reflection_input.final_step.content.strip()
        if final_text:
            parts.append("## 本轮最终结果")
            parts.append(self._shorten(final_text, 1200))

        if reflection_input.key_decisions:
            parts.append("## 关键决策")
            for item in reflection_input.key_decisions[:6]:
                cleaned = item.strip()
                if cleaned:
                    parts.append(f"- {self._shorten(cleaned, 220)}")

        if reflection_input.failures:
            parts.append("## 失败与风险")
            for item in reflection_input.failures[:6]:
                cleaned = item.strip()
                if cleaned:
                    parts.append(f"- {self._shorten(cleaned, 220)}")

        if reflection_input.files_touched:
            parts.append("## 涉及文件")
            for path in reflection_input.files_touched[:10]:
                cleaned = path.strip()
                if cleaned:
                    parts.append(f"- {cleaned}")

        trace_lines = self._collect_turn_trace(reflection_input.turn_messages)
        if trace_lines:
            parts.append("## 本轮执行轨迹")
            parts.extend(trace_lines[:12])

        combined = "\n".join(parts).strip()
        return self._shorten(combined, self.max_context_chars)

    def _collect_turn_trace(self, turn_messages: list[ChatMessage]) -> list[str]:
        """从当前轮消息中提取轻量 execution trace。"""
        trace_lines: list[str] = []

        for message in turn_messages:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()

            if role == "assistant_tool_call":
                tool_name = str(message.get("tool_name", "")).strip()
                if tool_name:
                    trace_lines.append(f"[tool_call] {tool_name}")
                continue

            if role == "tool_result" and content:
                tool_name = str(message.get("tool_name", "")).strip()
                is_error = bool(message.get("is_error", False))
                prefix = "[tool_error]" if is_error else "[tool_result]"
                trace_lines.append(f"{prefix} {tool_name}: {self._shorten(content, 220)}")
                continue

            if role == "assistant" and content:
                trace_lines.append(f"[assistant] {self._shorten(content, 220)}")

        return trace_lines

    def _call_reflection_model(self, context_text: str) -> list[ReflectionMemoryCandidate]:
        """调用模型做任务执行经验反思。"""
        system_prompt = """
你是一个代码 Agent 的任务执行经验反思器。
你的任务不是总结聊天内容，也不是提炼通用知识，而是从一次任务执行中提炼“未来仍值得复用”的项目执行经验。

只允许输出以下三类记忆：
1. convention: 与当前仓库直接相关的项目约定、工作方式、推荐做法
2. constraint: 与当前仓库直接相关的硬性实现约束、禁止项、结构边界
3. failure: 与当前仓库执行过程直接相关、未来可复用的失败经验或风险警告

请严格遵守下面的准则：
- 候选记忆默认会写入 project scope，所以内容必须对当前仓库的后续协作有长期价值
- 如果内容回答的是“这个问题怎么解”，而不是“这个仓库以后该怎么做”，不要输出
- 如果内容只是一次性实验、临时脚本、普通问答、题解模板、通用算法知识，不要输出
- 如果内容只是“这轮做了什么”，而不是“以后应记住什么”，不要输出
- 如果拿不准，宁可少写，不要多写

下面这些内容不能高分，通常应直接不输出：
- 本轮过程描述
- 一次性临时操作
- 礼貌性回复、确认性回复、寒暄
- 还未验证的猜测
- 只适合当前瞬时上下文的细节

下面这些内容可以高分：
- 项目长期约定
- 当前仓库中长期稳定的实现约束
- 当前仓库中明确的禁止项、边界条件、结构限制
- 当前仓库执行中暴露出的稳定风险与规避方式
- 可复用的失败经验

关于拆分规则：
- 如果是两个独立约定，必须拆成两条
- 如果只是同一条约定的不同表达，只保留更清晰的一条
- 一条 memory 最好只表达一个核心规则或一个核心失败经验

confidence 评分规则：
- 0.90-1.00: 高度稳定、已验证、未来多次复用都成立的项目执行经验
- 0.75-0.89: 较稳定且有复用价值，但验证强度略低于最高档
- 0.50-0.74: 有一定价值，但稳定性不足、范围偏窄、或仍带过程痕迹
- 0.00-0.49: 不应写入长期记忆的内容，这类内容尽量不要输出

只返回 JSON，格式如下：
{
  "memories": [
    {
      "content": "......",
      "category": "convention|constraint|failure",
      "tags": ["tag1", "tag2"],
      "confidence": 0.0,
      "domains": ["memory", "session"]
    }
  ]
}
""".strip()

        user_prompt = f"""
请基于下面这次任务执行信息，提炼值得长期保留的项目执行经验候选。

注意：
- 不要复述执行过程
- 不要产出临时说明
- 不要产出礼貌性语句
- 只有当内容与当前仓库实现或执行方式直接相关时才输出
- 如果这更像普通知识答案、算法题回答、tmp 实验结果，请直接输出空数组

{context_text}
""".strip()

        if not self.circuit_breaker.allow_request():
            return []

        def _request_model() -> Any:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )

        try:
            response = run_with_retry(
                _request_model,
                policy=self.retry_policy,
                should_retry=should_retry_model_error,
                on_retry=lambda attempt, error, delay: log_event(
                    (
                        f"长期记忆反思模型调用失败，准备第 {attempt + 1} 次尝试："
                        f"{type(error).__name__}: {error}，等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.circuit_breaker.record_failure(error)
            return []

        self.circuit_breaker.record_success()

        raw_content = response.choices[0].message.content or ""
        payload = self._parse_json_payload(raw_content)
        if not isinstance(payload, dict):
            return []

        raw_memories = payload.get("memories", [])
        if not isinstance(raw_memories, list):
            return []

        result: list[ReflectionMemoryCandidate] = []
        for item in raw_memories[: self.max_candidates]:
            if not isinstance(item, dict):
                continue

            content = " ".join(str(item.get("content", "")).strip().split())
            category = str(item.get("category", "")).strip().lower()

            raw_tags = item.get("tags", [])
            tags = [
                " ".join(str(tag).strip().lower().split())
                for tag in raw_tags
                if str(tag).strip()
            ] if isinstance(raw_tags, list) else []

            raw_domains = item.get("domains", [])
            domains = [
                " ".join(str(domain).strip().lower().split())
                for domain in raw_domains
                if str(domain).strip()
            ] if isinstance(raw_domains, list) else []

            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

            result.append(
                ReflectionMemoryCandidate(
                    content=content,
                    category=category,
                    tags=tags,
                    confidence=max(0.0, min(1.0, confidence)),
                    domains=domains,
                )
            )

        return result

    def _post_filter_candidates(
        self,
        candidates: list[ReflectionMemoryCandidate],
    ) -> list[ReflectionMemoryCandidate]:
        """
        对模型候选做本地清洗。

        这里刻意不再依赖“算法题 / LeetCode / 某道题名”这类主题关键词，
        而是只保留与长期 project memory 结构相关的兜底规则。
        """
        result: list[ReflectionMemoryCandidate] = []
        seen_keys: set[str] = set()

        for item in candidates:
            if not item.content:
                continue
            if item.category not in ALLOWED_MEMORY_CATEGORIES:
                continue
            if len(item.content) < 12:
                continue
            if self._looks_too_temporary(item.content):
                continue
            if self._looks_like_low_value_response(item.content):
                continue
            if self._confidence_is_too_low_for_category(item):
                continue
            if self._looks_over_merged(item):
                continue

            dedupe_key = f"{item.category}::{item.content.lower()}"
            if dedupe_key in seen_keys:
                continue

            seen_keys.add(dedupe_key)
            result.append(item)

        return result

    def _parse_json_payload(self, text: str) -> Any:
        """解析模型返回的 JSON，并兼容 ```json 代码块。"""
        raw = text.strip()

        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _looks_too_temporary(self, content: str) -> bool:
        """过滤明显带有临时过程痕迹的表述。"""
        lowered = content.lower()
        return any(marker in lowered for marker in TEMPORARY_CONTENT_MARKERS)

    def _looks_like_low_value_response(self, content: str) -> bool:
        """过滤礼貌回复、执行播报等低价值表述。"""
        lowered = content.lower()
        if any(phrase.lower() in lowered for phrase in LOW_VALUE_PHRASES):
            return True

        if len(content) <= 24 and ("可以" in content or "好的" in content or "收到" in content):
            return True

        return False

    def _confidence_is_too_low_for_category(self, item: ReflectionMemoryCandidate) -> bool:
        """按类别设置更保守的最低 confidence。"""
        min_confidence_by_category = {
            "convention": 0.80,
            "constraint": 0.82,
            "failure": 0.78,
        }
        threshold = min_confidence_by_category.get(item.category, 0.80)
        return item.confidence < threshold

    def _looks_over_merged(self, item: ReflectionMemoryCandidate) -> bool:
        """识别单条候选里是否混入了多个独立约定。"""
        if item.category != "convention":
            return False

        content = item.content
        split_markers = ("；", ";", "以及", "并且", "同时", "另外")
        marker_count = sum(content.count(marker) for marker in split_markers)

        if len(content) >= 60 and marker_count >= 2 and len(item.tags) >= 4:
            return True

        return False

    def _shorten(self, text: str, max_chars: int) -> str:
        """裁剪长文本，避免 reflection prompt 过重。"""
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."
