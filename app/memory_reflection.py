from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from app.types import AgentStep, ChatMessage


# 自动 reflection 当前只允许产出这四类项目级长期记忆。
ALLOWED_MEMORY_CATEGORIES = {
    "preference",  # 当前项目协作中长期稳定有效的偏好
    "convention",  # 项目约定、实现约束、工作方式
    "conclusion",  # 已验证的重要结论、方案、架构判断
    "failure",  # 可复用的失败经验、踩坑结论、风险警告
}

# 明显属于过程性、礼貌性、临时性的表达，不应该进入长期记忆。
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

    task_description: str  # 当前任务描述，通常来自本轮用户输入
    final_step: AgentStep  # 本轮最终 assistant 输出对应的 step
    turn_messages: list[ChatMessage]  # 本轮完整消息链，用来提取执行轨迹
    key_decisions: list[str] = field(default_factory=list)  # 本轮关键决策列表
    failures: list[str] = field(default_factory=list)  # 本轮失败、报错、阻断、风险列表
    files_touched: list[str] = field(default_factory=list)  # 本轮涉及的重要文件路径


@dataclass(slots=True)
class ReflectionMemoryCandidate:
    """
    反思模型返回的一条候选长期记忆。

    这里仍然只是“候选”，
    后续还要经过本地过滤和 guard 才会真正落盘。
    """

    content: str
    category: str
    tags: list[str]
    confidence: float
    domains: list[str] = field(default_factory=list)


class TaskMemoryReflectionEngine:
    """
    基于任务的长期记忆反思引擎。

    职责：
    1. 把当前任务执行整理成结构化上下文
    2. 让模型提炼“未来仍值得复用”的项目级记忆
    3. 对模型输出做本地清洗，减少乱合并、乱拆分和过程性污染
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        *,
        max_context_chars: int = 7000,
        max_candidates: int = 4,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.max_candidates = max_candidates

    def reflect(self, reflection_input: TaskReflectionInput) -> list[ReflectionMemoryCandidate]:
        """
        对当前任务做一次结构化反思。

        流程：
        1. 构造反思上下文
        2. 调模型生成候选记忆
        3. 对候选结果做一层本地清洗
        """
        context_text = self._build_reflection_context(reflection_input)
        if not context_text:
            return []

        raw_candidates = self._call_reflection_model(context_text)
        return self._post_filter_candidates(raw_candidates)

    def _build_reflection_context(self, reflection_input: TaskReflectionInput) -> str:
        """
        构造发给 reflection 模型的结构化上下文。

        模型需要看到的是：
        - 当前任务是什么
        - 最终结果是什么
        - 中间有哪些关键决策、失败、文件触点
        - 本轮执行轨迹的大致摘要
        """
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
        """
        从当前轮消息中提炼轻量 execution trace。

        这里不直接把所有原始消息整段塞给模型，
        只保留对长期记忆提炼更有用的轨迹信息。
        """
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
        """
        调模型做 task reflection。

        当前阶段重点有两类约束：
        1. 只留下稳定、可复用、项目级的记忆
        2. 尽量避免把多个独立约定乱合并，或把同一个约定拆成碎片
        """
        system_prompt = """
你是一个代码 Agent 的长期记忆反思器。

你的任务不是总结聊天内容，而是从一次任务执行中提炼“未来仍值得复用”的项目级长期记忆。

只允许输出以下几类记忆：
1. preference: 在当前项目协作中长期稳定有效的偏好
2. convention: 项目约定、实现约束、工作方式
3. conclusion: 已验证的重要结论、方案、架构判断
4. failure: 可复用的失败经验、踩坑结论、风险警告

你输出的是“候选长期记忆”，不是执行总结，也不是礼貌回复改写。

严格抽取原则：
- 只保留稳定、可复用、跨轮次仍有价值的信息
- 只保留未来再次协作时值得提醒模型的内容
- 默认这些记忆都会写入 project scope，所以不要输出只适合瞬时局部任务的内容
- 如果内容只是“这轮做了什么”，而不是“未来应记住什么”，不要输出

下面这些内容不能高分，通常应该直接不输出：
- 本轮过程描述
- 一次性临时操作
- 没有长期复用价值的解释
- 礼貌性回复、确认性回复、寒暄
- 还未验证的猜测
- 只适合当前瞬时上下文的细节

下面这些内容才可以高分：
- 项目长期约定
- 已验证的重要结论
- 可复用的失败经验
- 当前项目语境下长期稳定的协作偏好

关于“合并 / 拆分”，必须遵守下面规则：
- 如果两条信息属于同一个稳定约定的两个部分，可以合并成一条
- 如果两条信息分别对应两个独立约定，必须拆成两条，不要硬合并
- 不要把“主约定 + 附带说明 + 礼貌结尾”拼成一条
- 一条 memory 最好只表达一个核心规则或一个核心结论
- 如果一条 memory 中出现“；”、“以及”、“并且”连接了两个不同规则，优先拆开
- 但如果两部分本质上是在共同描述同一个约定，也可以保留为一条

confidence 打分规则必须严格使用以下口径：
- 0.90-1.00: 高度稳定、已验证、未来多次复用都成立的项目级记忆
- 0.75-0.89: 较稳定且有复用价值，但验证强度略弱于最高档
- 0.50-0.74: 有一定价值，但稳定性不足、范围偏窄、或仍带过程性痕迹
- 0.00-0.49: 不应写入长期记忆的内容；这类内容尽量不要输出

额外约束：
- 若拿不准，宁可少写，不要多写
- 单次最多输出 4 条
- content 必须写成一句清晰、可复用、去上下文依赖的规则或结论
- 不要在 content 里提“本轮”“刚刚”“这次先”“稍后再看”这类临时表达

只返回 JSON，格式如下：
{
  "memories": [
    {
      "content": "......",
      "category": "preference|convention|conclusion|failure",
      "tags": ["tag1", "tag2"],
      "confidence": 0.0,
      "domains": ["memory", "session"]
    }
  ]
}
""".strip()

        user_prompt = f"""
请基于下面这次任务执行信息，提炼值得长期保留的项目级记忆候选。

注意：
- 不要复述执行过程
- 不要产出临时说明
- 不要产出礼貌性语句
- 只有在“未来仍值得保留”时才输出
- 如果两条候选分别代表两个独立约定，请拆成两条
- 如果只是同一个约定的两种表述，请只保留一条更清晰的版本

{context_text}
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            return []

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
        对模型输出做本地清洗。

        这里不让 confidence 单独决定一切。
        除了 confidence 之外，还会同时看：
        - category 是否在白名单
        - 内容是否明显属于过程性、礼貌性、临时性
        - 内容长度和信息密度是否达标
        - 同一轮候选内是否重复
        - 是否出现“一个 content 混了多个独立约定”的迹象
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
        """解析模型返回的 JSON，兼容 ```json 代码块。"""
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
        """
        判断候选记忆是否过于临时。

        第一阶段先用启发式规则挡掉明显过程性内容，
        避免把“刚刚做了什么”直接写成长期记忆。
        """
        lowered = content.lower()
        return any(marker in lowered for marker in TEMPORARY_CONTENT_MARKERS)

    def _looks_like_low_value_response(self, content: str) -> bool:
        """
        过滤明显低价值的应答式内容。

        这类内容常见于礼貌回复、简单确认、执行状态播报，
        即便模型误给了较高 confidence，也不应该进入长期记忆。
        """
        lowered = content.lower()
        if any(phrase in lowered for phrase in LOW_VALUE_PHRASES):
            return True

        if len(content) <= 24 and ("可以" in content or "好的" in content or "收到" in content):
            return True

        return False

    def _confidence_is_too_low_for_category(self, item: ReflectionMemoryCandidate) -> bool:
        """
        根据类别设置更保守的最低 confidence。

        这样可以避免模型把过程性内容打到 0.5-0.7 之间时仍然被放过。
        """
        min_confidence_by_category = {
            "preference": 0.82,
            "convention": 0.80,
            "conclusion": 0.78,
            "failure": 0.78,
        }
        threshold = min_confidence_by_category.get(item.category, 0.80)
        return item.confidence < threshold

    def _looks_over_merged(self, item: ReflectionMemoryCandidate) -> bool:
        """
        识别“多个独立约定被硬合并到一条 memory”。

        这里只做保守拦截：
        - 只对 convention / conclusion 做这层检查
        - 如果 content 过长，同时出现多个强连接词，并且 tags 数量也偏多，
          通常说明模型把多个独立规则揉到了一条里
        """
        if item.category not in {"convention", "conclusion"}:
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
