


# 自动 reflection 允许写入的长期记忆类别。
# 自动写入时，scope固定位project作用域，user、local暂不设置为自动处理

from dataclasses import dataclass, field
import json
from typing import Any

from openai import OpenAI

from app.types import AgentStep, ChatMessage


ALLOWED_MEMORY_CATEGORIES={
    "preference",  # 当前项目协作中长期稳定有效的偏好
    "convention",  # 项目约定、实现约束、工作方式
    "conclusion",  # 已验证的重要结论、方案、架构判断
    "failure",     # 可复用的失败经验、踩坑结论、风险警告
}


@dataclass(slots=True)
class TaskReflectionInput:
    """
    一次任务反思所需的结构化输入。

    不是拿一整段聊天直接抽记忆，
    而是把当前任务整理成 task description + execution trace 风格的输入。
    """

    task_description: str  # 当前任务描述，一般就是本轮用户输入或任务目标摘要
    final_step: AgentStep  # 本轮最终 assistant 输出对应的 step，用来看最终产出
    turn_messages: list[ChatMessage]  # 本轮完整消息链，用来提取 execution trace
    key_decisions: list[str] = field(default_factory=list)  # 本轮关键决策列表
    failures: list[str] = field(default_factory=list)  # 本轮失败、报错、阻断、风险列表
    files_touched: list[str] = field(default_factory=list)  # 本轮涉及的重要文件路径列表



@dataclass(slots=True)
class ReflectionMemoryCandidate:
    """
    反思模型返回的一条候选长期记忆。

    字段说明：
    - content: 记忆正文
    - category: 记忆类别
    - tags: 辅助检索标签
    - confidence: 模型对这条记忆“值得长期保存”的置信度
    - domains: 可选领域标签，例如 memory / session / permissions
    """
    content: str
    category: str
    tags: list[str]
    confidence: float
    domains: list[str] = field(default_factory=list)

class TaskMemoryReflectionEngine:
    """
    task-based reflection 引擎。

    它的职责：
    1. 把当前一轮任务执行整理成结构化上下文
    2. 调模型提炼“值得长期保留”的项目级记忆
    3. 返回候选项给上层做 confidence / dedupe / conflict gate

    注意：
    - 这一层不负责真正写入 memory store
    - 这一层也不负责决定 user/local/project
    - 自动链路默认只服务 project 级长期记忆
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
        # # OpenAI 兼容客户端。
        # 这里沿用你项目现有的模型接入方式。
        self.client=OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        # 反思型模型名
        self.model_name=model_name

        # 给反思模型的最大上下文字符数。
        # 防止本轮的trace太长，把prompt撑爆。
        self.max_context_chars=max_context_chars

        # 单次最多返回多少条候选长期记忆。
        # 这里故意保守，避免“一轮写很多条”。
        self.max_candidates=max_candidates

    
    def reflect(self,reflection_input: TaskReflectionInput)-> list[ReflectionMemoryCandidate]:
        """
        对当前任务做一次结构化反思。

        流程：
        1. 先把输入整理成反思上下文
        2. 调模型产出候选记忆
        3. 对模型输出做一层本地清洗
        """
        context_text=self._build_reflection_context(reflection_input)

        # 没有有效上下文时，直接不抽。
        if not context_text:
            return []
        
        raw_candidates=self._call_reflection_model(context_text)

        return self._post_filter_candidates(raw_candidates)
    

    def _build_reflection_context(self,reflection_input: TaskReflectionInput)->str:
        """
        构建反思用上下文。

        目标是让模型看到：
        - 当前任务是什么
        - 最终结果是什么
        - 中间有哪些关键决策 / 失败 / 文件触点
        - 本轮大致做了哪些动作
        """
        parts: list[str]=[]

        # task_description: 用户这轮的任务描述。
        task_description=reflection_input.task_description.strip()
        if task_description:
            parts.append("### 当前任务：")
            parts.append(task_description)

        # final_text: 本轮最终 assistant 输出。
        final_text=reflection_input.final_step.content.strip()
        if final_text:
            parts.append("### 本轮最终结果：")
            parts.append(self._shorten(final_text, 1200))

        # key_decisions: 本轮出现的关键决策。
        if reflection_input.key_decisions:
            parts.append("## 关键决策")
            for item in reflection_input.key_decisions[:6]:
                item = item.strip()
                if item:
                    parts.append(f"- {self._shorten(item, 220)}")
        
        # failures: 本轮出现的失败、阻断、报错、风险信息。
        if reflection_input.failures:
            parts.append("## 失败与风险")
            for item in reflection_input.failures[:6]:
                item = item.strip()
                if item:
                    parts.append(f"- {self._shorten(item, 220)}")

        # files_touched: 本轮触及的重要文件。
        if reflection_input.files_touched:
            parts.append("## 涉及文件")
            for path in reflection_input.files_touched[:10]:
                path = path.strip()
                if path:
                    parts.append(f"- {path}")

        # trace_lines: 从本轮消息里提取出来的轻量执行痕迹。
        trace_lines = self._collect_turn_trace(reflection_input.turn_messages)
        if trace_lines:
            parts.append("## 本轮执行痕迹")
            parts.extend(trace_lines[:12])

        # 把所有段落拼成完整反思上下文。
        combined = "\n".join(parts).strip()

        # 最后统一做一次总长度限制。
        return self._shorten(combined, self.max_context_chars)
    


    def _collect_turn_trace(self, turn_messages: list[ChatMessage])-> list[str]:
        """
        从当前轮次消息中抽出一份轻量 execution trace。

        这里不是把所有消息原样塞给模型，
        而是只挑对长期记忆提炼更有价值的轨迹：
        - tool call
        - tool result
        - assistant 结果
        """
        trace_lines: list[str]=[]
        for message in turn_messages:
            # role: 当前消息角色，可能是 assistant / tool_result / assistant_tool_call 等。
            role = str(message.get("role", "")).strip()
            # content: 当前消息文本正文。
            content = str(message.get("content", "")).strip()
            # assistant_tool_call: 记录模型调用了哪个工具。
            if role == "assistant_tool_call":
                tool_name = str(message.get("tool_name", "")).strip()
                if tool_name:
                    trace_lines.append(f"[tool_call] {tool_name}")
            
            # tool_result: 记录工具执行结果。
            if role == "tool_result" and content:
                tool_name = str(message.get("tool_name", "")).strip()

                # is_error: 这条工具结果是不是错误结果。
                is_error = bool(message.get("is_error", False))

                prefix = "[tool_error]" if is_error else "[tool_result]"
                trace_lines.append(f"{prefix} {tool_name}: {self._shorten(content, 220)}")

            # assistant: 记录本轮 assistant 的关键输出。
            if role == "assistant" and content:
                trace_lines.append(f"[assistant] {self._shorten(content, 220)}")
            
        return trace_lines
                


    def _call_reflection_model(self,context_text: str)->list[ReflectionMemoryCandidate]:
        """
        调模型做 task reflection。

        注意：
        - 这里明确要求模型只产出“项目级长期记忆候选”
        - 不让模型自己决定 scope
        - scope 由上层固定视为 project
        """
        system_prompt = """
你是一个代码 Agent 的长期记忆反思器。

你的任务不是总结聊天内容，而是从一次任务执行中提炼“未来仍值得复用”的项目级长期记忆。

只允许输出以下几类记忆：
1. preference: 在当前项目协作中长期稳定有效的偏好
2. convention: 项目约定、实现约束、工作方式
3. conclusion: 已验证的重要结论、方案、架构判断
4. failure: 可复用的失败经验、踩坑结论、风险警告

抽取原则：
- 只保留稳定、可复用、跨轮次仍有价值的信息
- 不要保留一次性的临时细节
- 不要把普通礼貌回复写成记忆
- 如果只是“这轮做了什么”而不是“未来应记住什么”，不要输出
- 置信度必须保守，没把握就给低分
- 默认这些记忆都会写入 project scope，所以不要输出只适用于瞬时局部任务的内容

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
        
        # user_prompt: 真正送给模型的任务上下文。
        user_prompt = f"""
请基于下面这次任务执行信息，提炼值得长期保留的项目级记忆候选：

{context_text}
""".strip()
        
        try:
            response=self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system","content": system_prompt},
                    {"role": "user","content": user_prompt}
                ],
                # 和你项目其他地方保持一致，关闭 thinking。
                extra_body={"thinking": {"type": "disabled"}},
            )
        except:
            # 反思失败时直接返回空，不影响主流程。
            return []
        
        # raw_content: 模型返回的原始文本。
        raw_content=response.choices[0].message.content or ""

        # payload: 解析后的 JSON 对象。
        payload = self._parse_json_payload(raw_content)
        if not isinstance(payload,dict):
            return []
        
        # raw_memories: 模型返回的 memories 列表。
        raw_memories = payload.get("memories", [])
        if not isinstance(raw_memories, list):
            return []
        
        result: list[ReflectionMemoryCandidate] = []

        for item in raw_memories[: self.max_candidates]:
            if not isinstance(item, dict):
                continue

            # content: 候选记忆正文。
            content = " ".join(str(item.get("content", "")).strip().split())

            # category: 候选记忆类别。
            category = str(item.get("category", "")).strip().lower()

            # raw_tags: 模型给出的原始 tags。
            raw_tags = item.get("tags", [])
            tags = [
                " ".join(str(tag).strip().lower().split())
                for tag in raw_tags
                if str(tag).strip()
            ] if isinstance(raw_tags, list) else []

            # raw_domains: 模型给出的原始领域标签。
            raw_domains = item.get("domains", [])
            domains = [
                " ".join(str(domain).strip().lower().split())
                for domain in raw_domains
                if str(domain).strip()
            ] if isinstance(raw_domains, list) else []

            # confidence: 模型给出的置信度。
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

        这里不负责复杂语义判断，
        主要做：
        - 白名单类别过滤
        - 过短内容过滤
        - 临时性内容过滤
        - 同一轮内去重
        """
        result: list[ReflectionMemoryCandidate] = []
        seen_keys: set[str] = set()

        for item in candidates:
            # content 为空，直接跳过。
            if not item.content:
                continue

            # 只允许白名单类别。
            if item.category not in ALLOWED_MEMORY_CATEGORIES:
                continue

            # 太短通常没有信息量。
            if len(item.content) < 12:
                continue

            # 明显属于瞬时过程的信息，不进入长期记忆。
            if self._looks_too_temporary(item.content):
                continue

            # dedupe_key: 同一轮内部去重键。
            dedupe_key = f"{item.category}::{item.content.lower()}"
            if dedupe_key in seen_keys:
                continue

            seen_keys.add(dedupe_key)
            result.append(item)

        return result

    def _parse_json_payload(self, text: str) -> Any:
        """
        解析模型返回的 JSON。

        有些模型会把 JSON 包在 ```json 代码块里，
        这里顺手兼容一下。
        """
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
        判断一条候选记忆是否太临时。

        第一版先用启发式规则过滤，
        避免把“刚刚执行了什么”这种瞬时过程写进长期记忆。
        """
        lowered = content.lower()
        markers = [
            "本轮",
            "这一次",
            "刚刚",
            "临时",
            "暂时",
            "稍后",
            "马上",
            "this turn",
            "just now",
            "temporarily",
        ]
        return any(marker in lowered for marker in markers)
    
    def _shorten(self, text: str, max_chars: int) -> str:
        """
        裁剪长文本，避免 prompt 过重。

        这里会先压缩空白，再按最大字符数截断。
        """
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."
