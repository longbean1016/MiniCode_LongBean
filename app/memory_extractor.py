from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from app.memory_store import MemoryEntry, create_memory_entry
from app.types import AgentStep, ChatMessage


# 允许写入长期记忆的类别白名单。
# 这样可以避免模型输出项目里根本不认识的 category。
ALLOWED_MEMORY_CATEGORIES = {
    "preference",  # 用户长期偏好
    "convention",  # 项目约定 / 开发约束
    "conclusion",  # 已确认的重要结论 / 方案
    "failure",     # 失败经验 / 避坑点
}


@dataclass(slots=True)
class ExtractedMemoryCandidate:
    """
    表示模型抽取出来的一条候选长期记忆。

    先用这个轻量对象承接模型输出，
    后面再统一转成 MemoryEntry。
    """

    content: str  # 记忆正文
    category: str  # 记忆类别
    tags: list[str]  # 记忆标签
    should_store: bool = True  # 模型是否建议把这条记忆写入长期记忆


class LongTermMemoryExtractor:
    """
    长期记忆抽取器。

    作用：
    1. 从当前这一轮消息中抽取值得跨会话保留的信息
    2. 只允许抽取固定的长期记忆类别
    3. 在写入 memory store 之前做本地清洗、去重和过滤
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_context_chars: int = 6000,
        max_memories_per_turn: int = 4,
    ) -> None:
        # 复用当前项目已经在使用的 OpenAI 兼容客户端。
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # 抽取记忆所用的模型名。
        self.model_name = model_name

        # 发给抽取模型的上下文最大字符数。
        self.max_context_chars = max_context_chars

        # 每轮最多写入多少条长期记忆，避免噪音太多。
        self.max_memories_per_turn = max_memories_per_turn

    def extract_from_turn(
        self,
        *,
        user_input: str,
        final_step: AgentStep,
        turn_messages: list[ChatMessage],
        session_id: str,
    ) -> list[MemoryEntry]:
        """
        从当前一轮完整消息链路里抽取长期记忆。

        参数说明：
        - user_input: 用户本轮原始问题
        - final_step: 本轮最终返回给用户的 step
        - turn_messages: 当前这一轮的完整消息
        - session_id: 当前会话 id
        """
        # 先把这一轮消息整理成一段较短但保留关键事实的上下文。
        turn_context = self._build_turn_context(
            user_input=user_input,
            final_step=final_step,
            turn_messages=turn_messages,
        )

        # 没有可用上下文时直接不抽。
        if not turn_context.strip():
            return []

        # 调模型做结构化抽取。
        raw_candidates = self._call_extraction_model(turn_context)

        # 对模型输出做本地清洗。
        cleaned_candidates = self._post_filter_candidates(raw_candidates)

        # 统一转成项目里的 MemoryEntry。
        result: list[MemoryEntry] = []
        for candidate in cleaned_candidates[: self.max_memories_per_turn]:
            result.append(
                create_memory_entry(
                    content=candidate.content,
                    category=candidate.category,
                    tags=candidate.tags,
                    session_id=session_id,
                    extra={"source": "model_extractor"},
                )
            )

        return result

    def _build_turn_context(
        self,
        *,
        user_input: str,
        final_step: AgentStep,
        turn_messages: list[ChatMessage],
    ) -> str:
        """
        把当前一轮消息压成抽取模型需要的上下文文本。

        这里不会原样塞入整轮所有消息，
        而是优先保留更适合进入长期记忆的内容。
        """
        parts: list[str] = []

        # 用户原始问题通常最重要，单独作为第一段。
        if user_input.strip():
            parts.append("## 用户本轮问题")
            parts.append(user_input.strip())

        # 收集 assistant 文本消息。
        assistant_texts: list[str] = []

        # 收集关键工具结果。
        tool_results: list[str] = []

        for message in turn_messages:
            # role 表示这条消息属于 user / assistant / tool_result 等哪种类型。
            role = message.get("role")

            # content 是这条消息的文本正文。
            content = str(message.get("content", "")).strip()

            # 普通 assistant 文本消息。
            if role == "assistant" and content:
                assistant_texts.append(content)

            # 工具结果消息。
            if role == "tool_result" and content:
                # tool_name 表示是哪一个工具返回了这条结果。
                tool_name = str(message.get("tool_name", "")).strip()

                # is_error 表示是否为错误结果。
                is_error = bool(message.get("is_error", False))

                # preview 是裁短后的工具结果预览。
                preview = self._shorten(content, 600)

                if is_error:
                    tool_results.append(f"[错误结果] {tool_name}: {preview}")
                else:
                    tool_results.append(f"[工具结果] {tool_name}: {preview}")

        # 关键工具结果单独放一段。
        if tool_results:
            parts.append("## 本轮关键工具结果")
            parts.extend(tool_results[:6])

        # 只保留最后几条 assistant 文本，避免重复过多。
        if assistant_texts:
            parts.append("## 本轮 assistant 关键信息")
            for text in assistant_texts[-3:]:
                parts.append(self._shorten(text, 500))

        # final_step 是本轮最后对外暴露的结果。
        if final_step.content.strip():
            parts.append("## 本轮最终结果")
            parts.append(self._shorten(final_step.content, 800))

        # 拼成完整上下文并做总长度限制。
        combined = "\n".join(parts).strip()
        return self._shorten(combined, self.max_context_chars)

    def _call_extraction_model(self, turn_context: str) -> list[ExtractedMemoryCandidate]:
        """
        调模型抽取长期记忆。

        要求模型只返回 JSON。
        如果模型输出不合法，就降级为空列表。
        """
        system_prompt = """
你是一个“代码 Agent 的长期记忆提取器”。

你的任务不是总结整轮对话，而是从本轮上下文里抽取“值得跨会话保留”的长期记忆。

只允许抽取以下 4 类：
1. preference: 用户长期偏好，例如回答语言、代码风格、协作方式
2. convention: 项目约定，例如文件放置规则、架构要求、实现边界
3. conclusion: 已确认的重要结论或方案
4. failure: 有复用价值的失败经验、报错规律、避坑点

抽取原则：
- 只保留未来还可能有用的信息
- 不要保留一次性的临时细节
- 不要把普通客套话写成记忆
- 如果某条信息不值得长期保存，就不要输出
- 内容要简洁、明确、可复用

请只返回 JSON，格式如下：
{
  "memories": [
    {
      "content": "......",
      "category": "preference|convention|conclusion|failure",
      "tags": ["tag1", "tag2"],
      "should_store": true
    }
  ]
}
""".strip()

        user_prompt = f"""
以下是当前这一轮对话的关键上下文，请抽取值得长期保存的记忆。

{turn_context}
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # 与当前项目其他模型调用保持一致，关闭 thinking。
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            return []

        # raw_content 是模型直接返回的文本。
        raw_content = response.choices[0].message.content or ""
        if not raw_content.strip():
            return []

        # 解析模型 JSON。
        parsed = self._parse_json_payload(raw_content)
        if not isinstance(parsed, dict):
            return []

        # memories 应该是列表。
        raw_memories = parsed.get("memories", [])
        if not isinstance(raw_memories, list):
            return []

        result: list[ExtractedMemoryCandidate] = []

        for item in raw_memories:
            if not isinstance(item, dict):
                continue

            # content 是候选记忆正文。
            content = str(item.get("content", "")).strip()

            # category 是候选记忆类别。
            category = str(item.get("category", "")).strip().lower()

            # raw_tags 是模型返回的原始标签值。
            raw_tags = item.get("tags", [])

            # tags 统一清洗成字符串列表。
            if isinstance(raw_tags, list):
                tags = [
                    str(tag).strip()
                    for tag in raw_tags
                    if str(tag).strip()
                ]
            else:
                tags = []

            # should_store 表示模型是否建议保存这条记忆。
            should_store = bool(item.get("should_store", True))

            result.append(
                ExtractedMemoryCandidate(
                    content=content,
                    category=category,
                    tags=tags,
                    should_store=should_store,
                )
            )

        return result

    def _post_filter_candidates(
        self,
        candidates: list[ExtractedMemoryCandidate],
    ) -> list[ExtractedMemoryCandidate]:
        """
        对模型返回的候选记忆做本地过滤。

        这一层不负责重新理解语义，
        主要负责清洗脏数据、低价值数据和重复数据。
        """
        result: list[ExtractedMemoryCandidate] = []

        # seen_keys 用来做本轮内去重。
        seen_keys: set[str] = set()

        for candidate in candidates:
            # 模型明确说不存，就直接跳过。
            if not candidate.should_store:
                continue

            # 统一压缩空白字符，避免换行和多空格影响去重。
            content = " ".join(candidate.content.strip().split())

            # category 统一转小写，方便做白名单判断。
            category = candidate.category.strip().lower()

            # 空内容直接跳过。
            if not content:
                continue

            # 只允许固定的长期记忆类别。
            if category not in ALLOWED_MEMORY_CATEGORIES:
                continue

            # 太短通常没有信息量。
            if len(content) < 8:
                continue

            # 太长通常是在整段复述，不适合直接写入长期记忆。
            if len(content) > 220:
                content = self._shorten(content, 220)

            # 过滤明显的临时性内容。
            if self._looks_too_temporary(content):
                continue

            # 清洗标签。
            cleaned_tags: list[str] = []

            # seen_tags 用来做单条记忆内部标签去重。
            seen_tags: set[str] = set()

            for tag in candidate.tags:
                # normalized_tag 是清洗后的标签文本。
                normalized_tag = " ".join(tag.strip().lower().split())
                if not normalized_tag:
                    continue
                if normalized_tag in seen_tags:
                    continue
                seen_tags.add(normalized_tag)
                cleaned_tags.append(normalized_tag)

            # 第一版去重键使用 category + content。
            dedupe_key = f"{category}::{content.lower()}"
            if dedupe_key in seen_keys:
                continue

            seen_keys.add(dedupe_key)
            result.append(
                ExtractedMemoryCandidate(
                    content=content,
                    category=category,
                    tags=cleaned_tags,
                    should_store=True,
                )
            )

        return result

    def _parse_json_payload(self, text: str) -> Any:
        """
        解析模型返回的 JSON。

        一些兼容模型可能会把 JSON 包在 ```json 代码块里，
        这里顺手做兼容。
        """
        raw = text.strip()

        # 去掉 markdown 代码块包裹。
        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```JSON")
            raw = raw.removeprefix("```")
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
        避免把“刚刚执行了某命令”这种瞬时信息写进长期记忆。
        """
        lowered = content.lower()

        temporary_markers = [
            "刚刚",
            "本轮",
            "这一次",
            "临时",
            "暂时",
            "稍后",
            "马上",
            "刚才",
            "this turn",
            "just now",
            "temporarily",
        ]

        return any(marker in lowered for marker in temporary_markers)

    def _shorten(self, text: str, max_chars: int) -> str:
        """
        把长文本裁短，避免抽取器上下文或记忆内容过长。
        """
        # cleaned 是压缩空白后的文本。
        cleaned = " ".join(text.strip().split())

        if len(cleaned) <= max_chars:
            return cleaned

        return cleaned[:max_chars].rstrip() + "..."
