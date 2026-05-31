from __future__ import annotations

import hashlib

"""历史摘要模块，负责把较早轮次的对话压成可继续使用的摘要。"""

from openai import OpenAI

from app.circuit_breaker import CircuitBreaker
from app.compaction_policy import build_compaction_policy
from app.history_window import build_older_history_summary
from app.logger import log_event
from app.retry import RetryPolicy, run_with_retry, should_retry_model_error
from app.session import SessionData
from app.types import ChatMessage


class OlderHistorySummarizer:
    """
    旧历史摘要器。

    作用：
    1. 只对最近窗口之外的 older_messages 做摘要
    2. 摘要结果写入 SessionData.extra，作为可持久化的 context state
    3. 只有旧历史指纹变化且超过阈值时，才重新调用模型
    4. 模型失败时自动回退到规则摘要
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        *,
        max_context_chars: int = 7000,
        max_summary_chars: int = 600,
        min_message_count: int = 6,
        min_total_chars: int = 500,
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
        self.max_summary_chars = max_summary_chars
        self.min_message_count = min_message_count
        self.min_total_chars = min_total_chars
        # 历史摘要失败时有 fallback summary，所以这里采用保守重试策略即可。
        self.retry_policy = RetryPolicy(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay_seconds,
            backoff_multiplier=retry_backoff_multiplier,
            max_delay_seconds=retry_max_delay_seconds,
        )
        self.circuit_breaker = CircuitBreaker(
            name="history_summarizer",
            failure_threshold=circuit_failure_threshold,
            recovery_timeout_seconds=circuit_recovery_timeout_seconds,
        )

    def summarize(
        self,
        *,
        session: SessionData,
        older_messages: list[ChatMessage],
        older_round_count: int,
    ) -> str:
        """
        为旧历史生成摘要，并把摘要缓存写入 session.extra。

        extra 中维护：
        - older_history_summary
        - older_history_fingerprint
        - last_compacted_round_count
        - compaction_level
        """
        if not older_messages:
            self._clear_session_state(session)
            return ""

        # fallback summary 先行构造。
        # 这样后面无论是模型阈值不满足、命中缓存、还是模型调用失败，都有稳定兜底结果。
        fallback_summary = build_older_history_summary(older_messages)
        fingerprint = self._fingerprint_messages(older_messages)
        policy = build_compaction_policy(session)

        cached_summary = str(session.extra.get("older_history_summary", "")).strip()
        cached_fingerprint = str(session.extra.get("older_history_fingerprint", "")).strip()
        cached_round_count = self._safe_int(
            session.extra.get("last_compacted_round_count", 0),
            0,
        )
        compaction_level = self._safe_int(
            session.extra.get("compaction_level", 0),
            0,
        )

        # 旧历史太短时没必要调模型。
        # 对很小的一段 older_messages，规则摘要往往更稳定，也更省一次模型调用。
        if not self._should_use_model(older_messages):
            self._save_session_state(
                session=session,
                summary=fallback_summary,
                fingerprint=fingerprint,
                older_round_count=older_round_count,
                compaction_level=max(1, compaction_level),
            )
            return fallback_summary

        # 指纹一致说明 older 区域没有实质变化，直接复用缓存摘要即可。
        # 这里不重新总结，是为了避免“同一段旧历史被模型多次改写措辞”，降低漂移。
        if cached_summary and cached_fingerprint == fingerprint:
            self._save_session_state(
                session=session,
                summary=cached_summary,
                fingerprint=fingerprint,
                older_round_count=older_round_count,
                compaction_level=max(1, compaction_level),
            )
            return cached_summary

        # 即使 older 区域发生了变化，也不代表每轮都值得重摘要。
        # round_delta 是一个节流阈值，避免会话很长时摘要器频繁重跑。
        round_delta = max(0, older_round_count - cached_round_count)
        if cached_summary and round_delta < policy.min_round_delta_for_resummarize:
            self._save_session_state(
                session=session,
                summary=cached_summary,
                fingerprint=fingerprint,
                older_round_count=older_round_count,
                compaction_level=max(1, compaction_level),
            )
            return cached_summary

        # 传给摘要模型的输入不是原始消息 JSON，而是按 role 扁平化后的文本。
        # 这样做有两个好处：
        # 1. 更省 token
        # 2. 更明确告诉模型“哪些是 user / assistant / tool result”
        context_text = self._build_context_text(older_messages)
        if not context_text:
            self._save_session_state(
                session=session,
                summary=fallback_summary,
                fingerprint=fingerprint,
                older_round_count=older_round_count,
                compaction_level=max(1, compaction_level + 1),
            )
            return fallback_summary

        try:
            summary = self._call_model(
                context_text=context_text,
                compaction_level=policy.level,
            ).strip()
        except Exception:
            summary = ""

        final_summary = summary or fallback_summary
        final_summary = self._shorten(final_summary, self.max_summary_chars)

        self._save_session_state(
            session=session,
            summary=final_summary,
            fingerprint=fingerprint,
            older_round_count=older_round_count,
            compaction_level=max(1, compaction_level + 1),
        )
        return final_summary

    def _should_use_model(self, older_messages: list[ChatMessage]) -> bool:
        """判断这一批旧历史是否值得调用模型做摘要。"""
        if len(older_messages) < self.min_message_count:
            return False

        total_chars = 0
        for message in older_messages:
            total_chars += len(str(message.get("content", "")))

        return total_chars >= self.min_total_chars

    def _fingerprint_messages(self, older_messages: list[ChatMessage]) -> str:
        """为旧历史生成稳定指纹。"""
        parts: list[str] = []

        for message in older_messages:
            # 指纹故意只纳入会影响语义判断的字段：
            # role、tool_name、是否报错、content。
            # 这样既能捕获真正的事实变化，又不会因为无关元数据变化而频繁失效。
            role = str(message.get("role", "")).strip()
            tool_name = str(message.get("tool_name", "")).strip()
            content = str(message.get("content", "")).strip()
            is_error = "1" if bool(message.get("is_error", False)) else "0"
            parts.append(f"{role}|{tool_name}|{is_error}|{content}")

        raw_text = "\n".join(parts)
        return hashlib.sha1(raw_text.encode("utf-8")).hexdigest()

    def _build_context_text(self, older_messages: list[ChatMessage]) -> str:
        """把旧历史整理成摘要模型需要的输入文本。"""
        lines: list[str] = []

        for message in older_messages:
            role = str(message.get("role", "")).strip()
            content = self._normalize_text(str(message.get("content", "")))
            if not content:
                continue

            # 不同 role 单独打标签，是为了让摘要模型更容易识别：
            # 哪些是用户约束，哪些是 agent 的结论，哪些是工具证据。
            if role == "user":
                lines.append(f"[user] {self._shorten(content, 500)}")
            elif role == "assistant":
                lines.append(f"[assistant] {self._shorten(content, 500)}")
            elif role == "assistant_tool_call":
                tool_name = str(message.get("tool_name", "")).strip() or "unknown_tool"
                lines.append(f"[assistant_tool_call:{tool_name}] {self._shorten(content, 240)}")
            elif role == "tool_result":
                tool_name = str(message.get("tool_name", "")).strip() or "unknown_tool"
                prefix = "error" if bool(message.get("is_error", False)) else "result"
                lines.append(f"[tool_{prefix}:{tool_name}] {self._shorten(content, 300)}")

        return self._shorten("\n".join(lines).strip(), self.max_context_chars)

    def _call_model(self, *, context_text: str, compaction_level: int) -> str:
        """调模型生成旧历史主线摘要。"""
        system_prompt = (
            "你是一个代码 Agent 的旧历史摘要器。\n"
            "你的任务是总结当前最近几轮之前的历史主线，供后续轮次继续参考。\n"
            "请重点保留：长期目标、已确认方案、重要约束、关键报错与处理结果、用户偏好。\n"
            "不要逐条复述消息，不要写成流水账，不要重复最近几轮很可能已经出现的细节。\n"
            f"当前压缩等级: {compaction_level}。等级越高，越要偏向保留结论、约定、失败经验，少保留过程细节。\n"
            "输出使用简洁中文，控制在 4 到 8 行以内。"
        )

        user_prompt = (
            "以下是当前会话较早的历史消息，请总结成旧历史摘要。\n\n"
            f"{context_text}"
        )

        if not self.circuit_breaker.allow_request():
            raise RuntimeError("history_summarizer 熔断中")

        def _request_model() -> object:
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
                        f"旧历史摘要模型调用失败，准备第 {attempt + 1} 次尝试："
                        f"{type(error).__name__}: {error}，等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.circuit_breaker.record_failure(error)
            raise

        self.circuit_breaker.record_success()
        return response.choices[0].message.content or ""

    def _save_session_state(
        self,
        *,
        session: SessionData,
        summary: str,
        fingerprint: str,
        older_round_count: int,
        compaction_level: int,
    ) -> None:
        """把旧历史摘要相关的 context state 写入 session.extra。"""
        # 这些字段本质上就是“older 区域的持久化上下文状态”：
        # 下轮进来后，无需重新扫描完整旧历史，就能判断是否复用摘要、是否需要升级压缩等级。
        session.extra["older_history_summary"] = summary
        session.extra["older_history_fingerprint"] = fingerprint
        session.extra["last_compacted_round_count"] = older_round_count
        session.extra["compaction_level"] = compaction_level

    def _clear_session_state(self, session: SessionData) -> None:
        """没有旧历史时，清理这几个持久化的 context state 字段。"""
        session.extra.pop("older_history_summary", None)
        session.extra.pop("older_history_fingerprint", None)
        session.extra.pop("last_compacted_round_count", None)
        session.extra.pop("compaction_level", None)

    def _safe_int(self, value: object, default: int) -> int:
        """把 extra 里的值转成 int，避免旧数据格式不一致。"""
        try:
            return int(value) # type: ignore
        except (TypeError, ValueError):
            return default

    def _normalize_text(self, text: str) -> str:
        """清洗文本里的多余空白。"""
        return " ".join(text.strip().split())

    def _shorten(self, text: str, max_chars: int) -> str:
        """把长文本裁到固定长度。"""
        cleaned = self._normalize_text(text)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."
