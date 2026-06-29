from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

"""后台记忆反思模块。

参照 Hermes agent/background_review.py 的 _MEMORY_REVIEW_PROMPT 设计，
每 N 次工具调用后启动 daemon 线程，将对话历史发给 aux model 审阅，
通过 tool calling 调用 memory 工具写入持久记忆。
"""

from openai import OpenAI

from app.agent.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.agent.retry import RetryPolicy, run_with_retry, should_retry_model_error
from app.logger import log_event
from app.memory.memory_store import MemoryStore
from app.memory.memory_tool import get_memory_store
from app.types import ChatMessage

# ---------------------------------------------------------------------------
# 中文版后台反思提示词（参照 Hermes _MEMORY_REVIEW_PROMPT）
# ---------------------------------------------------------------------------

REVIEW_SYSTEM_PROMPT = """你是代码 Agent 的后台记忆管理员。请审阅对话历史，判断是否有值得跨会话保留的信息。

**什么时候该记（满足其一就应该写入）：**

1. 用户透露了身份、角色、技术水平、工作领域等个人信息 → 写入 user
2. 用户表达了关于回答风格的偏好或纠正：如"太啰嗦了"、"不要解释"、"直接给答案"、"用中文注释"、"结论优先" → 写入 user
3. 用户对工作方式提了要求：如"先读文件再改"、"不要重复读同一个文件"、"每次提交前先检查" → 写入 memory
4. 出现了项目特有的规范或约定：如"这个项目用 pytest"、"数据库连接池默认 20"、"部署要 Python 3.11+" → 写入 memory
5. 踩了坑并找到了解决方案：如"装了 X 包才能跑"、"配置了 Y 环境变量才正常" → 写入 memory
6. 用户明确说"记住…"、"下次…"、"以后都…" → 仔细判断后写入对应 target

**什么不该记（以下内容必须跳过）：**

1. 环境相关的临时错误：缺某个包、命令找不到、权限不够 → 用户修了下次不一定有
2. 一次性任务的描述：如"帮我分析下这个 PR"、"总结下今天的改动"
3. 工具的一次性失败：读文件被截断、grep 没找到结果再试一次就找到了
4. 已经完成的任务日志：创建了 X 文件、跑了 Y 命令这些纯进度信息
5. 对话中已经明确是"不用记"的内容

**如果没有值得保留的新信息，只输出"Nothing to save."然后停止。**

**写入格式：**
- 每条记忆必须是 Markdown 列表项，以 "- " 开头
- 内容精简，一条记忆只表达一个核心信息
- 写入 user 的是关于用户个人的信息
- 写入 memory 的是关于项目、环境、经验的信息
"""

REVIEW_USER_PROMPT_TEMPLATE = """请审阅以下对话历史，判断是否有值得跨会话保留的新信息。

{history_text}

如果发现了值得保留的内容，请调用 memory 工具写入。
如果没有值得保留的新信息，只回复"Nothing to save."。"""

NOTHING_TO_SAVE_MARKER = "Nothing to save."

# ---------------------------------------------------------------------------
# 后台反思 Runner
# ---------------------------------------------------------------------------


@dataclass
class BackgroundReviewRunner:
    """后台记忆反思执行器。

    每次 spawn_review() 会启动一个 daemon 线程，将完整对话历史发给 aux model，
    通过 tool calling 调用 memory 工具写入持久记忆。
    """

    api_key: str
    base_url: str
    model_name: str
    memory_store: MemoryStore
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 0.8
    retry_backoff_multiplier: float = 2.0
    retry_max_delay_seconds: float = 4.0
    circuit_failure_threshold: int = 3
    circuit_recovery_timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "client", OpenAI(api_key=self.api_key, base_url=self.base_url))
        self.retry_policy = RetryPolicy(
            max_attempts=self.retry_max_attempts,
            base_delay_seconds=self.retry_base_delay_seconds,
            backoff_multiplier=self.retry_backoff_multiplier,
            max_delay_seconds=self.retry_max_delay_seconds,
        )
        self.circuit_breaker = CircuitBreaker(
            name="background_review",
            failure_threshold=self.circuit_failure_threshold,
            recovery_timeout_seconds=self.circuit_recovery_timeout_seconds,
        )
        # memory 工具定义，传给 aux model 做 tool calling
        self._memory_tool_def = {
            "type": "function",
            "function": {
                "name": "memory",
                "description": (
                    "写入持久记忆。target='user' 写入用户记忆（偏好、身份、风格要求），"
                    "target='memory' 写入项目记忆（规范、环境、经验教训）。"
                    "content 必须是 Markdown 列表项，以 '- ' 开头。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add"],
                            "description": "固定为 add",
                        },
                        "target": {
                            "type": "string",
                            "enum": ["memory", "user"],
                            "description": "写入目标：memory 或 user",
                        },
                        "content": {
                            "type": "string",
                            "description": "要写入的记忆内容，以 '- ' 开头的 Markdown 列表项",
                        },
                    },
                    "required": ["action", "target", "content"],
                },
            },
        }

    def spawn_review(self, history: list[ChatMessage], *, session_id: str = "") -> None:
        """启动 daemon 线程执行后台反思。"""
        snapshot = [dict(message) for message in history]
        thread = threading.Thread(
            target=self._run_review_thread,
            args=(snapshot, session_id),
            name=f"memory-review-{session_id or 'session'}",
            daemon=True,
        )
        thread.start()

    # -------------------------------------------------------------------
    # 内部实现
    # -------------------------------------------------------------------

    def _run_review_thread(self, history: list[ChatMessage], session_id: str) -> None:
        """在 daemon 线程中执行反思：调模型 → 解析 tool_calls → 写入 MemoryStore。"""
        try:
            history_text = self._build_history_payload(history)
            user_prompt = REVIEW_USER_PROMPT_TEMPLATE.format(history_text=history_text)

            tool_calls = self._call_review_model(user_prompt)
            if not tool_calls:
                return

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                if tool_name != "memory":
                    continue
                arguments = tool_call.get("arguments", {})
                action = str(arguments.get("action", "")).strip().lower()
                target = str(arguments.get("target", "")).strip().lower()
                content = str(arguments.get("content", "")).rstrip()
                if action != "add" or target not in {"memory", "user"} or not content:
                    continue
                result = self.memory_store.add(
                    target=target,
                    content=content,
                    bypass_approval=True,
                )
                if not bool(result.get("success", False)):
                    log_event(
                        f"[session={session_id or '-'}] 后台反思写入失败 [{target}]: "
                        f"{result.get('error', '')}",
                        echo=False,
                    )
        except Exception as error:
            log_event(
                f"[session={session_id or '-'}] 后台反思线程异常: {error}",
                echo=False,
            )

    def _call_review_model(self, user_prompt: str) -> list[dict[str, Any]]:
        """调用 aux model 执行记忆反思，返回解析后的 tool_calls 列表。"""
        if not self.circuit_breaker.allow_request():
            raise CircuitOpenError(self.circuit_breaker.reject_reason())

        def _request_model() -> object:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[self._memory_tool_def],
                tool_choice="auto",
                extra_body={"thinking": {"type": "disabled"}},
            )

        try:
            response = run_with_retry(
                _request_model,
                policy=self.retry_policy,
                should_retry=should_retry_model_error,
                on_retry=lambda attempt, error, delay: log_event(
                    (
                        f"后台反思调用失败，准备第 {attempt + 1} 次尝试："
                        f"{type(error).__name__}: {error}，等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.circuit_breaker.record_failure(error)
            raise

        self.circuit_breaker.record_success()
        message = response.choices[0].message

        # 检查模型是否认为无需保存
        text_content = str(message.content or "").strip()
        if NOTHING_TO_SAVE_MARKER in text_content:
            return []

        # 解析 tool_calls
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        result: list[dict[str, Any]] = []
        import json

        for tc in raw_tool_calls:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            name = getattr(fn, "name", "")
            args_str = getattr(fn, "arguments", "{}") or "{}"
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                continue
            result.append({"name": name, "arguments": arguments})
        return result

    def _build_history_payload(self, history: list[ChatMessage]) -> str:
        """将对话历史压缩为反思模型可用的文本。"""
        lines: list[str] = []
        for message in history:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                # 系统提示词太长，只取前两行摘要
                brief = content.split("\n")[:2]
                lines.append("[system] " + " ".join(brief)[:200])
            elif role == "assistant_tool_call":
                tool_name = str(message.get("tool_name", "")).strip() or "unknown"
                lines.append(f"[调用工具:{tool_name}]")
            elif role == "tool_result":
                tool_name = str(message.get("tool_name", "")).strip() or "unknown"
                error_flag = bool(message.get("is_error", False))
                prefix = "工具失败" if error_flag else "工具返回"
                lines.append(f"[{prefix}:{tool_name}] {content[:300]}")
            elif role == "user":
                lines.append(f"[用户] {content[:500]}")
            elif role == "assistant":
                lines.append(f"[助手] {content[:500]}")
        return "\n".join(lines).strip()


def build_review_runner(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    memory_store: MemoryStore | None = None,
    retry_max_attempts: int = 3,
    retry_base_delay_seconds: float = 0.8,
    retry_backoff_multiplier: float = 2.0,
    retry_max_delay_seconds: float = 4.0,
    circuit_failure_threshold: int = 3,
    circuit_recovery_timeout_seconds: float = 45.0,
) -> BackgroundReviewRunner:
    """创建后台反思 runner 的工厂函数。"""
    return BackgroundReviewRunner(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        memory_store=memory_store or get_memory_store(),
        retry_max_attempts=retry_max_attempts,
        retry_base_delay_seconds=retry_base_delay_seconds,
        retry_backoff_multiplier=retry_backoff_multiplier,
        retry_max_delay_seconds=retry_max_delay_seconds,
        circuit_failure_threshold=circuit_failure_threshold,
        circuit_recovery_timeout_seconds=circuit_recovery_timeout_seconds,
    )
