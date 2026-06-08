from __future__ import annotations

from app.memory.reflection import TaskMemoryReflectionEngine, TaskReflectionInput
from app.memory.store import MemoryEntry, create_memory_entry
from app.types import AgentStep, ChatMessage


class LongTermMemoryExtractor:
    """
    长期记忆抽取器。

    当前主链路：
    - 不再按 turn 直接抽取
    - 先做 task-based reflection
    - 自动写入时统一视为 project scope
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        *,
        retry_max_attempts: int = 3,
        retry_base_delay_seconds: float = 0.8,
        retry_backoff_multiplier: float = 2.0,
        retry_max_delay_seconds: float = 4.0,
        circuit_failure_threshold: int = 3,
        circuit_recovery_timeout_seconds: float = 45.0,
    ) -> None:
        # 抽取器本身只做编排。
        # 真正的模型调用、重试和熔断逻辑都放在 reflection engine 里。
        self.reflection_engine = TaskMemoryReflectionEngine(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            retry_max_attempts=retry_max_attempts,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_backoff_multiplier=retry_backoff_multiplier,
            retry_max_delay_seconds=retry_max_delay_seconds,
            circuit_failure_threshold=circuit_failure_threshold,
            circuit_recovery_timeout_seconds=circuit_recovery_timeout_seconds,
        )

    def extract_from_task(
        self,
        *,
        task_description: str,
        final_step: AgentStep,
        turn_messages: list[ChatMessage],
        session_id: str,
        key_decisions: list[str] | None = None,
        failures: list[str] | None = None,
        files_touched: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """
        从一次任务反思中提取候选长期记忆。

        参数说明：
        - `task_description`: 当前任务描述，通常就是本轮用户输入
        - `final_step`: 本轮最终 assistant 输出
        - `turn_messages`: 当前轮完整消息链
        - `session_id`: 当前会话 id
        - `key_decisions`: 本轮关键决策列表
        - `failures`: 本轮失败、报错、阻断、风险列表
        - `files_touched`: 本轮涉及的重要项目文件路径列表
        """
        normalized_key_decisions = list(key_decisions or [])
        normalized_failures = list(failures or [])
        normalized_files_touched = list(files_touched or [])

        reflection_input = TaskReflectionInput(
            task_description=task_description,
            final_step=final_step,
            turn_messages=turn_messages,
            key_decisions=normalized_key_decisions,
            failures=normalized_failures,
            files_touched=normalized_files_touched,
        )

        candidates = self.reflection_engine.reflect(reflection_input)
        entries: list[MemoryEntry] = []

        for candidate in candidates:
            entries.append(
                create_memory_entry(
                    content=candidate.content,
                    category=candidate.category,
                    tags=candidate.tags,
                    session_id=session_id,
                    source="task_reflection",
                    scope="project",
                    domains=list(candidate.domains),
                    confidence=candidate.confidence,
                    extra={
                        # 给 guard / verifier / 后续审计保留结构化证据，
                        # 避免再回退到主题关键词判断。
                        "project_files_touched": normalized_files_touched,
                        "reflection_failure_count": len(normalized_failures),
                        "reflection_decision_count": len(normalized_key_decisions),
                    },
                )
            )

        return entries
