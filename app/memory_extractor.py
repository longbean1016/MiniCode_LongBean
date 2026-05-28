from __future__ import annotations

from app.memory_reflection import TaskMemoryReflectionEngine, TaskReflectionInput
from app.memory_store import MemoryEntry, create_memory_entry
from app.types import AgentStep, ChatMessage


class LongTermMemoryExtractor:
    """
    长期记忆抽取器。

    当前主链路：
    - 自动长期记忆不是按 turn 直接抽
    - 而是先做 task reflection
    - 自动写入时统一视为 project scope
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> None:
        # reflection_engine: 真正负责 task-based reflection 的引擎。
        self.reflection_engine = TaskMemoryReflectionEngine(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
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
        - task_description: 当前任务描述，一般就是本轮用户输入
        - final_step: 本轮最终 assistant 输出
        - turn_messages: 当前轮完整消息链
        - session_id: 当前会话 id
        - key_decisions: 关键决策列表
        - failures: 失败/报错/风险列表
        - files_touched: 涉及文件列表

        注意：
        - 这里只负责把反思结果转成 MemoryEntry
        - 不在这里做写入门槛判断
        - 自动链路的 scope 固定写成 project
        """
        reflection_input = TaskReflectionInput(
            task_description=task_description,
            final_step=final_step,
            turn_messages=turn_messages,
            key_decisions=list(key_decisions or []),
            failures=list(failures or []),
            files_touched=list(files_touched or []),
        )

        # candidates: 反思模型返回的候选长期记忆。
        candidates = self.reflection_engine.reflect(reflection_input)

        entries: list[MemoryEntry] = []

        for candidate in candidates:
            entries.append(
                create_memory_entry(
                    content=candidate.content,
                    category=candidate.category,
                    tags=candidate.tags,
                    session_id=session_id,
                    extra={
                        # source: 这条记忆来自 task reflection 自动提取。
                        "source": "task_reflection",

                        # scope: 严格对齐 minicode 当前主链路，自动写入只写 project。
                        "scope": "project",

                        # domains: 领域标签，后续做检索和 rerank 时可复用。
                        "domains": list(candidate.domains),

                        # confidence: 模型给出的长期记忆置信度。
                        "confidence": candidate.confidence,
                    },
                )
            )

        return entries