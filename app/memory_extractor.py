from __future__ import annotations

from app.memory_reflection import TaskMemoryReflectionEngine, TaskReflectionInput
from app.memory_store import MemoryEntry, create_memory_entry
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
    ) -> None:
        # 真正负责 task-based reflection 的引擎。
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
        - `task_description`: 当前任务描述，通常就是本轮用户输入
        - `final_step`: 本轮最终 assistant 输出
        - `turn_messages`: 当前轮完整消息链
        - `session_id`: 当前会话 id
        - `key_decisions`: 本轮关键决策列表
        - `failures`: 本轮失败、报错、阻断、风险列表
        - `files_touched`: 本轮涉及的重要文件路径列表

        这里的职责只有两件事：
        1. 把反思输入交给 reflection engine
        2. 把候选结果转成统一的 `MemoryEntry`

        真正的写入门槛判断不在这里做，而是交给 `MemoryWriteGuard`。
        """
        reflection_input = TaskReflectionInput(
            task_description=task_description,
            final_step=final_step,
            turn_messages=turn_messages,
            key_decisions=list(key_decisions or []),
            failures=list(failures or []),
            files_touched=list(files_touched or []),
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
                    extra={
                        # 记录来源，便于后面区分自动 reflection 和其他写入渠道。
                        "source": "task_reflection",
                        # 当前自动主链路严格只写 project。
                        "scope": "project",
                        # 反思模型给出的领域标签，后面做过滤和排序时可复用。
                        "domains": list(candidate.domains),
                        # 模型给出的保守置信度，后续由 guard 再做门槛判断。
                        "confidence": candidate.confidence,
                    },
                )
            )

        return entries
