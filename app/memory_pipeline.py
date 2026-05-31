from __future__ import annotations

from app.memory_feedback import MemoryFeedbackStore
from app.memory_models import (
    ExplicitMemoryHandleResult,
    MemoryContextResult,
    MemoryWriteResult,
)

# 记忆流水线编排层，统一协调读写、反馈与显式记忆入口。
from app.memory_read_pipeline import MemoryReadPipeline
from app.memory_write_pipeline import MemoryWritePipeline
from app.session import SessionData
from app.types import AgentStep, ChatMessage, ToolResult
from app.working_memory import WorkingMemory

# 这是记忆系统的总编排层：
# 读链路决定“给模型看什么”，
# 写链路决定“沉淀什么”，
# 反馈链路记录“这轮注入的记忆有没有帮上忙”。


class MemoryPipeline:
    _INJECTED_MEMORY_ID_ENTRY_TYPE = "injected_memory_id"

    def __init__(
        self,
        *,
        read_pipeline: MemoryReadPipeline,
        write_pipeline: MemoryWritePipeline,
        feedback_store: MemoryFeedbackStore | None = None,
    ) -> None:
        self.read_pipeline = read_pipeline
        self.write_pipeline = write_pipeline
        self.feedback_store = feedback_store

    def reset_turn_runtime(self, working_memory: WorkingMemory) -> None:
        # 清理只在单轮内有效的运行态痕迹，避免跨轮串味。
        self.write_pipeline.reset_turn_runtime(working_memory)
        working_memory.clear_entries_by_type(self._INJECTED_MEMORY_ID_ENTRY_TYPE)

    def remember_user_intent(self, working_memory: WorkingMemory, user_input: str) -> None:
        # 总管道不直接解析用户意图细节，而是把这件事委托给写链路，
        # 保证“即时工作记忆更新”和“后续长期沉淀策略”仍由同一处维护。
        self.write_pipeline.remember_user_intent(working_memory, user_input)

    def build_prompt_context(
        self,
        *,
        user_input: str,
        session: SessionData,
        working_memory: WorkingMemory,
        session_summary_override: str = "",
        top_k: int = 4,
        retrieval_top_k: int = 8,
        max_memory_chars_per_item: int = 180,
    ) -> MemoryContextResult:
        # 这里只记住“本轮到底注入了哪些长期记忆”，
        # 方便回合结束时做效果反馈。
        result = self.read_pipeline.build_context(
            user_input=user_input,
            session=session,
            working_memory=working_memory,
            session_summary_override=session_summary_override,
            top_k=top_k,
            retrieval_top_k=retrieval_top_k,
            max_memory_chars_per_item=max_memory_chars_per_item,
        )
        self._remember_injected_entries(working_memory, result.injected_entries)
        return result

    def record_tool_call(
        self,
        working_memory: WorkingMemory,
        *,
        tool_name: str,
        tool_input: object,
    ) -> None:
        self.write_pipeline.record_tool_call(
            working_memory,
            tool_name=tool_name,
            tool_input=tool_input,
        )

    def record_tool_failure(
        self,
        working_memory: WorkingMemory,
        *,
        tool_name: str,
        result: ToolResult,
    ) -> None:
        self.write_pipeline.record_tool_failure(
            working_memory,
            tool_name=tool_name,
            result=result,
        )

    def record_assistant_reply(
        self,
        working_memory: WorkingMemory,
        *,
        content: str,
    ) -> None:
        self.write_pipeline.record_assistant_reply(working_memory, content=content)

    def handle_explicit_input(
        self,
        *,
        user_input: str,
        session_id: str,
        history: list[ChatMessage],
        decay_log_enabled: bool,
        decay_log_echo: bool,
    ) -> ExplicitMemoryHandleResult:
        return self.write_pipeline.handle_explicit_input(
            user_input=user_input,
            session_id=session_id,
            history=history,
            decay_log_enabled=decay_log_enabled,
            decay_log_echo=decay_log_echo,
        )

    def reflect_and_persist(
        self,
        *,
        task_description: str,
        final_step: AgentStep,
        turn_messages: list[ChatMessage],
        session_id: str,
        working_memory: WorkingMemory,
        decay_log_enabled: bool,
        decay_log_echo: bool,
    ) -> MemoryWriteResult:
        return self.write_pipeline.reflect_and_persist(
            task_description=task_description,
            final_step=final_step,
            turn_messages=turn_messages,
            session_id=session_id,
            working_memory=working_memory,
            decay_log_enabled=decay_log_enabled,
            decay_log_echo=decay_log_echo,
        )

    def record_outcome_feedback(self, working_memory: WorkingMemory, *, success: bool) -> None:
        if self.feedback_store is None:
            return

        memory_ids = self._collect_injected_memory_ids(working_memory)
        if not memory_ids:
            return

        # 反馈记的是“被实际注入 prompt 的记忆最终是否帮助任务完成”。
        self.feedback_store.record(memory_ids, success=success)
        working_memory.clear_entries_by_type(self._INJECTED_MEMORY_ID_ENTRY_TYPE)

    def finalize_turn(
        self,
        *,
        task_description: str,
        final_step: AgentStep,
        turn_messages: list[ChatMessage],
        session_id: str,
        working_memory: WorkingMemory,
        decay_log_enabled: bool,
        decay_log_echo: bool,
    ) -> MemoryWriteResult:
        # 统一回合收口：先反馈，再做反思和长期写入。
        self.record_outcome_feedback(
            working_memory,
            success=self._is_successful_outcome(final_step),
        )
        return self.reflect_and_persist(
            task_description=task_description,
            final_step=final_step,
            turn_messages=turn_messages,
            session_id=session_id,
            working_memory=working_memory,
            decay_log_enabled=decay_log_enabled,
            decay_log_echo=decay_log_echo,
        )

    def _remember_injected_entries(
        self,
        working_memory: WorkingMemory,
        entries: list[object],
    ) -> None:
        existing_ids = set(self._collect_injected_memory_ids(working_memory))
        for entry in entries:
            memory_id = str(getattr(entry, "id", "")).strip()
            if not memory_id or memory_id in existing_ids:
                continue

            existing_ids.add(memory_id)
            # 这里只保存 memory id，而不是整段正文，避免 working memory 膨胀。
            working_memory.protect(
                memory_id,
                entry_type=self._INJECTED_MEMORY_ID_ENTRY_TYPE,
                ttl_seconds=3600,
                importance=0.2,
            )

    def _collect_injected_memory_ids(self, working_memory: WorkingMemory) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for entry in working_memory.get_entries_by_type(self._INJECTED_MEMORY_ID_ENTRY_TYPE):
            memory_id = entry.content.strip()
            if not memory_id or memory_id in seen:
                continue

            seen.add(memory_id)
            result.append(memory_id)

        return result

    def _is_successful_outcome(self, final_step: AgentStep) -> bool:
        if final_step.type != "assistant":
            return False

        content = final_step.content.strip()
        if not content:
            return False

        # 这里只区分“正常完成”还是“异常收尾”，不在这里评价答案质量。
        blocked_prefixes = (
            "模型调用失败:",
            "已达到最大循环步数",
            "未识别的模型返回类型",
            "模型返回了空的工具调用",
        )
        return not content.startswith(blocked_prefixes)
