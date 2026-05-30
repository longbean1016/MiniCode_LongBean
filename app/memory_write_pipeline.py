from __future__ import annotations

from app.explicit_memory import parse_manual_memory_input
from app.logger import log_event
from app.memory_decay import DecayRunResult
from app.memory_guard import MemoryWriteGuard
from app.memory_models import (
    ExplicitMemoryHandleResult,
    MemoryCuratorLike,
    MemoryDecayLike,
    MemoryExtractorLike,
    MemoryVerifierLike,
    MemoryWriteResult,
)
from app.memory_reflection_policy import decide_project_reflection
from app.memory_store import MemoryEntry, MemoryStore
from app.types import AgentStep, ChatMessage, ToolResult
from app.working_memory import WorkingMemory
from app.working_memory_updater import (
    extract_active_paths,
    extract_project_constraints,
    extract_recent_risks,
    extract_decision_from_assistant,
    extract_user_preferences,
    summarize_failure,
)


class MemoryWritePipeline:
    _BLOCKED_REFLECTION_PREFIXES = (
        "模型调用失败:",
        "已达到最大循环步数",
        "未识别的模型返回类型",
        "模型返回了空的工具调用",
    )

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        memory_extractor: MemoryExtractorLike,
        memory_verifier: MemoryVerifierLike,
        memory_curator: MemoryCuratorLike,
        memory_decay: MemoryDecayLike,
        memory_guard: MemoryWriteGuard | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.memory_extractor = memory_extractor
        self.memory_verifier = memory_verifier
        self.memory_curator = memory_curator
        self.memory_decay = memory_decay
        self.memory_guard = memory_guard or MemoryWriteGuard()

    def reset_turn_runtime(self, working_memory: WorkingMemory) -> None:
        working_memory.clear_entries_by_type(
            "reflection_decision",
            "reflection_failure",
            "reflection_file",
        )

    def remember_user_intent(self, working_memory: WorkingMemory, user_input: str) -> None:
        working_memory.protect(
            user_input,
            entry_type="user_intent",
            ttl_seconds=3600,
            importance=1.0,
            replace_latest_of_type=True,
        )
        for preference in extract_user_preferences(user_input):
            working_memory.protect(
                preference,
                entry_type="user_preference",
                ttl_seconds=7200,
                importance=0.98,
            )
        for constraint in extract_project_constraints(user_input):
            working_memory.protect(
                constraint,
                entry_type="project_constraint",
                ttl_seconds=7200,
                importance=0.98,
            )

    def record_tool_call(
        self,
        working_memory: WorkingMemory,
        *,
        tool_name: str,
        tool_input: object,
    ) -> None:
        for path in extract_active_paths(tool_name, tool_input):
            working_memory.protect(
                path,
                entry_type="active_task",
                ttl_seconds=1800,
                importance=0.8,
            )
            working_memory.protect(
                path,
                entry_type="reflection_file",
                ttl_seconds=1800,
                importance=0.7,
            )

    def record_tool_failure(
        self,
        working_memory: WorkingMemory,
        *,
        tool_name: str,
        result: ToolResult,
    ) -> None:
        failure_summary = summarize_failure(tool_name, result)
        working_memory.protect(
            failure_summary,
            entry_type="error_context",
            ttl_seconds=1800,
            importance=0.9,
        )
        for risk in extract_recent_risks(failure_summary):
            working_memory.protect(
                risk,
                entry_type="recent_risk",
                ttl_seconds=1800,
                importance=0.92,
            )
        working_memory.protect(
            failure_summary,
            entry_type="reflection_failure",
            ttl_seconds=1800,
            importance=0.9,
        )

    def record_assistant_reply(
        self,
        working_memory: WorkingMemory,
        *,
        content: str,
    ) -> None:
        for preference in extract_user_preferences(content):
            working_memory.protect(
                preference,
                entry_type="user_preference",
                ttl_seconds=7200,
                importance=0.9,
            )
        for constraint in extract_project_constraints(content):
            working_memory.protect(
                constraint,
                entry_type="project_constraint",
                ttl_seconds=5400,
                importance=0.92,
            )
        for risk in extract_recent_risks(content):
            working_memory.protect(
                risk,
                entry_type="recent_risk",
                ttl_seconds=3600,
                importance=0.93,
            )

        decision = extract_decision_from_assistant(content)
        if not decision:
            return

        working_memory.protect(
            decision,
            entry_type="key_decision",
            ttl_seconds=3600,
            importance=0.95,
        )
        working_memory.protect(
            decision,
            entry_type="reflection_decision",
            ttl_seconds=3600,
            importance=0.95,
        )

    def handle_explicit_input(
        self,
        *,
        user_input: str,
        session_id: str,
        history: list[ChatMessage],
        decay_log_enabled: bool,
        decay_log_echo: bool,
    ) -> ExplicitMemoryHandleResult:
        intent = parse_manual_memory_input(user_input)
        if intent is None:
            return ExplicitMemoryHandleResult(
                handled=False,
                history=list(history),
            )

        if not intent.should_store:
            if intent.continue_to_agent:
                return ExplicitMemoryHandleResult(
                    handled=False,
                    history=list(history),
                )

            assistant_text = intent.ack_message or "这条输入没有被写入长期记忆。"
            return ExplicitMemoryHandleResult(
                handled=True,
                history=self._append_direct_exchange(
                    history,
                    user_input=user_input,
                    assistant_text=assistant_text,
                ),
                assistant_text=assistant_text,
            )

        candidate_entry = intent.to_memory_entry(session_id=session_id)
        stored_entries = self._persist_entries([candidate_entry])
        self._finalize_stored_memories(
            session_id=session_id,
            stored_entries=stored_entries,
            decay_log_enabled=decay_log_enabled,
            decay_log_echo=decay_log_echo,
        )

        if stored_entries:
            assistant_text = intent.ack_message or "已记录到长期记忆。"
        else:
            assistant_text = (
                "已识别为显式记忆输入，但没有写入新的长期记忆。"
                "这通常表示它与现有内容重复，或被判定为不需要新增版本。"
            )

        if intent.continue_to_agent:
            return ExplicitMemoryHandleResult(
                handled=False,
                history=list(history),
                stored_entries=stored_entries,
            )

        return ExplicitMemoryHandleResult(
            handled=True,
            history=self._append_direct_exchange(
                history,
                user_input=user_input,
                assistant_text=assistant_text,
            ),
            assistant_text=assistant_text,
            stored_entries=stored_entries,
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
        if not self._should_attempt_reflection(final_step):
            return MemoryWriteResult(
                reflection_attempted=False,
                reflection_reason="当前结果不适合进入任务反思。",
            )

        key_decisions = self._collect_reflection_entries(
            working_memory,
            "reflection_decision",
            limit=6,
        )
        failures = self._collect_reflection_entries(
            working_memory,
            "reflection_failure",
            limit=6,
        )
        files_touched = self._collect_reflection_entries(
            working_memory,
            "reflection_file",
            limit=10,
        )

        reflection_decision = decide_project_reflection(
            task_description=task_description,
            final_response=final_step.content,
            key_decisions=key_decisions,
            failures=failures,
            files_touched=files_touched,
        )
        if not reflection_decision.should_reflect:
            return MemoryWriteResult(
                reflection_attempted=False,
                reflection_reason=reflection_decision.reason,
            )

        extracted_entries = self.memory_extractor.extract_from_task(
            task_description=task_description,
            final_step=final_step,
            turn_messages=turn_messages,
            session_id=session_id,
            key_decisions=key_decisions,
            failures=failures,
            files_touched=reflection_decision.project_files_touched,
        )
        self._annotate_reflection_candidates(extracted_entries, reflection_decision)
        stored_entries = self._persist_entries(extracted_entries)
        self._finalize_stored_memories(
            session_id=session_id,
            stored_entries=stored_entries,
            decay_log_enabled=decay_log_enabled,
            decay_log_echo=decay_log_echo,
        )

        return MemoryWriteResult(
            stored_entries=stored_entries,
            extracted_entries=extracted_entries,
            reflection_attempted=True,
            reflection_reason=reflection_decision.reason,
        )

    def _should_attempt_reflection(self, final_step: AgentStep) -> bool:
        if final_step.type != "assistant":
            return False
        if final_step.kind == "progress":
            return False

        content = final_step.content.strip()
        if not content:
            return False
        return not content.startswith(self._BLOCKED_REFLECTION_PREFIXES)

    def _collect_reflection_entries(
        self,
        working_memory: WorkingMemory,
        entry_type: str,
        *,
        limit: int,
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for entry in working_memory.get_entries_by_type(entry_type):
            content = entry.content.strip()
            if not content or content in seen:
                continue
            seen.add(content)
            result.append(content)
            if len(result) >= limit:
                break

        return result

    def _persist_entries(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        existing_entries = self.memory_store.load_memories()
        stored_entries: list[MemoryEntry] = []

        for entry in entries:
            guard_decision = self.memory_guard.should_store(entry)
            if not guard_decision.should_store:
                continue

            similar_entries = self.memory_verifier.find_similar_entries(
                entry,
                existing_entries,
            )
            verify_decision = self.memory_verifier.verify(entry, similar_entries)
            if verify_decision.action not in {"store", "supersede_store"}:
                continue

            if (
                verify_decision.action == "supersede_store"
                and verify_decision.matched_memory_id.strip()
            ):
                entry.extra["supersedes_memory_id"] = verify_decision.matched_memory_id.strip()
                entry.extra["write_action"] = "supersede_store"

            stored_entry = self.memory_store.add_memory(entry)
            existing_entries.append(stored_entry)
            stored_entries.append(stored_entry)

        return stored_entries

    def _annotate_reflection_candidates(
        self,
        entries: list[MemoryEntry],
        reflection_decision: object,
    ) -> None:
        admission_source = getattr(reflection_decision, "admission_source", "blocked")
        project_files_touched = list(
            getattr(reflection_decision, "project_files_touched", []) or []
        )

        for entry in entries:
            entry.extra["reflection_admission_source"] = str(admission_source)
            entry.extra["reflection_has_project_file_evidence"] = bool(project_files_touched)
            entry.extra["project_files_touched"] = project_files_touched

    def _finalize_stored_memories(
        self,
        *,
        session_id: str,
        stored_entries: list[MemoryEntry],
        decay_log_enabled: bool,
        decay_log_echo: bool,
    ) -> None:
        if not stored_entries:
            return

        self.memory_curator.curate_new_entries(stored_entries)
        try:
            incremental_decay_result = self.memory_decay.refresh_new_entries(stored_entries)
        except Exception as error:
            log_event(
                f"[session={session_id}] decay[incremental] 执行失败: {error}",
                echo=decay_log_echo,
            )
        else:
            self._log_decay_run_result(
                session_id=session_id,
                stage="incremental",
                result=incremental_decay_result,
                enabled=decay_log_enabled,
                echo=decay_log_echo,
            )

        if self.memory_curator.should_run_full_scan():
            self.memory_curator.curate_project_memories()
        if self.memory_decay.should_run_full_refresh():
            try:
                full_decay_result = self.memory_decay.refresh_project_memories()
            except Exception as error:
                log_event(
                    f"[session={session_id}] decay[full] 执行失败: {error}",
                    echo=decay_log_echo,
                )
            else:
                self._log_decay_run_result(
                    session_id=session_id,
                    stage="full",
                    result=full_decay_result,
                    enabled=decay_log_enabled,
                    echo=decay_log_echo,
                )

    def _log_decay_run_result(
        self,
        *,
        session_id: str,
        stage: str,
        result: DecayRunResult,
        enabled: bool,
        echo: bool,
    ) -> None:
        if not enabled or result.changed_count <= 0:
            return

        log_event(
            (
                f"[session={session_id}] decay[{stage}] "
                f"scanned={result.scanned_count} "
                f"changed={result.changed_count} "
                f"archived={result.archived_count()}"
            ),
            echo=echo,
        )

    def _append_direct_exchange(
        self,
        history: list[ChatMessage],
        *,
        user_input: str,
        assistant_text: str,
    ) -> list[ChatMessage]:
        updated_history = list(history)
        updated_history.append({"role": "user", "content": user_input})
        updated_history.append({"role": "assistant", "content": assistant_text})
        return updated_history
