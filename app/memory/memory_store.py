from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


MEMORY_DIR_NAME = ".memory"
MEMORY_FILE_NAME = "MEMORY.md"
USER_FILE_NAME = "USER.md"
MEMORY_CHAR_LIMIT = 2200
USER_CHAR_LIMIT = 1375
MEMORY_TITLE = "# 项目记忆"
USER_TITLE = "# 用户记忆"
FILTERED_ENTRY_TEXT = "[已被安全扫描过滤]"

_LIST_ITEM_RE = re.compile(r"^\s*-\s+(.*\S)\s*$")
_INDENTED_LINE_RE = re.compile(r"^(?: {2,}|\t+)(.*)$")
_BLOCKED_PATTERNS = (
    (re.compile(r"role\s*:\s*system", re.IGNORECASE), '内容包含 "role: system"'),
    (re.compile(r"<\s*system\s*>", re.IGNORECASE), '内容包含 "<system>"'),
    (re.compile(r"\btool_calls\b", re.IGNORECASE), '内容包含 "tool_calls"'),
    (
        re.compile(r'"tool_name"\s*:\s*"[^"]+"', re.IGNORECASE),
        "内容包含疑似工具调用 JSON",
    ),
    (
        re.compile(r"</?\s*memory-context\s*>", re.IGNORECASE),
        '内容包含 "<memory-context>"',
    ),
)

msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
    try:
        import msvcrt
    except ImportError:  # pragma: no cover
        msvcrt = None


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _scan_memory_content(content: str) -> str | None:
    for pattern, message in _BLOCKED_PATTERNS:
        if pattern.search(content):
            return message
    return None


def _hash_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _timestamp_suffix() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


@dataclass(slots=True)
class FrozenMemorySnapshot:
    memory_entries: list[str]
    user_entries: list[str]
    memory_text: str
    user_text: str
    memory_hash: str
    user_hash: str

    def format_for_prompt(self) -> str:
        sections: list[str] = []
        if self.memory_text.strip():
            sections.append(self.memory_text.strip())
        if self.user_text.strip():
            sections.append(self.user_text.strip())
        return "\n".join(sections).strip()


class MemoryStore:
    def __init__(
        self,
        workspace: str,
        *,
        memory_char_limit: int = MEMORY_CHAR_LIMIT,
        user_char_limit: int = USER_CHAR_LIMIT,
        write_approval: bool = False,
    ) -> None:
        self.workspace = str(Path(workspace).resolve())
        self.memory_char_limit = max(1, int(memory_char_limit))
        self.user_char_limit = max(1, int(user_char_limit))
        self.write_approval = bool(write_approval)
        self.memory_dir = Path(self.workspace) / MEMORY_DIR_NAME
        self.memory_path = self.memory_dir / MEMORY_FILE_NAME
        self.user_path = self.memory_dir / USER_FILE_NAME
        self._snapshot = FrozenMemorySnapshot(
            memory_entries=[],
            user_entries=[],
            memory_text="",
            user_text="",
            memory_hash="",
            user_hash="",
        )
        self._live_memory_hash = ""
        self._live_user_hash = ""

    def ensure_files(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self._write_file(self.memory_path, [], MEMORY_TITLE)
        if not self.user_path.exists():
            self._write_file(self.user_path, [], USER_TITLE)

    def load_snapshot(self) -> FrozenMemorySnapshot:
        self.ensure_files()
        memory_raw = self.memory_path.read_text(encoding="utf-8")
        user_raw = self.user_path.read_text(encoding="utf-8")
        memory_entries = self._read_entries(memory_raw)
        user_entries = self._read_entries(user_raw)
        sanitized_memory = [self._sanitize_for_snapshot(item) for item in memory_entries]
        sanitized_user = [self._sanitize_for_snapshot(item) for item in user_entries]
        self._snapshot = FrozenMemorySnapshot(
            memory_entries=list(memory_entries),
            user_entries=list(user_entries),
            memory_text=self._render_markdown(MEMORY_TITLE, sanitized_memory),
            user_text=self._render_markdown(USER_TITLE, sanitized_user),
            memory_hash=_hash_text(memory_raw),
            user_hash=_hash_text(user_raw),
        )
        self._live_memory_hash = self._snapshot.memory_hash
        self._live_user_hash = self._snapshot.user_hash
        return self._snapshot

    def current_snapshot(self) -> FrozenMemorySnapshot:
        if not self._snapshot.memory_hash and not self._snapshot.user_hash:
            return self.load_snapshot()
        return self._snapshot

    def view(self, target: str | None = None) -> str:
        self.ensure_files()
        memory_raw = self.memory_path.read_text(encoding="utf-8")
        user_raw = self.user_path.read_text(encoding="utf-8")
        memory_text = self._render_markdown(MEMORY_TITLE, self._read_entries(memory_raw))
        user_text = self._render_markdown(USER_TITLE, self._read_entries(user_raw))
        if target == "memory":
            return memory_text
        if target == "user":
            return user_text
        return "\n".join(
            part for part in (memory_text.strip(), user_text.strip()) if part
        ).strip()

    def add(self, *, target: str, content: str, bypass_approval: bool = False) -> dict[str, object]:
        normalized_target = self._normalize_target(target)
        rendered_content = self._normalize_new_entry(content)
        scan_error = _scan_memory_content(rendered_content)
        if scan_error:
            return {"success": False, "error": f"拒绝写入：{scan_error}"}
        if self.write_approval and not bypass_approval:
            return {"success": False, "error": "当前未接入交互式记忆写入审批。"}
        return self._mutate_with_drift_retry(
            normalized_target,
            lambda entries: self._append_entry(entries, rendered_content),
        )

    def remove(self, *, target: str, old_text: str) -> dict[str, object]:
        normalized_target = self._normalize_target(target)
        needle = " ".join(old_text.strip().split())
        if not needle:
            return {"success": False, "error": "old_text 不能为空。"}
        return self._mutate_with_drift_retry(
            normalized_target,
            lambda entries: self._remove_entry(entries, needle),
        )

    def get_prompt_context(self) -> str:
        return self.current_snapshot().format_for_prompt()

    def _append_entry(self, entries: list[str], content: str) -> tuple[bool, str, list[str]]:
        for existing in entries:
            normalized_existing = _normalize_text(existing)
            normalized_content = _normalize_text(content)
            if normalized_content == normalized_existing or normalized_content in normalized_existing:
                return False, "记忆已存在，跳过重复写入。", entries
        updated_entries = list(entries)
        updated_entries.append(content)
        return True, "已追加记忆。", updated_entries

    def _remove_entry(self, entries: list[str], needle: str) -> tuple[bool, str, list[str]]:
        lowered_needle = needle.lower()
        matched_index = -1
        for index, entry in enumerate(entries):
            if lowered_needle in entry.lower():
                matched_index = index
                break
        if matched_index < 0:
            return False, f"未找到包含以下片段的记忆：{needle}", entries
        updated_entries = list(entries)
        updated_entries.pop(matched_index)
        return True, "已删除记忆。", updated_entries

    def _mutate_with_drift_retry(
        self,
        target: str,
        mutation: callable,
    ) -> dict[str, object]:
        for attempt in range(2):
            try:
                return self._mutate_once(target, mutation)
            except _ExternalDriftError as error:
                self._backup_and_reload(error.target_path)
                if attempt == 0:
                    continue
                return {"success": False, "error": str(error)}
        return {"success": False, "error": "未完成记忆写入。"}

    def _mutate_once(self, target: str, mutation: callable) -> dict[str, object]:
        self.ensure_files()
        path = self.memory_path if target == "memory" else self.user_path
        title = MEMORY_TITLE if target == "memory" else USER_TITLE
        limit = self.memory_char_limit if target == "memory" else self.user_char_limit
        with self._file_lock(path):
            self._assert_no_external_drift(target, path)
            raw_text = path.read_text(encoding="utf-8")
            entries = self._read_entries(raw_text)
            changed, message, updated_entries = mutation(entries)
            if not changed:
                return {"success": False, "error": message}
            if len(self._render_entries_only(updated_entries)) > limit:
                return {
                    "success": False,
                    "error": f"写入后超过字符上限（{limit}）。",
                }
            self._write_file(path, updated_entries, title)
            self._refresh_live_hashes()
            return {
                "success": True,
                "message": message,
                "target": target,
                "path": str(path),
            }

    def _assert_no_external_drift(self, target: str, path: Path) -> None:
        self.current_snapshot()
        expected_hash = self._live_memory_hash if target == "memory" else self._live_user_hash
        current_hash = _hash_text(path.read_text(encoding="utf-8"))
        if expected_hash and current_hash != expected_hash:
            raise _ExternalDriftError(path)

    def _backup_and_reload(self, path: Path) -> None:
        raw_text = path.read_text(encoding="utf-8")
        backup_path = path.with_name(f"{path.name}.bak.{_timestamp_suffix()}")
        backup_path.write_text(raw_text, encoding="utf-8")
        self._refresh_live_hashes()

    def _refresh_live_hashes(self) -> None:
        self.ensure_files()
        self._live_memory_hash = _hash_text(self.memory_path.read_text(encoding="utf-8"))
        self._live_user_hash = _hash_text(self.user_path.read_text(encoding="utf-8"))

    def _sanitize_for_snapshot(self, entry: str) -> str:
        return FILTERED_ENTRY_TEXT if _scan_memory_content(entry) else entry

    def _normalize_target(self, target: str) -> str:
        normalized_target = target.strip().lower()
        if normalized_target not in {"memory", "user"}:
            raise ValueError("target 必须是 memory 或 user。")
        return normalized_target

    def _normalize_new_entry(self, content: str) -> str:
        cleaned = content.rstrip()
        if not cleaned:
            raise ValueError("content 不能为空。")
        if not cleaned.lstrip().startswith("- "):
            raise ValueError("content 必须是 Markdown 列表项，且以 '- ' 开头。")
        lines = cleaned.splitlines()
        normalized_lines: list[str] = []
        for index, line in enumerate(lines):
            stripped = line.rstrip()
            if index == 0:
                stripped = "- " + stripped.lstrip()[2:].strip()
            elif stripped.strip():
                stripped = "  " + stripped.strip()
            normalized_lines.append(stripped)
        return "\n".join(normalized_lines).strip()

    def _read_entries(self, raw_text: str) -> list[str]:
        entries: list[str] = []
        current_lines: list[str] = []
        for raw_line in raw_text.splitlines():
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current_lines:
                    current_lines.append("")
                continue
            if line.strip() == "---":
                continue
            if line.lstrip().startswith("#"):
                continue
            item_match = _LIST_ITEM_RE.match(line)
            if item_match:
                if current_lines:
                    entries.append("\n".join(current_lines).rstrip())
                current_lines = [f"- {item_match.group(1).strip()}"]
                continue
            indented_match = _INDENTED_LINE_RE.match(line)
            if indented_match and current_lines:
                current_lines.append(f"  {indented_match.group(1).rstrip()}")
        if current_lines:
            entries.append("\n".join(current_lines).rstrip())
        return [entry for entry in entries if entry.strip()]

    def _render_markdown(self, title: str, entries: list[str]) -> str:
        parts = [title, ""]
        if entries:
            parts.extend(entry.rstrip() for entry in entries)
            parts.append("")
        parts.append("---")
        return "\n".join(parts).rstrip() + "\n"

    def _render_entries_only(self, entries: list[str]) -> str:
        return "\n".join(entry.rstrip() for entry in entries).strip()

    def _write_file(self, path: Path, entries: list[str], title: str) -> None:
        content = self._render_markdown(title, entries)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=".memory_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except BaseException:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    @contextmanager
    def _file_lock(self, path: Path):
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None and msvcrt is None:
            yield
            return
        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")
        handle = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            elif msvcrt is not None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                except OSError:
                    pass
            elif msvcrt is not None:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            handle.close()


class _ExternalDriftError(RuntimeError):
    def __init__(self, target_path: Path) -> None:
        self.target_path = target_path
        super().__init__(f"检测到外部修改，已拒绝直接覆盖：{target_path}")
