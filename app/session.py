from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

"""会话持久化模块，负责创建、加载、保存和列出本地会话。"""

from app.types import ChatMessage


# 会话统一保存在当前工作区下的 .sessions 目录。
SESSIONS_DIR_NAME = ".sessions"


@dataclass(slots=True)
class SessionMeta:
    """
    会话轻量元信息。

    这层只负责：
    - 会话列表展示
    - 最近会话排序
    - 快速识别会话

    不再承担“高质量上下文摘要”的职责。
    """

    session_id: str
    created_at: float
    updated_at: float
    workspace: str
    message_count: int = 0
    first_user_message: str = ""
    last_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """把 SessionMeta 转成可写入 JSON 的字典。"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workspace": self.workspace,
            "message_count": self.message_count,
            "first_user_message": self.first_user_message,
            "last_message": self.last_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMeta":
        """从字典恢复 SessionMeta。"""
        return cls(
            session_id=str(data["session_id"]),
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
            workspace=str(data["workspace"]),
            message_count=int(data.get("message_count", 0)),
            first_user_message=str(data.get("first_user_message", "")),
            last_message=str(data.get("last_message", "")),
        )


@dataclass(slots=True)
class SessionData:
    """
    完整会话数据。

    messages:
        保存完整原始消息历史。
    meta:
        只保存轻量展示信息。
    extra:
        保存上下文压缩状态等扩展字段。
    """

    session_id: str
    created_at: float
    updated_at: float
    workspace: str
    messages: list[ChatMessage] = field(default_factory=list)
    meta: SessionMeta | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化后补齐 meta，但不主动刷新时间。"""
        if self.meta is None:
            self.meta = SessionMeta(
                session_id=self.session_id,
                created_at=self.created_at,
                updated_at=self.updated_at,
                workspace=self.workspace,
                message_count=len(self.messages),
            )

            # 旧数据恢复时如果已经有消息，顺手同步轻量 meta 字段。
            if self.messages:
                self.sync_meta_fields()

    def sync_meta_fields(self) -> None:
        """
        只同步轻量 meta 字段，不修改 updated_at。

        这里故意不再生成 session summary，
        因为高质量摘要已经迁移到 SessionData.extra 的 context state 中。
        """
        if self.meta is None:
            self.meta = SessionMeta(
                session_id=self.session_id,
                created_at=self.created_at,
                updated_at=self.updated_at,
                workspace=self.workspace,
            )

        self.meta.updated_at = self.updated_at
        self.meta.workspace = self.workspace
        self.meta.message_count = len(self.messages)
        self.meta.first_user_message = ""
        self.meta.last_message = ""

        # 第一条 user 消息用于会话列表展示。
        for message in self.messages:
            if message.get("role") == "user":
                self.meta.first_user_message = str(message.get("content", ""))[:100]
                break

        # 最后一条有内容的消息用于快速识别最近状态。
        for message in reversed(self.messages):
            content = str(message.get("content", "")).strip()
            if content:
                self.meta.last_message = content[:100]
                break

    def refresh_meta(self) -> None:
        """真实会话发生变更时，刷新 updated_at 和轻量 meta。"""
        self.updated_at = time.time()
        self.sync_meta_fields()

    def append_message(self, message: ChatMessage) -> None:
        """追加一条消息并刷新元信息。"""
        self.messages.append(message)
        self.refresh_meta()

    def replace_messages(self, messages: list[ChatMessage]) -> None:
        """整批替换消息历史并刷新元信息。"""
        self.messages = list(messages)
        self.refresh_meta()

    def to_dict(self) -> dict[str, Any]:
        """把 SessionData 转成可写入 JSON 的字典。"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workspace": self.workspace,
            "messages": list(self.messages),
            "meta": self.meta.to_dict() if self.meta else None,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], refresh: bool = False) -> "SessionData":
        """从字典恢复 SessionData。"""
        raw_meta = data.get("meta")
        meta = SessionMeta.from_dict(raw_meta) if isinstance(raw_meta, dict) else None

        raw_messages = data.get("messages", [])
        messages: list[ChatMessage] = []

        # 只接收 dict 形式的消息，避免脏数据污染会话恢复。
        for item in raw_messages:
            if isinstance(item, dict):
                messages.append(item)  # type: ignore[arg-type]

        raw_extra = data.get("extra", {})
        extra = dict(raw_extra) if isinstance(raw_extra, dict) else {}

        session = cls(
            session_id=str(data["session_id"]),
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
            workspace=str(data["workspace"]),
            messages=messages,
            meta=meta,
            extra=extra,
        )

        # 显式要求时才刷新，避免覆盖磁盘中的原始时间戳。
        if refresh:
            session.refresh_meta()

        return session


def create_new_session(workspace: str) -> SessionData:
    """创建一个新的空会话。"""
    now = time.time()
    session = SessionData(
        session_id=uuid.uuid4().hex[:12],
        created_at=now,
        updated_at=now,
        workspace=workspace,
        messages=[],
    )
    session.refresh_meta()
    return session


def get_sessions_dir(workspace: str) -> Path:
    """返回当前工作区下的会话目录。"""
    return Path(workspace).resolve() / SESSIONS_DIR_NAME


def get_session_file_path(workspace: str, session_id: str) -> Path:
    """返回某个会话对应的 JSON 文件路径。"""
    return get_sessions_dir(workspace) / f"{session_id}.json"


def save_session(session: SessionData) -> Path:
    """把完整会话安全写入本地 JSON 文件。"""
    # 保存前只同步轻量 meta，避免把保存动作误当成新一轮会话更新。
    session.sync_meta_fields()

    sessions_dir = get_sessions_dir(session.workspace)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    session_file = get_session_file_path(session.workspace, session.session_id)
    temp_file = session_file.with_suffix(".json.tmp")

    # 先写临时文件，再原子替换正式文件，避免写到一半损坏。
    temp_file.write_text(
        json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_file.replace(session_file)
    return session_file


def load_session(workspace: str, session_id: str) -> SessionData | None:
    """按 session_id 从本地 JSON 文件恢复会话。"""
    session_file = get_session_file_path(workspace, session_id)
    if not session_file.exists():
        return None

    try:
        raw_text = session_file.read_text(encoding="utf-8")
        raw_data = json.loads(raw_text)
        if not isinstance(raw_data, dict):
            return None
        return SessionData.from_dict(raw_data, refresh=False)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def list_sessions(workspace: str) -> list[SessionMeta]:
    """列出当前工作区下的所有会话元信息。"""
    sessions_dir = get_sessions_dir(workspace)
    if not sessions_dir.exists():
        return []

    metas: list[SessionMeta] = []

    for session_file in sessions_dir.glob("*.json"):
        try:
            raw_text = session_file.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
            if not isinstance(raw_data, dict):
                continue

            raw_meta = raw_data.get("meta")
            if isinstance(raw_meta, dict):
                metas.append(SessionMeta.from_dict(raw_meta))
                continue

            # 兼容早期没有 meta 的旧数据。
            session = SessionData.from_dict(raw_data, refresh=False)
            if session.meta is not None:
                metas.append(session.meta)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    metas.sort(key=lambda item: item.updated_at, reverse=True)
    return metas


def format_session_list(metas: list[SessionMeta]) -> str:
    """把会话列表格式化成终端可读文本。"""
    if not metas:
        return "当前工作区没有可恢复的会话。"

    lines: list[str] = []
    lines.append("可恢复的会话列表（按最近更新时间倒序）：")

    for index, meta in enumerate(metas, start=1):
        updated_text = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(meta.updated_at),
        )

        summary = meta.first_user_message or "(空会话)"
        if len(summary) > 15:
            summary = f"{summary[:15]}..."

        lines.append(
            f"{index}. session_id={meta.session_id} | "
            f"updated_at={updated_text} | "
            f"messages={meta.message_count}"
        )
        lines.append(f"first_user_message={summary}")

    return "\n".join(lines)


def get_latest_session(workspace: str) -> SessionData | None:
    """获取当前工作区最近一次更新的会话。"""
    metas = list_sessions(workspace)
    if not metas:
        return None
    return load_session(workspace, metas[0].session_id)
