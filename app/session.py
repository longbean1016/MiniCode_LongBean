from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.types import ChatMessage


# 会话文件统一存到项目根目录下的 .sessions 文件夹
SESSIONS_DIR_NAME = ".sessions"


@dataclass(slots=True)
class SessionMeta:
    """会话元信息：用于列表展示、最近会话恢复、索引检索。"""

    session_id: str  # 当前会话唯一标识
    created_at: float  # 会话创建时间戳
    updated_at: float  # 会话最后更新时间戳
    workspace: str  # 会话所属工作目录
    message_count: int = 0  # 当前会话消息数量
    first_user_message: str = ""  # 第一条用户消息摘要
    last_message: str = ""  # 最后一条消息摘要

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
    """完整会话数据：后续保存/加载时主要序列化这个对象。"""

    session_id: str  # 当前会话唯一标识
    created_at: float  # 会话创建时间戳
    updated_at: float  # 会话最后更新时间戳
    workspace: str  # 会话所属工作目录
    messages: list[ChatMessage] = field(default_factory=list)  # 完整消息历史
    meta: SessionMeta | None = None  # 会话元信息
    extra: dict[str, Any] = field(default_factory=dict)  # 预留扩展字段

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

            # 旧数据恢复时如果已有消息，补齐摘要和计数字段，但不改时间
            if self.messages:
                self.sync_meta_fields()

    def sync_meta_fields(self) -> None:
        """只同步摘要和计数，不修改 updated_at。"""
        # 防御式兜底
        if self.meta is None:
            self.meta = SessionMeta(
                session_id=self.session_id,
                created_at=self.created_at,
                updated_at=self.updated_at,
                workspace=self.workspace,
            )

        # 同步基础字段
        self.meta.updated_at = self.updated_at
        self.meta.workspace = self.workspace
        self.meta.message_count = len(self.messages)

        # 先清空摘要，避免旧值残留
        self.meta.first_user_message = ""
        self.meta.last_message = ""

        # 找到第一条 user 消息，作为会话开头摘要
        for msg in self.messages:
            if msg.get("role") == "user":
                self.meta.first_user_message = str(msg.get("content", ""))[:100]
                break

        # 从后往前找最后一条有内容的消息，作为会话结尾摘要
        for msg in reversed(self.messages):
            content = str(msg.get("content", "")).strip()
            if content:
                self.meta.last_message = content[:100]
                break

    def refresh_meta(self) -> None:
        """根据当前消息刷新元信息，并更新时间。"""
        # 只在真实会话变更时刷新 updated_at
        self.updated_at = time.time()
        self.sync_meta_fields()

    def append_message(self, message: ChatMessage) -> None:
        """追加一条消息，并立即刷新元信息。"""
        self.messages.append(message)
        self.refresh_meta()

    def replace_messages(self, messages: list[ChatMessage]) -> None:
        """批量替换完整消息历史，并刷新元信息。"""
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

        # 过滤脏数据，只接收 dict 形态的消息
        for item in raw_messages:
            if isinstance(item, dict):
                messages.append(item)  # type: ignore[arg-type]

        session = cls(
            session_id=str(data["session_id"]),
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
            workspace=str(data["workspace"]),
            messages=messages,
            meta=meta,
            extra=dict(data.get("extra", {})),
        )

        # 只有显式要求时才刷新，避免覆盖磁盘中的原始时间
        if refresh:
            session.refresh_meta()

        return session


def create_new_session(workspace: str) -> SessionData:
    """创建一个新的空会话。"""
    now = time.time()
    session_id = uuid.uuid4().hex[:12]

    session = SessionData(
        session_id=session_id,
        created_at=now,
        updated_at=now,
        workspace=workspace,
        messages=[],
    )

    # 新会话创建后补一遍摘要字段
    session.refresh_meta()
    return session


def get_sessions_dir(workspace: str) -> Path:
    """返回当前工作区下的会话目录。"""
    return Path(workspace).resolve() / SESSIONS_DIR_NAME


def get_session_file_path(workspace: str, session_id: str) -> Path:
    """返回某个会话对应的 JSON 文件路径。"""
    return get_sessions_dir(workspace) / f"{session_id}.json"


def save_session(session: SessionData) -> Path:
    """把会话完整保存到本地 JSON 文件。"""
    # 保存前只同步摘要和计数，不篡改最后会话变更时间
    session.sync_meta_fields()

    # 确保会话目录存在
    sessions_dir = get_sessions_dir(session.workspace)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # 计算正式文件路径和临时文件路径
    session_file = get_session_file_path(session.workspace, session.session_id)
    temp_file = session_file.with_suffix(".json.tmp")

    # 先写临时文件，避免写一半中断导致正式文件损坏
    temp_file.write_text(
        json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 原子替换正式文件
    temp_file.replace(session_file)
    return session_file


def load_session(workspace: str, session_id: str) -> SessionData | None:
    """按 session_id 从本地 JSON 文件加载会话。"""
    session_file = get_session_file_path(workspace, session_id)

    # 文件不存在，直接返回 None
    if not session_file.exists():
        return None

    try:
        # 读取 JSON 文本
        raw_text = session_file.read_text(encoding="utf-8")

        # 反序列化成字典
        raw_data = json.loads(raw_text)

        # 从字典恢复 SessionData
        if not isinstance(raw_data, dict):
            return None

        return SessionData.from_dict(raw_data, refresh=False)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        # 文件损坏或数据格式异常时，不让主流程崩
        return None


def list_sessions(workspace: str) -> list[SessionMeta]:
    """列出当前工作区下所有会话的元信息。"""
    sessions_dir = get_sessions_dir(workspace)

    # 没有目录时返回空列表
    if not sessions_dir.exists():
        return []

    metas: list[SessionMeta] = []

    # 遍历所有 json 会话文件
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

            # 兼容没有 meta 的旧数据
            session = SessionData.from_dict(raw_data, refresh=False)
            if session.meta is not None:
                metas.append(session.meta)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            # 跳过损坏文件
            continue

    # 按更新时间倒序，最近的排前面
    metas.sort(key=lambda item: item.updated_at, reverse=True)
    return metas


def format_session_list(metas: list[SessionMeta]) -> str:
    """把会话列表格式化成适合终端展示的文本。"""
    # 没有可恢复会话时直接返回提示
    if not metas:
        return "当前工作区没有可恢复的会话。"

    # 收集最终输出的每一行
    lines: list[str] = []

    # 先输出标题
    lines.append("可恢复的会话列表（按最近更新时间倒序）：")

    # 逐条格式化会话信息
    for index, meta in enumerate(metas, start=1):
        # 把更新时间格式化到分钟，不显示秒
        updated_text = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(meta.updated_at),
        )

        # 使用第一条用户消息作为摘要；没有则标记为空会话
        summary = meta.first_user_message or "(空会话)"

        # 摘要只展示前 15 个字符，避免列表过长
        if len(summary) > 15:
            summary = f"{summary[:15]}..."

        # 第一行展示核心字段
        lines.append(
            f"{index}. session_id={meta.session_id} | "
            f"updated_at={updated_text} | "
            f"messages={meta.message_count}"
        )

        # 第二行展示摘要
        lines.append(f"first_user_message={summary}")

    # 拼接成最终字符串返回
    return "\n".join(lines)


def get_latest_session(workspace: str) -> SessionData | None:
    """获取当前工作区最近一次更新的会话。"""
    metas = list_sessions(workspace)

    # 没有任何会话时返回 None
    if not metas:
        return None

    # 取更新时间最新的第一条
    latest_meta = metas[0]
    return load_session(workspace, latest_meta.session_id)
