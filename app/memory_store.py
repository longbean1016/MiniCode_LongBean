

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Protocol
import uuid


MEMORY_DIR_NAME = ".memory"
MEMORY_FILE_NAME = "memory.json"

def _normalize_text(text: str)->str:
    """
    把文本做最基础的标准化，便于后面做简单检索。
    """

    return " ".join(text.strip().lower().split())



def _tokenize(text: str)->list[str]:
    """
    把文本拆成简单关键词列表。

    第一版先不做复杂分词，只按空白切分。
    后面如果要升级 TF-IDF / 向量检索，可以替换这里。
    """
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


@dataclass(slots=True)
class MemoryEntry:
    """
    一条长期记忆。

    这类记忆不是完整聊天记录，而是抽取后的高价值信息，
    比如用户偏好、项目约定、任务结论、失败经验等。
    """

    id: str  # 记忆唯一 ID
    content: str # 记忆正文
    category: str="note"   # 记忆分类，例如 preference / convention / conclusion
    tags: list[str] = field(default_factory=list)  # 标签，便于后续过滤和检索
    created_at: float = field(default_factory=time.time)  # 创建时间戳
    updated_at: float = field(default_factory=time.time)  # 最后更新时间戳
    session_id: str = ""  # 这条记忆来自哪个会话
    extra: dict[str, Any] = field(default_factory=dict)  # 预留扩展字段

    def to_dict(self) -> dict[str, Any]:
        """
        转成可写入 JSON 的普通字典。
        """
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "session_id": self.session_id,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """
        从字典恢复一条 MemoryEntry。
        """
        return cls(
            id=str(data["id"]),
            content=str(data["content"]),
            category=str(data.get("category", "note")),
            tags=[
                str(item).strip()
                for item in data.get("tags", [])
                if str(item).strip()
            ],
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            session_id=str(data.get("session_id", "")),
            extra=dict(data.get("extra", {})),
        )
    
class MemoryStore(Protocol):
    """
    长期记忆存储接口。

    后面如果你要替换成 sqlite-vss、FAISS、Chroma，
    只需要换实现，不需要改上层调用逻辑。
    """

    def load_memories(self) -> list[MemoryEntry]:
        """加载全部长期记忆。"""
        ...

    def save_memories(self, entries: list[MemoryEntry]) -> None:
        """保存全部长期记忆。"""
        ...

    def add_memory(self, entry: MemoryEntry) -> MemoryEntry:
        """新增一条长期记忆。"""
        ...

    def search_memories(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """按查询语句返回最相关的若干条长期记忆。"""
        ...


class JsonMemoryStore:
    """
    基于本地 JSON 文件的长期记忆存储实现。

    第一版先走最简单方案：
    - 数据持久化到工作区下的 .memory/memory.json
    - 检索先用简单关键词打分
    """

    def __init__(self, workspace: str) -> None:
        # workspace 是当前项目工作区根目录
        self.workspace = str(Path(workspace).resolve())

        # 记忆目录，例如 D:\MiniCode-ByMyself\.memory
        self.memory_dir = Path(self.workspace) / MEMORY_DIR_NAME

        # 记忆文件，例如 D:\MiniCode-ByMyself\.memory\memory.json
        self.memory_file = self.memory_dir / MEMORY_FILE_NAME

    def _ensure_dir(self) -> None:
        """
        确保记忆目录存在。
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _read_raw_entries(self) -> list[dict[str, Any]]:
        """
        从 JSON 文件读取原始字典列表。

        如果文件不存在或损坏，直接兜底为空列表。
        """
        if not self.memory_file.exists():
            return []

        try:
            raw_text = self.memory_file.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)

            if not isinstance(raw_data, dict):
                return []

            entries = raw_data.get("entries", [])
            if not isinstance(entries, list):
                return []

            return [item for item in entries if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []

    def load_memories(self) -> list[MemoryEntry]:
        """
        加载全部长期记忆，并转成 MemoryEntry 列表。
        """
        raw_entries = self._read_raw_entries()
        result: list[MemoryEntry] = []

        for item in raw_entries:
            try:
                result.append(MemoryEntry.from_dict(item))
            except (KeyError, TypeError, ValueError):
                # 跳过脏数据，避免整份记忆文件因为一条坏数据就全崩
                continue

        return result

    def save_memories(self, entries: list[MemoryEntry]) -> None:
        """
        把全部长期记忆完整写回 JSON 文件。
        """
        self._ensure_dir()

        payload = {
            "entries": [entry.to_dict() for entry in entries],
        }

        temp_file = self.memory_file.with_suffix(".json.tmp")

        # 先写临时文件，再原子替换，避免写一半中断导致正式文件损坏
        temp_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_file.replace(self.memory_file)

    def add_memory(self, entry: MemoryEntry) -> MemoryEntry:
        """
        新增一条长期记忆，并保存到本地文件。
        """
        entries = self.load_memories()

        # 如果没有 id，就自动生成一个
        if not entry.id.strip():
            entry.id = uuid.uuid4().hex[:12]

        # 新增时同步更新时间戳
        now = time.time()
        if not entry.created_at:
            entry.created_at = now
        entry.updated_at = now

        entries.append(entry)
        self.save_memories(entries)
        return entry

    def search_memories(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        按查询语句检索最相关的长期记忆。

        第一版先用简单关键词重叠打分：
        - query 和 memory 的 token 重叠越多，分数越高
        - title/tag/category 命中可额外加一点分
        """
        normalized_query = _normalize_text(query)
        if not normalized_query:
            return []

        query_tokens = set(_tokenize(normalized_query))
        if not query_tokens:
            return []

        scored_entries: list[tuple[float, MemoryEntry]] = []

        for entry in self.load_memories():
            content_tokens = set(_tokenize(entry.content))
            if not content_tokens:
                continue

            # 基础分：query token 和 content token 的重叠数
            overlap = query_tokens & content_tokens
            score = float(len(overlap))

            # category 命中时额外加分
            if entry.category and _normalize_text(entry.category) in normalized_query:
                score += 0.5

            # tags 命中时额外加分
            for tag in entry.tags:
                normalized_tag = _normalize_text(tag)
                if normalized_tag and normalized_tag in normalized_query:
                    score += 0.5

            # 分数大于 0 才认为相关
            if score > 0:
                scored_entries.append((score, entry))

        # 分数高的排前面；同分时更新更近的排前面
        scored_entries.sort(
            key=lambda item: (item[0], item[1].updated_at),
            reverse=True,
        )

        return [entry for _, entry in scored_entries[:top_k]]


def create_memory_entry(
    content: str,
    category: str = "note",
    tags: list[str] | None = None,
    session_id: str = "",
    extra: dict[str, Any] | None = None,
) -> MemoryEntry:
    """
    创建一条新的长期记忆，方便上层调用时少写样板代码。
    """
    now = time.time()

    return MemoryEntry(
        id=uuid.uuid4().hex[:12],
        content=content.strip(),
        category=category.strip() or "note",
        tags=[tag.strip() for tag in (tags or []) if tag.strip()],
        created_at=now,
        updated_at=now,
        session_id=session_id.strip(),
        extra=dict(extra or {}),
    )



