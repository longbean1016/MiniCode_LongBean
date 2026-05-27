from __future__ import annotations

from typing import Iterable

from app.memory_store import MemoryEntry, MemoryStore
from app.session import SessionData
from app.working_memory import WorkingMemory


def _shorten(text: str, max_chars: int) -> str:
    """
    把一段文本截短，避免注入 prompt 的内容过长。
    """
    # 先做基础清洗，去掉首尾空格和多余换行
    cleaned = " ".join(text.strip().split())

    # 空文本直接返回空字符串
    if not cleaned:
        return ""

    # 长度没超出时原样返回
    if len(cleaned) <= max_chars:
        return cleaned

    # 超出时截断并补省略号
    return cleaned[:max_chars].rstrip() + "..."


def _dedupe_memories(entries: Iterable[MemoryEntry]) -> list[MemoryEntry]:
    """
    对长期记忆做简单去重，避免同一条内容重复注入。
    """
    # seen 用来记录已经收过的“去空白后内容”
    seen: set[str] = set()

    # result 保存最终去重后的记忆列表
    result: list[MemoryEntry] = []

    for entry in entries:
        # 用 content 做第一版去重键，后面如果你要升级可以改成 id 或 hash
        normalized_content = " ".join(entry.content.strip().split())

        # 空内容直接跳过
        if not normalized_content:
            continue

        # 已经出现过就不再重复加入
        if normalized_content in seen:
            continue

        seen.add(normalized_content)
        result.append(entry)

    return result


def _format_long_term_memories(
    entries: list[MemoryEntry],
    max_items: int,
    max_chars_per_item: int,
) -> str:
    """
    把长期记忆列表格式化成适合注入 prompt 的文本。
    """
    # 没有检索到长期记忆时返回空字符串
    if not entries:
        return ""

    # 先做去重，再限制条数
    picked_entries = _dedupe_memories(entries)[:max_items]

    # lines 用来逐行拼接最终文本
    lines: list[str] = ["## 相关长期记忆"]

    for index, entry in enumerate(picked_entries, start=1):
        # category 作为记忆类型标签，方便模型理解这条记忆属于什么
        category = entry.category.strip() or "note"

        # content 做长度控制，避免单条长期记忆太长
        content = _shorten(entry.content, max_chars_per_item)

        # tags 用来补充额外语义线索
        tags_text = ""
        if entry.tags:
            tags_text = f" [tags: {', '.join(entry.tags)}]"

        # 主体一行
        lines.append(f"{index}. ({category}) {content}{tags_text}")

    return "\n".join(lines).strip()


def build_memory_context(
    *,
    user_input: str,
    session: SessionData,
    working_memory: WorkingMemory | None,
    memory_store: MemoryStore | None,
    session_summary_override: str = "",
    top_k: int = 5,
    max_summary_chars: int = 400,
    max_memory_chars_per_item: int = 180,
) -> str:
    """
    构建注入 prompt 的记忆上下文。

    这层不是完整历史消息本身，
    而是把“会话摘要 + 短期工作记忆 + 长期记忆检索结果”拼成一段更紧凑的辅助上下文。
    """
    # sections 用来收集各个记忆分段
    sections: list[str] = []

    # 1. 注入会话摘要
    # 优先使用外部传入的旧历史摘要。
    # 这样主循环做完“近轮保留 + 旧轮摘要”后，就不会再把 recent tail 重复概括一遍。
    session_summary = _shorten(session_summary_override, max_summary_chars)

    # 如果外部没有覆盖值，再退回到 session.meta.summary。
    if not session_summary and session.meta is not None:
        session_summary = _shorten(session.meta.summary, max_summary_chars)

    # 有摘要时才放进去，避免空标题污染 prompt
    if session_summary:
        sections.append("## 会话摘要\n" + session_summary)

    # 2. 注入短期工作记忆
    # working_memory 里一般放当前目标、最近决策、最近失败、活跃路径
    if working_memory is not None:
        try:
            working_memory_text = working_memory.format_for_prompt().strip()
        except Exception:
            working_memory_text = ""

        # 非空时再注入
        if working_memory_text:
            sections.append("## 当前工作记忆\n" + working_memory_text)

    # 3. 注入长期记忆
    # 第一版直接按 user_input 检索最相关的 top-k
    long_term_memory_text = ""
    if memory_store is not None:
        try:
            related_memories = memory_store.search_memories(
                query=user_input,
                top_k=top_k,
            )
        except Exception:
            # 长期记忆检索失败时直接降级，不让主流程炸掉
            related_memories = []

        long_term_memory_text = _format_long_term_memories(
            entries=related_memories,
            max_items=top_k,
            max_chars_per_item=max_memory_chars_per_item,
        )

    # 有内容时再加入
    if long_term_memory_text:
        sections.append(long_term_memory_text)

    # 如果三层记忆都没有，返回空字符串即可
    if not sections:
        return ""

    # 最前面补一个总说明，告诉模型这段内容是什么
    header = (
        "以下内容是从当前会话和本地记忆中整理出的辅助上下文。"
        "它用于帮助你保持任务连续性、记住约定和避免重复犯错。"
        "当这些内容与用户本轮明确要求冲突时，以用户本轮要求为准。"
    )

    # 用双换行把不同记忆块隔开，结构更清晰
    return header + "\n\n" + "\n\n".join(sections)
