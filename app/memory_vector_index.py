from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from app.memory_store import MemoryEntry


@dataclass(slots=True)
class VectorSearchHit:
    """
    一条向量召回结果。

    字段说明：
    - memory_id: 命中的长期记忆 id
    - score: Qdrant 返回的相似度分数
    - payload: 该点位附带的元数据
    """

    memory_id: str
    score: float
    payload: dict[str, Any]


class MemoryVectorIndex:
    """
    长期记忆的 Qdrant 向量索引封装。

    这一层只负责两件事：
    1. 把长期记忆写入 Qdrant，建立语义索引
    2. 根据查询文本做语义召回，返回最相近的 memory id

    设计原则：
    - 只依赖 Qdrant 服务端，不做嵌入式本地模式
    - JSON 仍然是权威数据源，Qdrant 只负责语义检索和可视化
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        embedding_model: str,
        embedding_dimensions: int,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str,
    ) -> None:
        """
        初始化向量索引客户端。

        这里使用动态导入，是为了让项目在未启用 Qdrant 时仍可正常运行；
        只有真正打开 `QDRANT_ENABLED=true` 时，才要求安装 `qdrant-client`。
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qdrant_models
        except ImportError as error:
            raise RuntimeError(
                "启用 Qdrant 前需要先安装 qdrant-client 依赖。"
            ) from error

        self._qdrant_models = qdrant_models
        self.embedding_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.embedding_model = embedding_model
        self.embedding_dimensions = max(0, int(embedding_dimensions))
        self.collection_name = collection_name
        self.qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key or None,
            timeout=10.0, # type: ignore
            # 当前本地 Docker 服务已可正常访问。
            # 这里关闭客户端的版本兼容性探测，避免启动时出现无意义警告。
            check_compatibility=False,
            # 某些环境下 httpx 会继承外部网络设置，导致对本地 Qdrant 的请求被错误处理成 502。
            # 这里显式关闭环境继承，确保 localhost / 127.0.0.1 直连本机服务。
            trust_env=False,
        )

        # collection 会在第一次真正拿到 embedding 维度后再自动创建。
        self._collection_initialized = False
        self._vector_size: int | None = None

    def upsert_memory(self, entry: MemoryEntry) -> None:
        """
        把一条长期记忆写入 Qdrant。

        写入内容分成两部分：
        - vector: 用于语义相似度检索
        - payload: 用于 dashboard 可视化和过滤查询
        """
        vector = self._embed_text(self._build_embedding_text(entry))
        self._ensure_collection(vector_size=len(vector))

        point = self._qdrant_models.PointStruct(
            id=self._build_qdrant_point_id(entry.id),
            vector=vector,
            payload=self._build_payload(entry),
        )
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True,
        )

    def search_similar_memories(
        self,
        *,
        query_text: str,
        top_k: int = 5,
        scope: str | None = None,
        category: str | None = None,
        domains: list[str] | None = None,
        include_archived: bool = False,
        exclude_ids: list[str] | None = None,
    ) -> list[VectorSearchHit]:
        """
        使用语义向量召回相似长期记忆。

        参数说明：
        - query_text: 要检索的语义查询文本
        - top_k: 返回多少条候选
        - scope/category/domains: 通过 payload 做过滤
        - include_archived: 是否包含已归档记忆
        - exclude_ids: 需要排除的 memory id
        """
        cleaned_query = " ".join(query_text.strip().split())
        if not cleaned_query:
            return []

        query_vector = self._embed_text(cleaned_query)
        self._ensure_collection(vector_size=len(query_vector))

        query_filter = self._build_filter(
            scope=scope,
            category=category,
            domains=domains,
            include_archived=include_archived,
            exclude_ids=exclude_ids or [],
        )

        search_result = self.qdrant.search( # type: ignore
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        hits: list[VectorSearchHit] = []
        for item in search_result:
            payload = dict(item.payload or {})
            memory_id = str(payload.get("memory_id", "")).strip() or str(item.id)
            hits.append(
                VectorSearchHit(
                    memory_id=memory_id,
                    score=float(item.score),
                    payload=payload,
                )
            )

        return hits

    def _build_embedding_text(self, entry: MemoryEntry) -> str:
        """
        把一条长期记忆整理成 embedding 输入文本。

        这里不只编码 content，
        还会把 category / tags / domains 一起编码进去，
        让向量更容易保留“这条记忆属于什么类型”的语义。
        """
        parts = [
            f"category: {entry.category}",
            f"scope: {entry.scope}",
            f"content: {entry.content}",
        ]
        if entry.tags:
            parts.append(f"tags: {', '.join(entry.tags)}")
        if entry.domains:
            parts.append(f"domains: {', '.join(entry.domains)}")
        return "\n".join(parts).strip()

    def _build_payload(self, entry: MemoryEntry) -> dict[str, Any]:
        """
        构造写入 Qdrant 的 payload。

        这些字段既用于 dashboard 可视化，也用于后续 metadata filter。
        """
        return {
            "memory_id": entry.id,
            "content": entry.content,
            "category": entry.category,
            "tags": list(entry.tags),
            "session_id": entry.session_id,
            "scope": entry.scope,
            "confidence": entry.confidence,
            "domains": list(entry.domains),
            "source": entry.source,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "usage_count": entry.usage_count,
            "last_accessed_at": entry.last_accessed_at,
            "decay_score": entry.decay_score,
            "archived": entry.archived,
            "extra_json": json.dumps(entry.extra, ensure_ascii=False),
        }

    def _build_qdrant_point_id(self, memory_id: str) -> str:
        """
        把本地 memory id 转成 Qdrant 可接受的 point id。

        当前项目里的长期记忆 id 是短十六进制串，
        这对本地 JSON 很友好，但 Qdrant 只接受：
        - 无符号整数
        - UUID

        所以这里用 uuid5 做一个稳定映射：
        - 同一个 memory_id 永远映射到同一个 UUID
        - 不需要改动本地 JSON 的 id 方案
        """
        normalized_memory_id = memory_id.strip()
        if not normalized_memory_id:
            normalized_memory_id = uuid.uuid4().hex

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"minicode-memory:{normalized_memory_id}",
            )
        )

    def _build_filter(
        self,
        *,
        scope: str | None,
        category: str | None,
        domains: list[str] | None,
        include_archived: bool,
        exclude_ids: list[str],
    ) -> Any | None:
        """
        按 payload 字段构造 Qdrant 过滤器。

        当前支持的过滤维度：
        - scope
        - category
        - domains
        - archived
        - exclude_ids
        """
        must_conditions: list[Any] = []
        must_not_conditions: list[Any] = []

        if scope:
            must_conditions.append(
                self._qdrant_models.FieldCondition(
                    key="scope",
                    match=self._qdrant_models.MatchValue(value=scope),
                )
            )

        if category:
            must_conditions.append(
                self._qdrant_models.FieldCondition(
                    key="category",
                    match=self._qdrant_models.MatchValue(value=category),
                )
            )

        if domains:
            must_conditions.append(
                self._qdrant_models.FieldCondition(
                    key="domains",
                    match=self._qdrant_models.MatchAny(any=list(domains)),
                )
            )

        if not include_archived:
            must_conditions.append(
                self._qdrant_models.FieldCondition(
                    key="archived",
                    match=self._qdrant_models.MatchValue(value=False),
                )
            )

        if exclude_ids:
            must_not_conditions.append(
                self._qdrant_models.HasIdCondition(has_id=list(exclude_ids))
            )

        if not must_conditions and not must_not_conditions:
            return None

        return self._qdrant_models.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
        )

    def _ensure_collection(self, *, vector_size: int) -> None:
        """
        确保目标 collection 已存在，且维度与当前 embedding 模型一致。

        这里不会再使用 `recreate_collection` 粗暴重建集合，
        因为那样一旦连错环境或误配维度，可能直接清空已有向量数据。
        当前策略是：
        - collection 不存在 -> 创建
        - collection 已存在但维度不一致 -> 直接报错，让用户显式处理
        """
        if self._collection_initialized and self._vector_size == vector_size:
            return

        collections_info = self.qdrant.get_collections()
        collection_names = {item.name for item in collections_info.collections}

        if self.collection_name not in collection_names:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._qdrant_models.VectorParams(
                    size=vector_size,
                    distance=self._qdrant_models.Distance.COSINE,
                ),
            )
        else:
            collection_info = self.qdrant.get_collection(self.collection_name)
            current_size = int(collection_info.config.params.vectors.size) # type: ignore
            if current_size != vector_size:
                raise RuntimeError(
                    f"Qdrant collection `{self.collection_name}` 的向量维度是 {current_size}，"
                    f"但当前 embedding 维度是 {vector_size}。"
                )

        self._collection_initialized = True
        self._vector_size = vector_size

    def _embed_text(self, text: str) -> list[float]:
        """
        调用 embedding 模型生成向量。

        这里统一复用 OpenAI-compatible 接口，
        与项目当前的模型接入方式保持一致。
        """
        request_kwargs: dict[str, Any] = {
            "model": self.embedding_model,
            "input": text,
            "encoding_format": "float",
        }

        # 百炼的 text-embedding-v3 / v4 支持显式指定 dimensions。
        # 如果环境变量没有配置维度，就不传这个参数，避免和其他兼容服务不兼容。
        if self.embedding_dimensions > 0:
            request_kwargs["dimensions"] = self.embedding_dimensions

        response = self.embedding_client.embeddings.create(**request_kwargs)
        return list(response.data[0].embedding)
