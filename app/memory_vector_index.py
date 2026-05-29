from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

from app.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.logger import log_event
from app.memory_store import MemoryEntry
from app.retry import RetryPolicy, run_with_retry, should_retry_vector_error


@dataclass(slots=True)
class VectorSearchHit:
    """
    一条向量召回结果。
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
    - JSON 仍然是权威数据源
    - Qdrant 只负责语义检索、语义去重和可视化
    - embedding / Qdrant 属于非主链路能力，失败时应该允许降级
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
        retry_max_attempts: int = 3,
        retry_base_delay_seconds: float = 0.8,
        retry_backoff_multiplier: float = 2.0,
        retry_max_delay_seconds: float = 4.0,
        circuit_failure_threshold: int = 3,
        circuit_recovery_timeout_seconds: float = 45.0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qdrant_models
        except ImportError as error:
            raise RuntimeError("启用 Qdrant 前需要先安装 qdrant-client 依赖。") from error

        self._qdrant_models = qdrant_models
        self.embedding_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.embedding_model = embedding_model
        self.embedding_dimensions = max(0, int(embedding_dimensions))
        self.collection_name = collection_name

        # embedding 和 Qdrant 都不是主回答链路，
        # 所以这里允许轻量重试和短时间熔断。
        self.embedding_retry_policy = RetryPolicy(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay_seconds,
            backoff_multiplier=retry_backoff_multiplier,
            max_delay_seconds=retry_max_delay_seconds,
        )
        self.qdrant_retry_policy = RetryPolicy(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay_seconds,
            backoff_multiplier=retry_backoff_multiplier,
            max_delay_seconds=retry_max_delay_seconds,
        )
        self.embedding_breaker = CircuitBreaker(
            name="embedding_service",
            failure_threshold=circuit_failure_threshold,
            recovery_timeout_seconds=circuit_recovery_timeout_seconds,
        )
        self.qdrant_breaker = CircuitBreaker(
            name="qdrant_service",
            failure_threshold=circuit_failure_threshold,
            recovery_timeout_seconds=circuit_recovery_timeout_seconds,
        )

        self.qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key or None,
            timeout=10.0,  # type: ignore
            check_compatibility=False,
            trust_env=False,
        )

        # collection 在第一次拿到 embedding 维度后再自动创建。
        self._collection_initialized = False
        self._vector_size: int | None = None

    def upsert_memory(self, entry: MemoryEntry) -> None:
        """
        把一条长期记忆写入 Qdrant。
        """

        vector = self._embed_text(self._build_embedding_text(entry))
        self._ensure_collection(vector_size=len(vector))

        point = self._qdrant_models.PointStruct(
            id=self._build_qdrant_point_id(entry.id),
            vector=vector,
            payload=self._build_payload(entry),
        )
        self._call_qdrant(
            operation_name="upsert_memory",
            operation=lambda: self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True,
            ),
        )

    def delete_memories(self, memory_ids: list[str]) -> None:
        """
        按 memory id 批量删除 Qdrant 中的点位。
        """

        point_ids = [
            self._build_qdrant_point_id(memory_id)
            for memory_id in memory_ids
            if memory_id.strip()
        ]
        if not point_ids:
            return

        self._call_qdrant(
            operation_name="delete_memories",
            operation=lambda: self.qdrant.delete(  # type: ignore
                collection_name=self.collection_name,
                points_selector=self._qdrant_models.PointIdsList(
                    points=point_ids,  # type: ignore
                ),
                wait=True,
            ),
        )

    def list_memory_ids(self) -> set[str]:
        """
        列出当前 collection 里的全部 payload.memory_id。
        """

        collections_info = self._call_qdrant(
            operation_name="list_memory_ids.get_collections",
            operation=lambda: self.qdrant.get_collections(),
        )
        collection_names = {item.name for item in collections_info.collections}
        if self.collection_name not in collection_names:
            return set()

        result: set[str] = set()
        offset: Any | None = None

        while True:
            points, next_offset = self._call_qdrant(
                operation_name="list_memory_ids.scroll",
                operation=lambda: self.qdrant.scroll(  # type: ignore
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            for point in points:
                payload = dict(point.payload or {})
                memory_id = str(payload.get("memory_id", "")).strip()
                if memory_id:
                    result.add(memory_id)

            if next_offset is None:
                break
            offset = next_offset

        return result

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

        search_result = self._call_qdrant(
            operation_name="search_similar_memories",
            operation=lambda: self.qdrant.search(  # type: ignore
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            ),
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
        把本地 memory id 转成 Qdrant 可接受的稳定 UUID。
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
                self._qdrant_models.HasIdCondition(
                    has_id=[self._build_qdrant_point_id(item) for item in exclude_ids],
                )
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
        """

        if self._collection_initialized and self._vector_size == vector_size:
            return

        collections_info = self._call_qdrant(
            operation_name="_ensure_collection.get_collections",
            operation=lambda: self.qdrant.get_collections(),
        )
        collection_names = {item.name for item in collections_info.collections}

        if self.collection_name not in collection_names:
            self._call_qdrant(
                operation_name="_ensure_collection.create_collection",
                operation=lambda: self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self._qdrant_models.VectorParams(
                        size=vector_size,
                        distance=self._qdrant_models.Distance.COSINE,
                    ),
                ),
            )
        else:
            collection_info = self._call_qdrant(
                operation_name="_ensure_collection.get_collection",
                operation=lambda: self.qdrant.get_collection(self.collection_name),
            )
            current_size = int(collection_info.config.params.vectors.size)  # type: ignore
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
        """

        request_kwargs: dict[str, Any] = {
            "model": self.embedding_model,
            "input": text,
            "encoding_format": "float",
        }
        if self.embedding_dimensions > 0:
            request_kwargs["dimensions"] = self.embedding_dimensions

        response = self._call_embedding(
            operation_name="_embed_text",
            operation=lambda: self.embedding_client.embeddings.create(**request_kwargs),
        )
        return list(response.data[0].embedding)

    def _call_embedding(
        self,
        *,
        operation_name: str,
        operation: Callable[[], Any],
    ) -> Any:
        """
        统一执行 embedding 请求。

        如果 embedding 服务连续失败，会短时间熔断；
        上层可以捕获异常并继续主流程。
        """

        if not self.embedding_breaker.allow_request():
            raise CircuitOpenError(self.embedding_breaker.reject_reason())

        try:
            result = run_with_retry(
                operation,
                policy=self.embedding_retry_policy,
                should_retry=should_retry_vector_error,
                on_retry=lambda attempt, error, delay: log_event(
                    (
                        f"embedding 调用失败，准备第 {attempt + 1} 次尝试："
                        f"{operation_name} {type(error).__name__}: {error}，"
                        f"等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.embedding_breaker.record_failure(error)
            raise

        self.embedding_breaker.record_success()
        return result

    def _call_qdrant(
        self,
        *,
        operation_name: str,
        operation: Callable[[], Any],
    ) -> Any:
        """
        统一执行 Qdrant 请求。
        """

        if not self.qdrant_breaker.allow_request():
            raise CircuitOpenError(self.qdrant_breaker.reject_reason())

        try:
            result = run_with_retry(
                operation,
                policy=self.qdrant_retry_policy,
                should_retry=should_retry_vector_error,
                on_retry=lambda attempt, error, delay: log_event(
                    (
                        f"Qdrant 调用失败，准备第 {attempt + 1} 次尝试："
                        f"{operation_name} {type(error).__name__}: {error}，"
                        f"等待 {delay:.1f}s"
                    ),
                    echo=False,
                ),
            )
        except Exception as error:
            self.qdrant_breaker.record_failure(error)
            raise

        self.qdrant_breaker.record_success()
        return result
