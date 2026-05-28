from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.types import AppConfig


# 加载当前项目目录下的 .env 文件。
# 如果 .env 不存在也不会报错，只是环境变量需要从系统环境里取。
load_dotenv(override=True)


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """
    统一读取环境变量。

    参数说明：
    - `name`: 环境变量名
    - `default`: 默认值；变量不存在时回退到这里
    - `required`: 是否为必填项
    """
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        raise ValueError(f"Missing required environment variable: {name}")
    return value or ""


def _get_bool_env(name: str, default: bool = False) -> bool:
    """
    读取布尔环境变量。

    兼容常见写法：
    - true / false
    - 1 / 0
    - yes / no
    - on / off
    """
    raw_value = _get_env(name, default="true" if default else "false").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int = 0) -> int:
    """
    读取整数环境变量。

    如果配置值不是合法整数，则退回默认值，避免因为配置格式错误直接中断启动。
    """
    raw_value = _get_env(name, default=str(default)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float = 0.0) -> float:
    """
    读取浮点环境变量。

    如果配置值不是合法浮点数，则回退到默认值，
    避免因为配置格式错误直接中断启动。
    """
    raw_value = _get_env(name, default=str(default)).strip()
    try:
        return float(raw_value)
    except ValueError:
        return default


def load_config() -> AppConfig:
    """
    加载项目运行配置，并组装成 `AppConfig` 返回。

    当前除了模型基础配置外，还补充了长期记忆向量索引相关配置：
    - `EMBEDDING_MODEL`: 生成语义向量时使用的 embedding 模型
    - `EMBEDDING_API_KEY`: embedding 服务专用 API Key
    - `EMBEDDING_BASE_URL`: embedding 服务专用接口地址
    - `EMBEDDING_DIMENSIONS`: 可选的向量维度
    - `QDRANT_ENABLED`: 是否启用 Qdrant 服务端向量索引
    - `QDRANT_URL`: Qdrant 服务地址
    - `QDRANT_API_KEY`: Qdrant API Key（本地无鉴权时可留空）
    - `QDRANT_COLLECTION`: 向量集合名
    """
    api_key = _get_env("OPENAI_API_KEY", required=True)
    base_url = _get_env("OPENAI_BASE_URL", default="https://api.openai.com/v1")
    model = _get_env("OPENAI_MODEL", default="gpt-4o-mini")
    embedding_model = _get_env("EMBEDDING_MODEL", default="text-embedding-3-small")
    embedding_api_key = _get_env("EMBEDDING_API_KEY", default=api_key)
    embedding_base_url = _get_env("EMBEDDING_BASE_URL", default=base_url)
    embedding_dimensions = _get_int_env("EMBEDDING_DIMENSIONS", default=0)

    workspace_root = _get_env("WORKSPACE_ROOT", default=".")
    workspace_root = str(Path(workspace_root).resolve())

    qdrant_enabled = _get_bool_env("QDRANT_ENABLED", default=False)
    qdrant_url = _get_env("QDRANT_URL", default="http://localhost:6333")
    qdrant_api_key = _get_env("QDRANT_API_KEY", default="")
    qdrant_collection = _get_env("QDRANT_COLLECTION", default="project_memories")
    decay_full_scan_trigger_count = _get_int_env("DECAY_FULL_SCAN_TRIGGER_COUNT", default=40)
    decay_min_score = _get_float_env("DECAY_MIN_SCORE", default=0.05)
    decay_archive_threshold = _get_float_env("DECAY_ARCHIVE_THRESHOLD", default=0.12)
    decay_archive_age_days = _get_float_env("DECAY_ARCHIVE_AGE_DAYS", default=45.0)
    decay_archive_confidence_threshold = _get_float_env(
        "DECAY_ARCHIVE_CONFIDENCE_THRESHOLD",
        default=0.72,
    )
    decay_archive_usage_threshold = _get_int_env("DECAY_ARCHIVE_USAGE_THRESHOLD", default=1)
    decay_log_enabled = _get_bool_env("DECAY_LOG_ENABLED", default=True)
    decay_log_echo = _get_bool_env("DECAY_LOG_ECHO", default=False)

    return AppConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_dimensions=embedding_dimensions,
        workspace_root=workspace_root,
        qdrant_enabled=qdrant_enabled,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=qdrant_collection,
        decay_full_scan_trigger_count=decay_full_scan_trigger_count,
        decay_min_score=decay_min_score,
        decay_archive_threshold=decay_archive_threshold,
        decay_archive_age_days=decay_archive_age_days,
        decay_archive_confidence_threshold=decay_archive_confidence_threshold,
        decay_archive_usage_threshold=decay_archive_usage_threshold,
        decay_log_enabled=decay_log_enabled,
        decay_log_echo=decay_log_echo,
    )
