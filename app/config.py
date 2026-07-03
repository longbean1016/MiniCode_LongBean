from __future__ import annotations

import os
from pathlib import Path

"""应用配置加载模块，从 ~/.bean/settings.json 读取配置并构造统一配置对象。"""

from app.types import AppConfig


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """
    统一读取环境变量。

    参数说明：
    - `name`: 环境变量名
    - `default`: 默认值；变量不存在时回退到这里
    - `required`: 是否为必填项
    """
    # 所有配置读取统一走这里，目的是把“默认值、必填校验、空字符串处理”收口到一个地方，
    # 避免每个配置字段各自写一套 if/else，后续难以维护。
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
    - `QDRANT_ENABLED`: 是否启用 Qdrant 向量索引
    - `QDRANT_URL`: Qdrant 服务地址；为空时可退回本地 embedded 模式
    - `QDRANT_PATH`: Qdrant 本地持久化目录；非空时优先使用
    - `QDRANT_API_KEY`: Qdrant API Key（本地无鉴权时可留空）
    - `QDRANT_COLLECTION`: 向量集合名
    """
    # ── 从 ~/.bean/settings.json 读取 API 和模型配置 ──
    # 对标 Claude Code ~/.claude/settings.json 的配置体系
    from app.infra.user_config import load_user_config
    user_cfg = load_user_config()

    if not user_cfg.api_key:
        print("未配置 API Key，请运行: python -m app.main --setup")
        import sys; sys.exit(1)

    api_key = user_cfg.api_key
    base_url = user_cfg.base_url
    model = user_cfg.model

    # ── 工作目录配置 ──
    workspace_root = _get_env("WORKSPACE_ROOT", default=".")
    workspace_root = str(Path(workspace_root).resolve())

    workspace_additional_dirs_raw = _get_env("WORKSPACE_ADDITIONAL_DIRS", default="")
    workspace_additional_dirs: list[str] = []
    if workspace_additional_dirs_raw.strip():
        workspace_additional_dirs = [
            str(Path(p.strip()).resolve())
            for p in workspace_additional_dirs_raw.split(";")
            if p.strip()
        ]

    # ── 内部默认值（不再暴露给用户配置）──
    model_retry_max_attempts = 3
    model_retry_base_delay_seconds = 0.8
    model_retry_backoff_multiplier = 2.0
    model_retry_max_delay_seconds = 4.0
    model_circuit_failure_threshold = 3
    model_circuit_recovery_timeout_seconds = 30.0
    aux_model_retry_max_attempts = 3
    aux_model_retry_base_delay_seconds = 0.8
    aux_model_retry_backoff_multiplier = 2.0
    aux_model_retry_max_delay_seconds = 4.0
    aux_model_circuit_failure_threshold = 3
    aux_model_circuit_recovery_timeout_seconds = 45.0

    return AppConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        workspace_root=workspace_root,
        workspace_additional_dirs=workspace_additional_dirs,
        model_retry_max_attempts=model_retry_max_attempts,
        model_retry_base_delay_seconds=model_retry_base_delay_seconds,
        model_retry_backoff_multiplier=model_retry_backoff_multiplier,
        model_retry_max_delay_seconds=model_retry_max_delay_seconds,
        model_circuit_failure_threshold=model_circuit_failure_threshold,
        model_circuit_recovery_timeout_seconds=model_circuit_recovery_timeout_seconds,
        aux_model_retry_max_attempts=aux_model_retry_max_attempts,
        aux_model_retry_base_delay_seconds=aux_model_retry_base_delay_seconds,
        aux_model_retry_backoff_multiplier=aux_model_retry_backoff_multiplier,
        aux_model_retry_max_delay_seconds=aux_model_retry_max_delay_seconds,
        aux_model_circuit_failure_threshold=aux_model_circuit_failure_threshold,
        aux_model_circuit_recovery_timeout_seconds=aux_model_circuit_recovery_timeout_seconds,
    )
