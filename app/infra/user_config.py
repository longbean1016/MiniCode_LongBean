"""用户级配置模块：读写 ~/.bean/settings.json。

   对标 Claude Code ~/.claude/settings.json 的配置体系。
   所有配置集中在用户目录下的 .bean 文件夹中，包括：
   - api_key / base_url / model：API 连接和模型选择
   - models：可选模型列表（用户可在 settings.json 中手动添加）
   - permissions：权限规则（allow/deny/ask）
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── 用户配置目录 ──
def _get_user_bean_dir() -> Path:
    """获取用户级 .bean 配置目录路径（~/.bean）。"""
    home = Path.home()
    return home / ".bean"


def _get_settings_path() -> Path:
    """获取用户级 settings.json 的完整路径。"""
    return _get_user_bean_dir() / "settings.json"


# ── 默认配置 ──
_DEFAULT_SETTINGS: dict[str, Any] = {
    "api_key": "",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "models": [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "gpt-4o-mini",
        "gpt-4o",
    ],
}


@dataclass(slots=True)
class UserConfig:
    """用户配置数据对象。"""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    models: list[str] = field(default_factory=lambda: ["deepseek-v4-flash", "deepseek-v4-pro"])
    raw: dict[str, Any] = field(default_factory=dict)  # 原始 JSON，权限等扩展字段从这里读


def ensure_user_config() -> UserConfig:
    """确保用户配置存在：读取已有配置，或创建默认配置。

       对标 Claude Code 首次启动时自动创建 ~/.claude/settings.json。
       如果 settings.json 不存在或缺少 api_key，写入默认配置（api_key 为空），
       让上层决定是否进入配置向导。

       Returns:
           UserConfig: 用户配置对象
    """
    settings_path = _get_settings_path()
    bean_dir = _get_user_bean_dir()

    # 确保 .bean 目录存在
    bean_dir.mkdir(parents=True, exist_ok=True)

    # 尝试读取已有配置
    existing: dict[str, Any] = {}
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    # 合并默认值：已有配置优先，缺失字段用默认值补
    merged = dict(_DEFAULT_SETTINGS)
    merged.update(existing)

    # 如果文件不存在或内容有变化，回写
    if not settings_path.exists() or existing != merged:
        _write_settings(merged)

    return UserConfig(
        api_key=str(merged.get("api_key", "")).strip(),
        base_url=str(merged.get("base_url", _DEFAULT_SETTINGS["base_url"])).strip(),
        model=str(merged.get("model", _DEFAULT_SETTINGS["model"])).strip(),
        models=_normalize_models(merged.get("models", [])),
        raw=merged,
    )


def load_user_config() -> UserConfig:
    """加载用户配置（不触发自动创建）。"""
    settings_path = _get_settings_path()
    if not settings_path.exists():
        return UserConfig()

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return UserConfig()

    return UserConfig(
        api_key=str(data.get("api_key", "")).strip(),
        base_url=str(data.get("base_url", _DEFAULT_SETTINGS["base_url"])).strip(),
        model=str(data.get("model", _DEFAULT_SETTINGS["model"])).strip(),
        models=_normalize_models(data.get("models", [])),
        raw=data,
    )


def save_user_config(config: UserConfig) -> None:
    """保存用户配置到 ~/.bean/settings.json。

       Args:
           config: 要保存的 UserConfig 对象
    """
    data = dict(config.raw)
    data["api_key"] = config.api_key
    data["base_url"] = config.base_url
    data["model"] = config.model
    data["models"] = config.models
    _write_settings(data)


def update_user_model(model_id: str) -> None:
    """切换当前使用的模型并持久化。

       Args:
           model_id: 新的模型 ID（如 "deepseek-v4-pro"）
    """
    config = load_user_config()
    config.model = model_id
    # 如果模型不在已有列表中，自动追加
    if model_id not in config.models:
        config.models.append(model_id)
    save_user_config(config)


def get_available_models(config: UserConfig) -> list[str]:
    """获取可用模型列表。用户自定义的 models 字段优先，否则用代码默认值。

       对标 Claude Code：/model 命令展示的模型列表来源。
    """
    if config.models:
        return config.models
    return _DEFAULT_SETTINGS.get("models", [])


def has_api_key() -> bool:
    """检查是否已配置 API Key。"""
    config = load_user_config()
    return bool(config.api_key)


def get_permissions_data() -> dict[str, Any]:
    """从 settings.json 中读取 permissions 节点（供权限系统使用）。
       保持向后兼容：如果 project/.bean/settings.json 存在，优先读取。
    """
    config = load_user_config()
    return config.raw.get("permissions", {})


def save_permissions_data(data: dict[str, Any]) -> None:
    """将权限规则保存到 settings.json 的 permissions 节点。"""
    config = load_user_config()
    config.raw["permissions"] = data
    save_user_config(config)


# ── 内部辅助函数 ──

def _write_settings(data: dict[str, Any]) -> None:
    """写入 settings.json 文件。"""
    bean_dir = _get_user_bean_dir()
    bean_dir.mkdir(parents=True, exist_ok=True)
    settings_path = _get_settings_path()
    temp_path = settings_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    temp_path.replace(settings_path)


def _normalize_models(models: object) -> list[str]:
    """规范化模型列表：支持字符串列表格式。

       ["deepseek-v4-flash", "gpt-4o-mini"]
    """
    if not isinstance(models, list):
        return _DEFAULT_SETTINGS.get("models", [])
    result: list[str] = []
    for m in models:
        if isinstance(m, str) and m.strip():
            result.append(m.strip())
    return result or _DEFAULT_SETTINGS.get("models", [])
