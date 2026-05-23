from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.types import AppConfig


# 加载当前项目目录下的 .env 文件
# 如果 .env 不存在也不会报错，只是环境变量需要从系统环境里取
load_dotenv(override=True)


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """
    统一读取环境变量。

    参数说明：
    - name: 环境变量名
    - default: 默认值，如果环境变量不存在则返回它
    - required: 是否为必填项；如果为 True 且变量为空，则抛出异常

    返回值：
    - 返回读取到的字符串值
    """
    value = os.getenv(name, default)

    # 如果这个配置是必填的，并且没有值，就直接报错
    if required and (value is None or value.strip() == ""):
        raise ValueError(f"Missing required environment variable: {name}")

    # 如果 value 仍然是 None，这里转成空字符串，避免类型不稳定
    return value or ""


def load_config() -> AppConfig:
    """
    加载项目运行配置，并组装成 AppConfig 返回。

    当前第一版只关心 4 个核心配置：
    - OPENAI_API_KEY: 调用模型服务所需的密钥（必填）
    - OPENAI_BASE_URL: 模型服务地址（可默认）
    - OPENAI_MODEL: 使用的模型名（可默认）
    - WORKSPACE_ROOT: agent 允许工作的根目录（可默认）
    """
    api_key = _get_env("OPENAI_API_KEY", required=True)

    # 默认走 OpenAI 官方接口
    # 如果后面切换到 DeepSeek，只需要在 .env 里改成 https://api.deepseek.com
    base_url = _get_env("OPENAI_BASE_URL", default="https://api.openai.com/v1")

    # 默认模型名，第一版先给一个简单默认值
    model = _get_env("OPENAI_MODEL", default="gpt-4o-mini")

    # 默认工作目录为当前项目目录
    workspace_root = _get_env("WORKSPACE_ROOT", default=".")

    # 把工作目录转成绝对路径，避免后面工具执行时出现相对路径混乱
    workspace_root = str(Path(workspace_root).resolve())

    return AppConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        workspace_root=workspace_root,
    )