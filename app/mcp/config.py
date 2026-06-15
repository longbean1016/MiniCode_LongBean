"""MCP 配置文件读写。

配置文件路径: {项目根}/.mcp.json
格式: {"mcpServers": {"server-name": {"command": "...", "args": [...], "enabled": true}}}
"""

from __future__ import annotations

import json
from pathlib import Path

# 项目根目录下的 MCP 配置文件名
MCP_CONFIG_FILE = ".mcp.json"


def load_mcp_config(cwd: str) -> dict[str, dict]:
    """读取 .mcp.json，仅返回 enabled 的 Server 配置。

    文件不存在时返回空字典，不影响项目正常启动。
    """
    path = Path(cwd) / MCP_CONFIG_FILE
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # 配置文件损坏时返回空配置，不阻塞启动
        return {}

    servers = data.get("mcpServers", {})
    if not isinstance(servers, dict):
        return {}

    # 只返回 enabled=True 的 server（未显式设置 enabled 的默认启用）
    return {
        name: cfg
        for name, cfg in servers.items()
        if isinstance(cfg, dict) and cfg.get("enabled", True)
    }


def save_mcp_config(cwd: str, servers: dict[str, dict]) -> None:
    """全量写入 .mcp.json。"""
    path = Path(cwd) / MCP_CONFIG_FILE
    path.write_text(
        json.dumps({"mcpServers": servers}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def add_server_to_config(cwd: str, name: str, command: str, args: list[str]) -> None:
    """向 .mcp.json 追加一个 MCP Server（保留已有配置）。"""
    # 直接读原始文件获取全部 Server 配置（包括 enabled=False 的）
    path = Path(cwd) / MCP_CONFIG_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            all_servers = data.get("mcpServers", {}) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            all_servers = {}
    else:
        all_servers = {}

    all_servers[name] = {
        "command": command,
        "args": args,
        "enabled": True,
    }
    save_mcp_config(cwd, all_servers)


def remove_server_from_config(cwd: str, name: str) -> bool:
    """从 .mcp.json 移除一个 MCP Server。

    返回 True 表示成功删除，False 表示文件不存在或 Server 不存在。
    """
    path = Path(cwd) / MCP_CONFIG_FILE
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return False

    if name in servers:
        del servers[name]
        save_mcp_config(cwd, servers)
        return True
    return False
