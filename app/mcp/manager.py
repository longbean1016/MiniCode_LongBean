# app/mcp/manager.py
"""MCP Server 生命周期管理器。

统一管理所有 MCP 客户端（StdioMcpClient / HttpMcpClient），提供:
    - bootstrap():   启动时批量初始化已配置的 Server
    - add_server():  运行时添加并启动 Server（/mcp add，支持 stdio 和 HTTP）
    - remove_server(): 运行时移除并停止 Server（/mcp remove）
    - list_servers(): 列出所有 Server 及连接状态（/mcp list）
    - dispose():     退出时关闭所有连接
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable

from app.agent.tooling import ToolRegistry
from app.mcp.client import StdioMcpClient, create_mcp_backed_tools
from app.mcp.http_client import HttpMcpClient, create_http_mcp_backed_tools
from app.mcp.config import (
    add_server_to_config,
    load_mcp_config,
    remove_server_from_config,
)


class McpManager:
    """MCP Server 生命周期管理器。

    职责:
        - 持有所有活跃的 StdioMcpClient
        - 处理 /mcp add / remove / list 命令
        - 退出时清理所有子进程
    """

    def __init__(self, cwd: str, tool_registry: ToolRegistry) -> None:
        """初始化 MCP 管理器。

        Args:
            cwd: 项目根目录（用于读写 .mcp.json）
            tool_registry: 工具注册表（用于动态注册/注销 MCP 工具）
        """
        self._cwd = cwd
        self._tool_registry = tool_registry
        # 活跃的 MCP 客户端: {server_name: StdioMcpClient}
        self._clients: dict[str, StdioMcpClient] = {}
        # 异步操作结果回调（由 TUI 注入，用于 call_from_thread 安全更新界面）
        self._result_callback: Callable[[str], None] | None = None
        # 正在添加中的 Server 名称集合（防止重复添加）
        self._pending_adds: set[str] = set()

    def set_result_callback(self, callback: Callable[[str], None]) -> None:
        """设置异步操作结果回调（由 TUI 层注入）。

        回调会在后台线程中调用，TUI 层需自行通过 call_from_thread 安全更新界面。
        """
        self._result_callback = callback

    # ---- 启动时批量初始化 ----

    def bootstrap(self, mcp_config: dict[str, dict]) -> list[str]:
        """启动时批量初始化 .mcp.json 中已配置的 MCP Server。

        单个 Server 启动失败不阻塞其他 Server。

        Args:
            mcp_config: load_mcp_config() 返回的配置字典

        Returns:
            启动失败的 Server 名称列表（用于日志提示）
        """
        failed: list[str] = []

        for server_name, config in mcp_config.items():
            try:
                self._start_server(server_name, config)
            except Exception as error:
                failed.append(f"{server_name}: {error}")

        return failed

    # ---- 运行时 /mcp 命令 ----

    def handle_command(self, user_input: str) -> str:
        """解析并执行 /mcp 命令。

        支持子命令:
            /mcp list                              → 显示所有 Server 状态
            /mcp add <name> -- <command> [args...] → 添加 stdio Server
            /mcp add <name> --url <url>            → 添加 HTTP Server
            /mcp remove <name>                     → 移除并停止 Server

        Returns:
            显示给用户的结果文本。
        """
        text = user_input.strip()
        parts = text.split()

        if len(parts) < 2:
            return self._help_text()

        subcommand = parts[1].lower()

        if subcommand == "list":
            return self.list_servers()

        if subcommand == "remove" and len(parts) >= 3:
            name = parts[2]
            return self.remove_server(name)

        if subcommand == "add" and len(parts) >= 3:
            name = parts[2]

            # HTTP 模式: /mcp add <name> --url <url>
            if "--url" in parts:
                try:
                    url_index = parts.index("--url", 3)
                except ValueError:
                    return "格式错误。HTTP 模式正确格式: /mcp add <name> --url <url>"
                if url_index + 1 >= len(parts):
                    return "格式错误：缺少 URL。正确格式: /mcp add <name> --url <url>"
                url = parts[url_index + 1]
                return self.add_server(name, url=url)

            # Stdio 模式: /mcp add <name> -- <command> [args...]
            try:
                dash_index = parts.index("--", 3)
            except ValueError:
                return (
                    "格式错误。Stdio 模式: /mcp add <name> -- <command> [args...]\n"
                    "HTTP 模式: /mcp add <name> --url <url>"
                )

            if dash_index + 1 >= len(parts):
                return (
                    "格式错误：缺少 command。\n"
                    "Stdio 模式: /mcp add <name> -- <command> [args...]\n"
                    "HTTP 模式: /mcp add <name> --url <url>"
                )

            command = parts[dash_index + 1]
            args = parts[dash_index + 2:] if dash_index + 2 < len(parts) else []
            return self.add_server(name, command=command, args=args)

        return self._help_text()

    def add_server(
        self,
        name: str,
        command: str = "",
        args: list[str] | None = None,
        *,
        url: str = "",
    ) -> str:
        """异步添加并启动一个 MCP Server（不阻塞 TUI）。

        支持两种传输方式:
            - stdio: 传入 command + args，通过子进程通信
            - HTTP:  传入 url，通过 HTTP POST 通信

        流程:
            1. 验证名称合法性（重复 / 正在添加中）
            2. 返回 "⏳ 正在添加..." 后立即恢复 TUI 交互
            3. 后台线程完成: 持久化 → 连接 → 发现工具 → 注册
            4. 通过回调通知 TUI 最终结果
        """
        # 检查重复
        if name in self._clients:
            return f"❌ MCP Server '{name}' 已存在，请先 /mcp remove {name}"

        # 检查是否正在添加中（防止并发重复添加）
        if name in self._pending_adds:
            return f"⏳ MCP Server '{name}' 正在添加中，请稍候..."

        self._pending_adds.add(name)

        # 后台线程执行实际的添加逻辑
        threading.Thread(
            target=self._add_server_sync,
            args=(name, command, args or [], url),
            daemon=True,
        ).start()

        return f"⏳ 正在后台添加 MCP Server: {name}..."

    def _add_server_sync(
        self, name: str, command: str, args: list[str], url: str
    ) -> None:
        """后台线程: 同步执行 MCP Server 添加（持久化 + 连接 + 工具发现 + 注册）。"""
        result: str
        is_http = bool(url)
        try:
            if is_http:
                # HTTP 模式: 持久化 URL 配置 → 连接 → 发现工具
                server_config = {"url": url}
                add_server_to_config(self._cwd, name, server_config)

                client = HttpMcpClient(name, server_config, self._cwd)
                try:
                    client.start()
                except Exception as error:
                    try:
                        remove_server_from_config(self._cwd, name)
                    except Exception:
                        pass
                    result = f"❌ 连接 MCP Server '{name}' 失败: {error}"
                    if self._result_callback is not None:
                        self._result_callback(result)
                    return

                try:
                    mcp_tools = create_http_mcp_backed_tools(
                        server_name=name, client=client
                    )
                except Exception as error:
                    client.close()
                    try:
                        remove_server_from_config(self._cwd, name)
                    except Exception:
                        pass
                    result = f"❌ 发现 MCP Server '{name}' 的工具失败: {error}"
                    if self._result_callback is not None:
                        self._result_callback(result)
                    return

            else:
                # Stdio 模式: 原有逻辑
                server_config = {"command": command, "args": args}
                add_server_to_config(self._cwd, name, server_config)

                client = StdioMcpClient(name, server_config, self._cwd)
                try:
                    client.start()
                except Exception as error:
                    try:
                        remove_server_from_config(self._cwd, name)
                    except Exception:
                        pass
                    result = f"❌ 启动 MCP Server '{name}' 失败: {error}"
                    if self._result_callback is not None:
                        self._result_callback(result)
                    return

                try:
                    mcp_tools = create_mcp_backed_tools(
                        server_name=name, client=client
                    )
                except Exception as error:
                    client.close()
                    try:
                        remove_server_from_config(self._cwd, name)
                    except Exception:
                        pass
                    result = f"❌ 发现 MCP Server '{name}' 的工具失败: {error}"
                    if self._result_callback is not None:
                        self._result_callback(result)
                    return

            for tool in mcp_tools:
                self._tool_registry.register_tool(tool)

            self._clients[name] = client

            result = f"✅ 已添加 MCP Server: {name}（{len(mcp_tools)} 个工具）"
        finally:
            self._pending_adds.discard(name)

        # 通过回调通知 TUI 最终结果
        if self._result_callback is not None:
            self._result_callback(result)

    def remove_server(self, name: str) -> str:
        """移除并停止一个 MCP Server。

        流程:
            1. 检查是否存在
            2. 从 ToolRegistry 注销该 Server 的所有工具
            3. 关闭 StdioMcpClient（停止子进程）
            4. 从 _clients 和 .mcp.json 删除
        """
        client = self._clients.get(name)
        if client is None:
            return f"❌ 未找到 MCP Server: {name}"

        # 从 ToolRegistry 注销该 Server 的所有工具
        prefix = f"mcp__{self._sanitize_name(name)}__"
        tools_to_remove = [
            tool_name
            for tool_name in self._tool_registry.list_tool_name()
            if tool_name.startswith(prefix)
        ]
        for tool_name in tools_to_remove:
            self._tool_registry.unregister_tool(tool_name)

        # 停止子进程
        try:
            client.close()
        except Exception:
            pass

        del self._clients[name]

        # 从配置文件移除
        try:
            remove_server_from_config(self._cwd, name)
        except Exception as error:
            return f"⚠️ Server 已停止，但清理 .mcp.json 失败: {error}"

        return f"✅ 已移除 MCP Server: {name}（{len(tools_to_remove)} 个工具已注销）"

    def list_servers(self) -> str:
        """列出所有已配置的 MCP Server 及连接状态。

        读取 .mcp.json 获取全部配置（包括未启动的），
        交叉比对活跃 _clients 标注连接状态。
        """
        # 读取原始文件获取完整配置（包括 disabled 的 server）
        path = Path(self._cwd) / ".mcp.json"
        all_config: dict[str, dict] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                all_config = data.get("mcpServers", {}) if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError):
                all_config = {}

        if not all_config:
            return "没有配置任何 MCP Server。使用 /mcp add <name> -- <command> [args...] 添加。"

        lines = ["MCP Servers:"]
        for srv_name, srv_config in all_config.items():
            command = srv_config.get("command", "?")
            enabled = srv_config.get("enabled", True)

            if srv_name in self._clients:
                # 活跃的 server
                tool_count = len(
                    [
                        t
                        for t in self._tool_registry.list_tool_name()
                        if t.startswith(f"mcp__{self._sanitize_name(srv_name)}__")
                    ]
                )
                status = f"connected ({tool_count} tools)"
            elif not enabled:
                status = "disabled"
            else:
                status = "disconnected"

            # HTTP Server 显示 url，Stdio Server 显示 command
            label = srv_config.get("url") or command
            lines.append(f"  {srv_name}: {label} ({status})")

        return "\n".join(lines)

    def dispose(self) -> None:
        """退出时关闭所有 MCP 子进程。"""
        for client in list(self._clients.values()):
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()

    # ---- 内部工具方法 ----

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """将 Server 名中的特殊字符替换为下划线（与 client.py 中一致）。"""
        normalized = "".join(
            char.lower() if char.isalnum() or char in {"_", "-"} else "_"
            for char in name
        )
        return normalized.strip("_") or "server"

    def _start_server(self, server_name: str, config: dict) -> None:
        """内部方法: 启动单个 MCP Server（bootstrap 用）。

        根据 config 中是否有 url 自动选择 HTTP 或 stdio 客户端。

        Raises:
            RuntimeError: 启动失败时抛出
        """
        if "url" in config:
            # HTTP 模式
            client = HttpMcpClient(server_name, config, self._cwd)
            client.start()
            mcp_tools = create_http_mcp_backed_tools(
                server_name=server_name, client=client
            )
        else:
            # Stdio 模式
            client = StdioMcpClient(server_name, config, self._cwd)
            client.start()
            mcp_tools = create_mcp_backed_tools(server_name=server_name, client=client)

        for tool in mcp_tools:
            self._tool_registry.register_tool(tool)
        self._clients[server_name] = client

    @staticmethod
    def _help_text() -> str:
        """返回 /mcp 命令帮助文本。"""
        return (
            "/mcp 命令:\n"
            "  /mcp list                              查看已配置的 MCP Server\n"
            "  /mcp add <name> -- <command> [args...]  添加 stdio MCP Server\n"
            "  /mcp add <name> --url <url>             添加 HTTP MCP Server\n"
            "  /mcp remove <name>                      移除并停止 MCP Server"
        )
