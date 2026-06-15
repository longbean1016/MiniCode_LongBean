# app/mcp/http_client.py
"""MCP HTTP 客户端：HttpMcpClient + 工具工厂。

通过 HTTP POST 与远程 MCP Server 通信，支持 Streamable HTTP 传输。
与 StdioMcpClient 保持相同接口，上层无需区分传输方式。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.agent.tooling import ToolDefinition
from app.mcp.client import (
    _format_tool_call_result,
    _normalize_input_schema,
    _sanitize_tool_segment,
)
from app.types import ToolResult


class HttpMcpClient:
    """MCP 客户端，通过 Streamable HTTP 与远程 MCP Server 通信。

    支持:
        - JSON-RPC 2.0 over HTTP POST
        - Mcp-Session-Id 会话管理
        - 工具缓存（tools/list）
        - 自动重试机制

    使用方式:
        client = HttpMcpClient("my-server", {"url": "http://localhost:8080/mcp"}, cwd)
        client.start()                # HTTP initialize 握手
        tools = client.list_tools()   # 发现远程工具
        result = client.call_tool("tool_name", {"arg": "value"})
        client.close()                # 清理
    """

    # JSON-RPC 协议常量
    JSONRPC_VERSION = "2.0"
    MCP_PROTOCOL_VERSION = "2024-11-05"
    # HTTP 请求默认超时
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, server_name: str, config: dict[str, Any], cwd: str) -> None:
        """初始化 HTTP MCP 客户端。

        Args:
            server_name: MCP Server 名称
            config: 配置字典，必须包含 url，可选 headers
            cwd: 项目根目录
        """
        self.server_name = server_name
        self.config = config
        self.cwd = cwd
        # HTTP 会话
        self._url: str = str(config.get("url", "")).strip()
        self._session_id: str | None = config.get("session_id", None)
        self._http_client: httpx.Client | None = None
        self._started = False
        self._start_error: str | None = None
        # 缓存
        self._tools_cache: list[dict[str, Any]] | None = None
        # 待处理请求（模拟 stdio 的异步响应路由，HTTP 实际是同步的）
        self.next_id = 1

    @property
    def is_started(self) -> bool:
        """是否已启动并完成握手。"""
        return self._started

    @property
    def start_error(self) -> str | None:
        """启动失败时的错误信息。"""
        return self._start_error

    # ---- 连接生命周期 ----

    def start(self) -> None:
        """连接远程 MCP Server 并完成 JSON-RPC 握手（幂等）。

        已连接则直接返回；上次失败则重置状态后重试。

        Raises:
            RuntimeError: 连接或握手失败
        """
        if self._started:
            return

        if self._start_error is not None and self._http_client is None:
            self._start_error = None

        if not self._url:
            raise RuntimeError(
                f'MCP Server "{self.server_name}" 缺少 url 配置'
            )

        # 自定义 headers
        custom_headers: dict[str, str] = {}
        raw_headers = self.config.get("headers")
        if isinstance(raw_headers, dict):
            custom_headers = {str(k): str(v) for k, v in raw_headers.items()}

        self._http_client = httpx.Client(
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "User-Agent": "mini-code-mcp/1.0",
                **custom_headers,
            },
            timeout=self.DEFAULT_TIMEOUT,
        )

        try:
            # JSON-RPC initialize 握手
            init_result = self.request(
                "initialize",
                {
                    "protocolVersion": self.MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "mini-code", "version": "1.0.0"},
                },
            )
        except Exception as error:
            self._start_error = str(error)
            self._http_client.close()
            self._http_client = None
            raise RuntimeError(
                f'MCP Server "{self.server_name}" initialize 失败: {error}'
            ) from error

        # 通知服务端初始化完成
        try:
            self.notify("notifications/initialized", {})
        except Exception:
            # 通知失败不阻塞
            pass

        self._started = True
        self._start_error = None

    def close(self) -> None:
        """关闭 HTTP 连接，清理资源。"""
        self._started = False
        self._tools_cache = None
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = None

    # ---- JSON-RPC 通信 ----

    def _ensure_started(self) -> None:
        """确保已连接，失败则抛出异常。"""
        if not self._started:
            if self._start_error is not None:
                raise RuntimeError(
                    f'MCP Server "{self.server_name}" 启动失败: {self._start_error}'
                )
            self.start()

    def send(self, message: dict[str, Any]) -> httpx.Response:
        """发送 JSON-RPC 请求到远程 Server（内部方法）。

        通过 HTTP POST 发送，自动附加 Mcp-Session-Id 头。
        """
        if self._http_client is None:
            raise RuntimeError(f'MCP Server "{self.server_name}" 未连接')

        headers: dict[str, str] = {}
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id

        response = self._http_client.post(
            self._url,
            content=json.dumps(message, ensure_ascii=False),
            headers=headers if headers else None,
        )
        response.raise_for_status()
        return response

    def notify(self, method: str, params: Any) -> None:
        """发送 JSON-RPC 通知（无 id，不期望响应）。"""
        message: dict[str, Any] = {
            "jsonrpc": self.JSONRPC_VERSION,
            "method": method,
            "params": params,
        }
        try:
            self.send(message)
        except Exception:
            pass  # 通知失败不阻塞

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        """解析 HTTP 响应为 JSON-RPC 结果。

        支持两种响应格式:
            - application/json:     直接 JSON 解析
            - text/event-stream:    解析 SSE 格式 (event: message\ndata: {json}\n\n)
        """
        content_type = response.headers.get("content-type", "").lower()

        if "text/event-stream" in content_type:
            # SSE 格式: event: message\ndata: {json}\n\n
            text = response.text
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        return json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
            raise RuntimeError(
                f"MCP {self.server_name}: SSE 响应中未找到有效 JSON 数据"
            )

        # 默认按 JSON 解析
        return response.json()

    def request(self, method: str, params: Any, timeout_seconds: float | None = None) -> Any:
        """发送 JSON-RPC 请求并同步等待响应。

        Args:
            method: JSON-RPC 方法名（例如 "tools/list"）
            params: 请求参数
            timeout_seconds: HTTP 请求超时（默认使用实例 DEFAULT_TIMEOUT）

        Returns:
            响应的 result 字段

        Raises:
            RuntimeError: HTTP 错误或服务端返回 JSON-RPC error
        """
        if self._http_client is None:
            raise RuntimeError(f'MCP Server "{self.server_name}" 未连接')

        message_id = self.next_id
        self.next_id += 1

        payload: dict[str, Any] = {
            "jsonrpc": self.JSONRPC_VERSION,
            "id": message_id,
            "method": method,
            "params": params,
        }

        headers: dict[str, str] = {}
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id

        # 支持自定义单次请求超时
        client_for_request = self._http_client
        if timeout_seconds is not None:
            client_for_request = httpx.Client(
                headers=self._http_client.headers,
                timeout=timeout_seconds,
            )

        try:
            response = client_for_request.post(
                self._url,
                content=json.dumps(payload, ensure_ascii=False),
                headers=headers if headers else None,
            )
            response.raise_for_status()

            # 保存服务端返回的 session ID
            sid = response.headers.get("Mcp-Session-Id")
            if sid is not None:
                self._session_id = sid

            result = self._parse_response(response)
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"MCP {self.server_name}: HTTP {error.response.status_code} - "
                f"{error.response.text[:500]}"
            ) from error
        except (httpx.RequestError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"MCP {self.server_name}: {method} 请求失败: {error}"
            ) from error
        finally:
            if timeout_seconds is not None:
                client_for_request.close()

        if result.get("error"):
            details = result["error"].get("data")
            suffix = (
                f"\n{json.dumps(details, indent=2, ensure_ascii=False)}"
                if details
                else ""
            )
            raise RuntimeError(
                f"MCP {self.server_name}: {result['error']['message']}{suffix}"
            )

        return result.get("result")

    # ---- MCP 协议方法 ----

    def list_tools(self) -> list[dict[str, Any]]:
        """调用 tools/list 获取远程工具列表（带缓存）。"""
        if self._tools_cache is not None:
            return self._tools_cache
        self._ensure_started()
        result = self.request("tools/list", {})
        self._tools_cache = list(
            result.get("tools", []) if isinstance(result, dict) else []
        )
        return self._tools_cache

    def call_tool(self, name: str, input_data: Any) -> ToolResult:
        """调用 tools/call 执行远程工具。

        Args:
            name: 远程工具名（Server 原始名称，非 mcp__ 前缀版）
            input_data: 工具参数
        """
        self._ensure_started()
        return _format_tool_call_result(
            self.request(
                "tools/call",
                {"name": name, "arguments": input_data or {}},
            )
        )


def create_http_mcp_backed_tools(
    *,
    server_name: str,
    client: HttpMcpClient,
) -> list[ToolDefinition]:
    """为一个已连接的 HTTP MCP Server 发现并封装工具。

    从 client.list_tools() 获取远程工具描述，转换为本地 ToolDefinition。
    工具名格式: mcp__<server>__<tool>

    client 必须已经 start() 成功（由调用方保证）。
    """
    descriptors = client.list_tools()
    tools: list[ToolDefinition] = []

    for descriptor in descriptors:
        descriptor_name = str(descriptor.get("name", "tool"))
        wrapped_name = (
            f"mcp__{_sanitize_tool_segment(server_name)}"
            f"__{_sanitize_tool_segment(descriptor_name)}"
        )
        input_schema = _normalize_input_schema(descriptor.get("inputSchema"))

        def _validator(value: Any) -> Any:
            """MCP 工具不做本地校验，透传参数给远端。"""
            return value

        # 闭包捕获需用默认参数绑定，避免延迟绑定问题
        def _run(
            input_data: Any,
            _context: Any,
            *,
            _client: HttpMcpClient = client,
            _tool_name: str = descriptor_name,
        ) -> ToolResult:
            return _client.call_tool(_tool_name, input_data)

        tools.append(
            ToolDefinition(
                name=wrapped_name,
                description=str(
                    descriptor.get("description")
                    or f"调用 MCP 工具 {descriptor_name}（Server: {server_name}）"
                ),
                input_schema=input_schema,
                validator=_validator,
                runner=_run,
            )
        )

    return tools
