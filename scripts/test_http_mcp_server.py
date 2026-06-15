# scripts/test_http_mcp_server.py
"""最小 HTTP MCP Server — 用于测试 /mcp add --url 功能。
基于 Python 标准库 http.server，无需额外依赖。

提供两个工具: echo、random_number

启动方式: python scripts/test_http_mcp_server.py [--port 8080]
"""

from __future__ import annotations

import json
import random
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

# MCP 协议常量
JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "test-http-mcp"
SERVER_VERSION = "0.1.0"

# 模拟的工具列表
TOOLS = [
    {
        "name": "echo",
        "description": "回显输入文本",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要回显的文本"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "random_number",
        "description": "生成指定范围内的随机整数",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min": {"type": "integer", "description": "最小值（含），默认 1"},
                "max": {"type": "integer", "description": "最大值（含），默认 100"},
            },
        },
    },
]


class McpHandler(BaseHTTPRequestHandler):
    """处理 MCP JSON-RPC 请求。"""

    # 禁用请求日志，避免干扰
    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        """发送 JSON 响应。"""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Mcp-Session-Id", "test-session-001")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        """处理 POST 请求 — 所有 MCP JSON-RPC 都走这个入口。"""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            request = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, ValueError):
            self._send_json(
                {"jsonrpc": JSONRPC_VERSION, "error": {"code": -32700, "message": "Parse error"}},
                status=400,
            )
            return

        response = self._handle_request(request)
        if response is not None:
            self._send_json(response)

    def do_GET(self) -> None:
        """GET 请求返回服务器信息（方便浏览器测试连接）。"""
        info = {
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "protocol": "MCP Streamable HTTP",
            "tools": len(TOOLS),
        }
        self._send_json(info)

    def do_DELETE(self) -> None:
        """DELETE 请求用于关闭会话（可选）。"""
        self.send_response(204)
        self.end_headers()

    def _handle_request(self, request: dict) -> dict | None:
        """处理 JSON-RPC 请求，返回响应或 None。"""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        # ---- initialize ----
        if method == "initialize":
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            }

        # ---- notifications/initialized ----
        if method == "notifications/initialized":
            return None  # 通知不响应

        # ---- tools/list ----
        if method == "tools/list":
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": req_id,
                "result": {"tools": TOOLS},
            }

        # ---- tools/call ----
        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            if tool_name == "echo":
                text = arguments.get("text", "")
                result_text = f"[HTTP MCP] Echo: {text}"
            elif tool_name == "random_number":
                lo = int(arguments.get("min", 1))
                hi = int(arguments.get("max", 100))
                value = random.randint(lo, hi)
                result_text = f"[HTTP MCP] 随机数 ({lo}-{hi}): {value}"
            else:
                return {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": req_id,
                    "error": {"code": -32601, "message": f"未知工具: {tool_name}"},
                }

            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                },
            }

        # ---- 未知方法 ----
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "error": {"code": -32601, "message": f"未知方法: {method}"},
        }


def main() -> None:
    port = 8080
    if len(sys.argv) > 1:
        if sys.argv[1] == "--port" and len(sys.argv) > 2:
            port = int(sys.argv[2])

    server = HTTPServer(("127.0.0.1", port), McpHandler)
    print(f"🧪 HTTP MCP Server 已启动: http://127.0.0.1:{port}")
    print(f"   工具: echo, random_number")
    print(f"   Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
