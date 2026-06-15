"""最小 MCP Server — 仅用于测试 /mcp 功能。
提供两个简单工具：
    - echo: 回显输入文本
    - add: 两数相加
"""
import json
import sys


def handle_request(request: dict) -> dict | None:
    """处理单个 JSON-RPC 请求，返回响应或 None（通知类请求）。"""
    req_id = request.get("id")
    method = request.get("method", "")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "test-mcp-server",
                    "version": "0.1.0",
                },
            },
        }

    if method == "notifications/initialized":
        return None  # 通知，不需要响应

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "回显输入文本",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "要回显的文本",
                                }
                            },
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "add",
                        "description": "计算两个数字的和",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": {
                                    "type": "number",
                                    "description": "第一个数字",
                                },
                                "b": {
                                    "type": "number",
                                    "description": "第二个数字",
                                },
                            },
                            "required": ["a", "b"],
                        },
                    },
                ]
            },
        }

    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "echo":
            text = arguments.get("text", "")
            result_text = f"Echo: {text}"
        elif tool_name == "add":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            result_text = f"{a} + {b} = {a + b}"
        else:
            result_text = f"未知工具: {tool_name}"

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False,
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"未知方法: {method}"},
    }


def run_loop() -> None:
    """主循环：从 stdin 读取 JSON-RPC 请求，写入响应到 stdout。
    使用 newline-json 协议（每行一个完整的 JSON 对象）。
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_loop()
