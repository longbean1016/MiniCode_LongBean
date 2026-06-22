# app/mcp/client.py
"""MCP 客户端核心：StdioMcpClient + 安全校验 + 工具工厂。

通过 subprocess 启动外部 MCP Server，使用 JSON-RPC 协议通信。
支持 content-length 和 newline-json 两种协议格式。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from queue import Empty, Queue
from typing import Any

from app.agent.tooling import ToolDefinition
from app.types import ToolResult

# ---- 安全常量 ----

# shell 元字符：禁止在 MCP 命令参数中出现，防止命令注入
DANGEROUS_SHELL_CHARS = set('|&;`$(){}<>\n\r')

# MCP payload 大小上限（防止恶意服务端制造 OOM）
MAX_MCP_PAYLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# MCP Server 允许的命令白名单（相对路径只能在白名单内）
ALLOWED_COMMANDS = {
    'node', 'npm', 'npx', 'python', 'python3', 'pip', 'pip3',
    'uv', 'deno', 'bun', 'cargo', 'go', 'java', 'javac',
    'ruby', 'gem', 'dotnet', 'curl', 'wget',
}


def _sanitize_tool_segment(value: str) -> str:
    """将 server_name 或 tool_name 中的非字母数字字符替换为下划线。

    用于生成 mcp__<server>__<tool> 格式的工具名。
    """
    normalized = "".join(
        char.lower() if char.isalnum() or char in {"_", "-"} else "_"
        for char in value
    )
    normalized = normalized.strip("_")
    return normalized or "tool"


def _validate_mcp_command(command: str) -> None:
    """验证 MCP command 的合法性。

    规则:
        - 禁止路径遍历（.. 和 ~）
        - 禁止系统 shell（cmd.exe / powershell.exe）
        - 相对路径必须在白名单内
    """
    # 禁止路径遍历字符
    if '..' in command or '~' in command:
        raise RuntimeError(f"MCP command 包含路径遍历字符: {command}")

    # 提取命令的基本名称
    base_command = command.lower().replace('\\', '/').split('/')[-1]
    if base_command.endswith('.exe'):
        base_command = base_command[:-4]

    # 禁止危险的系统 shell
    dangerous_shells = {'cmd.exe', 'command.com', 'powershell.exe', 'pwsh.exe'}
    if base_command in dangerous_shells:
        raise RuntimeError(f"禁止使用系统 shell 作为 MCP command: {command}")

    # 相对路径必须在白名单内
    if '\\' not in command and '/' not in command:
        if base_command not in ALLOWED_COMMANDS:
            raise RuntimeError(
                f"MCP command '{command}' 不在允许列表中。"
                f"允许的命令: {', '.join(sorted(ALLOWED_COMMANDS))}"
            )


def _validate_mcp_args(args: list[str]) -> None:
    """验证 MCP 参数不包含危险的 shell 元字符。"""
    for arg in args:
        for char in arg:
            if char in DANGEROUS_SHELL_CHARS:
                raise RuntimeError(
                    f"MCP 参数包含危险字符 '{char}': {arg}"
                )


def _normalize_input_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """规范化 MCP Server 返回的 inputSchema。

    如果 Server 返回了非法 schema，兜底为宽松的 object 类型。
    """
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "additionalProperties": True}


def _format_content_block(block: Any) -> str:
    """格式化 MCP 返回的单个 content block 为文本。"""
    if not isinstance(block, dict):
        return json.dumps(block, indent=2, ensure_ascii=False)
    if block.get("type") == "text" and "text" in block:
        return str(block["text"])
    return json.dumps(block, indent=2, ensure_ascii=False)


def _format_tool_call_result(result: Any) -> ToolResult:
    """将 MCP tools/call 的返回值格式化为 ToolResult。

    MCP 返回格式: {"content": [...], "isError": false}
    """
    if not isinstance(result, dict):
        return ToolResult(ok=True, output=json.dumps(result, indent=2, ensure_ascii=False))

    parts: list[str] = []
    content = result.get("content")
    if isinstance(content, list) and content:
        parts.append("\n\n".join(_format_content_block(block) for block in content))

    if not parts:
        parts.append(json.dumps(result, indent=2, ensure_ascii=False))

    return ToolResult(
        ok=not bool(result.get("isError")),
        output="\n\n".join(parts).strip(),
    )


class StdioMcpClient:
    """MCP 客户端，通过 stdio 与外部 MCP Server 子进程通信。

    核心设计:
        - 惰性启动: 首次调用工具时才启动子进程
        - 自动重试: 连接失败后可在下次请求时重试
        - 工具缓存: tools/list 结果缓存，避免重复请求
        - 协议协商: 自动检测 content-length / newline-json

    使用方式:
        client = StdioMcpClient("my-server", {"command": "npx", "args": [...]}, cwd)
        client.start()                # 启动子进程 + JSON-RPC 握手
        tools = client.list_tools()   # 发现远程工具（带缓存）
        result = client.call_tool("tool_name", {"arg": "value"})
        client.close()                # 终止子进程
    """

    def __init__(self, server_name: str, config: dict[str, Any], cwd: str) -> None:
        self.server_name = server_name
        self.config = config
        self.cwd = cwd
        # 子进程相关
        self.process: subprocess.Popen[bytes] | None = None
        self.protocol: str | None = None  # "content-length" | "newline-json"
        self.next_id = 1  # JSON-RPC 自增请求 ID
        self._pending: dict[int, Queue[Any]] = {}
        self._lock = threading.Lock()
        # stderr 缓存（最近 8 行，用于错误诊断）
        self.stderr_lines: list[str] = []
        # 惰性启动状态
        self._started = False
        self._start_error: str | None = None
        # 结果缓存
        self._tools_cache: list[dict[str, Any]] | None = None
        # 后台线程
        self._stderr_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def start_error(self) -> str | None:
        return self._start_error

    # ---- 子进程生命周期 ----

    def start(self) -> None:
        """启动 MCP Server 子进程并完成 JSON-RPC 握手（幂等）。

        已启动则直接返回；上次失败则重置状态后重试。
        支持 content-length 和 newline-json 两种协议的自动协商。
        """
        if self._started:
            return

        if self._start_error is not None and self.process is None:
            # 上次启动失败，重置后重试
            self._start_error = None

        last_error: Exception | None = None
        # 按优先级尝试协议: 先 content-length，再 newline-json
        for protocol in ["content-length", "newline-json"]:
            try:
                self._spawn_process()
                self.protocol = protocol
                # JSON-RPC initialize 握手
                self.request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "mini-code", "version": "1.0.0"},
                    },
                    timeout_seconds=30.0,
                )
                # 通知服务端初始化完成
                self.notify("notifications/initialized", {})
                self._started = True
                self._start_error = None
                return
            except Exception as error:
                last_error = error
                self.close()

        self._start_error = str(last_error or f'连接 MCP Server "{self.server_name}" 失败')
        raise RuntimeError(self._start_error)

    def close(self) -> None:
        """终止 MCP Server 子进程，清理所有资源。

        跨平台处理:
            Windows: taskkill /T /F /PID (终止进程树)
            Unix:    SIGTERM → 3s 超时 → SIGKILL
        """
        # 通知所有 pending 请求：连接已关闭
        with self._lock:
            for q in self._pending.values():
                q.put({
                    "error": {
                        "message": f'MCP Server "{self.server_name}" 在处理请求前关闭'
                    }
                })
            self._pending.clear()

        if self.process is not None:
            try:
                if os.name == "nt":
                    # Windows: 终止整个进程树
                    try:
                        subprocess.run(
                            ["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                            capture_output=True,
                            timeout=5,
                        )
                    except (subprocess.TimeoutExpired, Exception):
                        try:
                            self.process.kill()
                        except OSError:
                            pass
                else:
                    # Unix: 先 SIGTERM 优雅退出，超时后强制 SIGKILL
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        try:
                            self.process.kill()
                        except OSError:
                            pass
            except OSError:
                pass
            finally:
                self.process = None

        self.protocol = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._started = False
        self._tools_cache = None

    # ---- 内部: 子进程启停 ----

    def _spawn_process(self) -> None:
        """启动 MCP Server 子进程。

        验证 command/args 安全性后，通过 subprocess.Popen 启动，
        并启动 stderr 消费线程。
        """
        command = str(self.config.get("command", "")).strip()
        if not command:
            raise RuntimeError(f'MCP Server "{self.server_name}" 缺少 command 配置')

        # 安全验证
        _validate_mcp_command(command)
        _validate_mcp_args(list(self.config.get("args", []) or []))

        # 解析可执行文件的实际路径（Windows 上 npx → npx.cmd 等）
        resolved = shutil.which(command)
        if resolved is None:
            raise RuntimeError(
                f'命令未找到: {command}。请确保已安装并在 PATH 中可用。'
            )

        # 处理环境变量
        env = os.environ.copy()
        for key, value in dict(self.config.get("env", {}) or {}).items():
            env[str(key)] = str(value)

        popen_kwargs: dict = {}
        server_args = list(self.config.get("args", []) or [])

        # Windows 上 .cmd/.bat 文件必须通过 cmd.exe /c 启动（CreateProcess 无法直接执行）
        if os.name == "nt" and resolved.lower().endswith((".cmd", ".bat")):
            CREATE_NO_WINDOW = 0x08000000
            popen_kwargs["creationflags"] = CREATE_NO_WINDOW
            popen_args = ["cmd.exe", "/c", resolved, *server_args]
        elif os.name == "nt":
            CREATE_NO_WINDOW = 0x08000000
            popen_kwargs["creationflags"] = CREATE_NO_WINDOW
            popen_args = [resolved, *server_args]
        else:
            popen_args = [resolved, *server_args]

        try:
            self.process = subprocess.Popen(
                popen_args,
                cwd=self.cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **popen_kwargs,
            )
        except FileNotFoundError:
            raise RuntimeError(f"命令未找到: {command}。请确保已安装并在 PATH 中可用。") from None

        self.stderr_lines = []
        with self._lock:
            self._pending = {}

        # 启动 stderr 消费线程
        self._stderr_thread = threading.Thread(target=self._consume_stderr, daemon=True)
        self._stderr_thread.start()

    def _is_process_alive(self) -> bool:
        """检查子进程是否仍在运行。"""
        return self.process is not None and self.process.poll() is None

    def _ensure_started(self) -> None:
        """确保子进程已启动（惰性启动入口）。

        如果进程意外退出，先清理再重试。
        """
        if self._started and not self._is_process_alive():
            self.close()
        if not self._started:
            self.start()

    # ---- 内部: stderr/stdout 消费 ----

    def _consume_stderr(self) -> None:
        """后台线程：持续读取 MCP Server 的 stderr，保留最近 8 行用于错误诊断。"""
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            try:
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self.stderr_lines.append(text)
                    self.stderr_lines = self.stderr_lines[-8:]
            except Exception:
                continue

    def _ensure_stdout_thread(self) -> None:
        """确保 stdout 消费线程已启动（首次 send 时惰性启动）。"""
        if self._stdout_thread is not None:
            return
        self._stdout_thread = threading.Thread(target=self._consume_stdout, daemon=True)
        self._stdout_thread.start()

    def _consume_stdout(self) -> None:
        """后台线程：持续读取 MCP Server 的 stdout，解析 JSON-RPC 消息。

        支持两种协议:
            content-length: HTTP 风格头部 + body
            newline-json:   每行一个 JSON 消息
        """
        assert self.process is not None and self.process.stdout is not None

        try:
            while True:
                try:
                    line_bytes = self.process.stdout.readline()
                except (OSError, ValueError):
                    break
                if not line_bytes:
                    break

                try:
                    line = line_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                stripped = line.strip()
                if not stripped:
                    continue

                # 消息体大小检查
                if len(line_bytes) > MAX_MCP_PAYLOAD_BYTES:
                    self.stderr_lines.append(
                        f"MCP payload 过大: {len(line_bytes)} bytes (限制 {MAX_MCP_PAYLOAD_BYTES})"
                    )
                    continue

                # 自动检测协议（仅使用局部变量，不写回 self.protocol）
                detected_protocol = self.protocol
                if detected_protocol is None:
                    if line.lower().startswith("content-length:"):
                        detected_protocol = "content-length"
                    else:
                        detected_protocol = "newline-json"

                if detected_protocol == "newline-json":
                    try:
                        self._handle_message(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
                else:
                    # content-length 协议: 解析头部 → 读取 body
                    header_lines = [line.rstrip("\r\n")]
                    while True:
                        try:
                            next_bytes = self.process.stdout.readline()
                        except (OSError, ValueError):
                            break
                        if not next_bytes:
                            return
                        try:
                            next_line = next_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            return
                        h_stripped = next_line.rstrip("\r\n")
                        if h_stripped == "":
                            break
                        header_lines.append(h_stripped)

                    # 解析 Content-Length 头部
                    content_length = 0
                    for header in header_lines:
                        if header.lower().startswith("content-length:"):
                            try:
                                content_length = int(header.split(":", 1)[1].strip())
                            except ValueError:
                                pass
                            break

                    if content_length > MAX_MCP_PAYLOAD_BYTES:
                        self.stderr_lines.append(
                            f"MCP payload 过大: {content_length} bytes (限制 {MAX_MCP_PAYLOAD_BYTES})"
                        )
                        continue

                    if content_length > 0:
                        # 逐块读取，避免 content_length 偏大时永久阻塞
                        body_chunks: list[bytes] = []
                        remaining = content_length
                        while remaining > 0:
                            try:
                                chunk = self.process.stdout.read(min(remaining, 65536))
                            except (OSError, ValueError):
                                break
                            if not chunk:
                                break
                            body_chunks.append(chunk)
                            remaining -= len(chunk)
                        if remaining > 0:
                            # 未读取完整的 body，放弃处理
                            break
                        body_bytes = b"".join(body_chunks)
                        try:
                            self._handle_message(json.loads(body_bytes.decode("utf-8")))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass
        finally:
            # 进程退出时，通知所有 pending 请求
            if self.process:
                exit_code = self.process.poll()
                error_msg = {
                    "error": {
                        "code": -1,
                        "message": f"MCP Server 进程退出 (exit_code={exit_code})",
                    }
                }
                with self._lock:
                    for q in self._pending.values():
                        q.put(error_msg)
                    self._pending.clear()

    def _handle_message(self, message: dict[str, Any]) -> None:
        """将 JSON-RPC 响应路由到对应的 pending 请求队列。"""
        message_id = message.get("id")
        if not isinstance(message_id, int):
            return
        with self._lock:
            queue = self._pending.pop(message_id, None)
            if queue is not None:
                queue.put(message)

    # ---- JSON-RPC 通信 ----

    def send(self, message: dict[str, Any]) -> None:
        """发送 JSON-RPC 消息到 MCP Server 的 stdin。

        根据协议类型自动选择编码方式:
            content-length: HTTP 风格头部 + JSON body
            newline-json:   每行一个 JSON
        """
        if self.process is None or self.process.stdin is None:
            raise RuntimeError(f'MCP Server "{self.server_name}" 未在运行')

        payload_bytes = json.dumps(message, ensure_ascii=False).encode("utf-8")

        if self.protocol == "newline-json":
            self.process.stdin.write(payload_bytes + b"\n")
            self.process.stdin.flush()
            self._ensure_stdout_thread()
            return

        # content-length 协议
        header = f"Content-Length: {len(payload_bytes)}\r\n\r\n".encode("utf-8")
        self.process.stdin.write(header + payload_bytes)
        self.process.stdin.flush()
        self._ensure_stdout_thread()

    def notify(self, method: str, params: Any) -> None:
        """发送 JSON-RPC 通知（无 id，不期望响应）。"""
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any, timeout_seconds: float = 5.0) -> Any:
        """发送 JSON-RPC 请求并同步等待响应。

        Args:
            method: JSON-RPC 方法名（例如 "tools/list"）
            params: 请求参数
            timeout_seconds: 超时时间

        Returns:
            响应的 result 字段

        Raises:
            RuntimeError: 超时或服务端返回错误
        """
        response_queue: Queue[Any] = Queue(maxsize=1)
        with self._lock:
            message_id = self.next_id
            self.next_id += 1
            self._pending[message_id] = response_queue

        self.send({
            "jsonrpc": "2.0",
            "id": message_id,
            "method": method,
            "params": params,
        })

        try:
            message = response_queue.get(timeout=timeout_seconds)
        except Empty:
            with self._lock:
                self._pending.pop(message_id, None)
            stderr = "\n".join(self.stderr_lines)
            raise RuntimeError(
                f"MCP {self.server_name}: {method} 请求超时"
                + (f"\nstderr:\n{stderr}" if stderr else "")
            )

        if message.get("error"):
            details = message["error"].get("data")
            suffix = f"\n{json.dumps(details, indent=2, ensure_ascii=False)}" if details else ""
            raise RuntimeError(
                f"MCP {self.server_name}: {message['error']['message']}{suffix}"
            )

        return message.get("result")

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
            self.request("tools/call", {"name": name, "arguments": input_data or {}})
        )


def create_mcp_backed_tools(
    *,
    server_name: str,
    client: StdioMcpClient,
) -> list[ToolDefinition]:
    """为一个已连接的 MCP Server 发现并封装工具。

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

        # 注意: 闭包捕获需用默认参数绑定，避免延迟绑定问题
        def _run(
            input_data: Any,
            _context: Any,
            *,
            _client: StdioMcpClient = client,
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
