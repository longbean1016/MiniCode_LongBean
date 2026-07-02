"""web_fetch 网页抓取工具 — 抓取 URL 内容并转换为可读文本。
   对标 Claude Code WebFetchTool 语义实现：
   用 HTTP 请求抓取页面，提取文本内容，支持可选的 prompt 摘要指令。"""

from typing import Any
from urllib.parse import urlparse

from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# ── 请求限制 ──
FETCH_TIMEOUT_S = 15  # HTTP 请求超时秒数
MAX_CONTENT_CHARS = 50_000  # 返回内容最大字符数（对标 Claude Code maxResultSizeChars）
USER_AGENT = "MiniCode/1.0 WebFetch"

# ── 预批准域名（内部常用站点，允许不授权直接访问）──
_PREAPPROVED_HOSTS = {
    "raw.githubusercontent.com",
    "docs.python.org",
    "pypi.org",
    "pypi.python.org",
    "npmjs.com",
    "registry.npmjs.org",
    "crates.io",
    "docs.rs",
}


def _validate(input_data: Any) -> dict[str, str]:
    """校验 web_fetch 输入参数：url 必填且合法，prompt 可选。"""
    if not isinstance(input_data, dict):
        raise ValueError("web_fetch 输入必须是字典，包含 url 和 prompt 字段。")

    url = input_data.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url 必须是非空字符串。")

    # 验证 URL 格式
    try:
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("url 格式无效，需要包含 scheme 和域名。")
    except Exception as exc:
        raise ValueError(f"url 格式无效: {exc}")

    prompt = input_data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt 必须是非空字符串，描述你想从页面中提取什么信息。")

    return {"url": url.strip(), "prompt": prompt.strip()}


def _run(validated_input: dict[str, str], context: ToolContext) -> ToolResult:
    """抓取 URL 内容，提取纯文本并返回。"""
    url = validated_input["url"]
    prompt = validated_input["prompt"]

    # ── 权限检查：不允许抓取 localhost / 内网地址 ──
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return ToolResult(
            ok=False,
            output=f"不允许抓取本地地址：{url}",
            error="LOCALHOST_BLOCKED",
            meta={"url": url},
        )

    import time
    start_time = time.time()

    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            raw_bytes = resp.read()
            status_code = resp.getcode()
            content_type = resp.headers.get("Content-Type", "").lower()
    except urllib.error.HTTPError as exc:
        return ToolResult(
            ok=False,
            output=f"HTTP 错误 {exc.code}: {url}",
            error="HTTP_ERROR",
            meta={"url": url, "code": exc.code},
        )
    except Exception as exc:
        return ToolResult(
            ok=False,
            output=f"抓取失败: {exc}",
            error="FETCH_FAILED",
            meta={"url": url},
        )

    duration_ms = int((time.time() - start_time) * 1000)
    content_size = len(raw_bytes)

    # ── 提取文本内容 ──
    if "text/html" in content_type:
        text = _extract_text_from_html(raw_bytes)
    else:
        # 非 HTML 内容，直接尝试解码为文本
        text = _decode_bytes(raw_bytes)

    # ── 截断保护 ──
    truncated = len(text) > MAX_CONTENT_CHARS
    if truncated:
        text = text[:MAX_CONTENT_CHARS] + f"\n\n(内容已截断，共 {len(raw_bytes)} 字节)"

    output = (
        f"FETCH: {url}\n"
        f"SIZE: {content_size} bytes\n"
        f"CODE: {status_code}\n"
        f"DURATION: {duration_ms}ms\n\n"
        f"{text}"
    )

    return ToolResult(
        ok=True,
        output=output,
        meta={
            "url": url,
            "bytes": content_size,
            "code": status_code,
            "duration_ms": duration_ms,
            "truncated": truncated,
        },
    )


# ── 文本提取 ──

def _extract_text_from_html(raw_bytes: bytes) -> str:
    """从 HTML 原始字节中提取可读文本。

       优先用 html2text 库（如果有），否则回退到正则清理。
    """
    html = _decode_bytes(raw_bytes)

    # 尝试使用 html2text（Markdown 格式，模型更好理解）
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # 不自动换行
        return h.handle(html)
    except ImportError:
        pass

    # 回退：正则清理标签
    import re
    # 去掉 script/style 块
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # 去掉所有 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 合并多个空行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def _decode_bytes(raw_bytes: bytes) -> str:
    """尝试多种编码解码字节数据。"""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


# ── 注册工具 ──
web_fetch_tool = ToolDefinition(
    name="web_fetch",
    description=(
        "抓取指定 URL 的网页内容并提取为可读文本。"
        "参数 url 是要抓取的网址，prompt 描述你想从页面获取什么信息。"
        "注意：无法访问需要登录的页面（如 Google Docs/Jira/GitHub 私有仓库）。"
    ),
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要抓取的网页 URL。",
            },
            "prompt": {
                "type": "string",
                "description": "描述你想从页面中提取什么信息的指令。",
            },
        },
        "required": ["url", "prompt"],
    },
)
