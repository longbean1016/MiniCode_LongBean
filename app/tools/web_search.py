"""web_search 网页搜索工具 — 执行网络搜索并返回结果列表。
   对标 Claude Code WebSearchTool 语义实现：
   通过搜索 API 或 DuckDuckGo 引擎查询，返回标题/URL/摘要。"""

from typing import Any

from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

# ── 搜索限制 ──
DEFAULT_NUM_RESULTS = 8
MAX_NUM_RESULTS = 20
SEARCH_TIMEOUT_S = 15
USER_AGENT = "MiniCode/1.0 WebSearch"


def _validate(input_data: Any) -> dict[str, Any]:
    """校验 web_search 输入参数：query 必填，allowed_domains/blocked_domains 互斥。"""
    if not isinstance(input_data, dict):
        raise ValueError("web_search 输入必须是字典，包含 query 字段。")

    query = input_data.get("query")
    if not isinstance(query, str) or len(query.strip()) < 2:
        raise ValueError("query 必须是非空字符串（最少 2 字符）。")

    allowed_domains = input_data.get("allowed_domains")
    if allowed_domains is not None and not isinstance(allowed_domains, list):
        raise ValueError("allowed_domains 必须是字符串列表。")

    blocked_domains = input_data.get("blocked_domains")
    if blocked_domains is not None and not isinstance(blocked_domains, list):
        raise ValueError("blocked_domains 必须是字符串列表。")

    if allowed_domains and blocked_domains:
        raise ValueError("不能同时指定 allowed_domains 和 blocked_domains。")

    raw_num = input_data.get("num_results", DEFAULT_NUM_RESULTS)
    try:
        num_results = int(raw_num)
    except (TypeError, ValueError):
        num_results = DEFAULT_NUM_RESULTS
    if num_results < 1 or num_results > MAX_NUM_RESULTS:
        raise ValueError(f"num_results 必须在 1 到 {MAX_NUM_RESULTS} 之间。")

    return {
        "query": query.strip(),
        "allowed_domains": allowed_domains or [],
        "blocked_domains": blocked_domains or [],
        "num_results": num_results,
    }


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """执行网页搜索并返回结果列表。"""
    query = validated_input["query"]
    allowed_domains = validated_input["allowed_domains"]
    blocked_domains = validated_input["blocked_domains"]
    num_results = validated_input["num_results"]

    import time
    start_time = time.time()

    # ── 优先尝试 DuckDuckGo Lite HTML 搜索（无需 API Key）──
    results = _search_duckduckgo(query, num_results, allowed_domains, blocked_domains)

    duration_sec = round(time.time() - start_time, 2)

    if not results:
        return ToolResult(
            ok=True,
            output=f"搜索完成，未找到相关结果: \"{query}\"",
            meta={
                "query": query,
                "num_results": 0,
                "duration_seconds": duration_sec,
            },
        )

    # ── 格式化输出（对标 Claude Code 格式）──
    lines = [
        f"Web 搜索结果: \"{query}\"",
        f"找到 {len(results)} 条结果 ({duration_sec}s)",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['title']}]({r['url']})")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        lines.append("")

    lines.append('提示：回答时请在末尾附上 Sources: 来源链接列表。')

    return ToolResult(
        ok=True,
        output="\n".join(lines),
        meta={
            "query": query,
            "num_results": len(results),
            "duration_seconds": duration_sec,
        },
    )


# ── DuckDuckGo 搜索实现 ──

def _search_duckduckgo(
    query: str,
    num: int,
    allowed_domains: list[str],
    blocked_domains: list[str],
) -> list[dict[str, str]]:
    """使用 DuckDuckGo Lite 接口搜索，返回 (标题, URL, 摘要) 列表。

       DuckDuckGo 支持通过 site: 语法限定域名范围。
    """
    search_query = query
    if allowed_domains:
        # 限制搜索范围到指定域名
        site_clause = " OR ".join(f"site:{d}" for d in allowed_domains)
        search_query = f"{query} ({site_clause})"
    if blocked_domains:
        # 排除指定域名
        exclude_clause = " ".join(f"-site:{d}" for d in blocked_domains)
        search_query = f"{query} {exclude_clause}"

    try:
        import urllib.request
        import urllib.parse
        import re

        # DuckDuckGo Lite HTML 搜索（纯 HTML，无 JS，稳定）
        params = urllib.parse.urlencode({"q": search_query})
        req_url = f"https://lite.duckduckgo.com/lite/?{params}"
        req = urllib.request.Request(req_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT_S) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    # ── 解析搜索结果 HTML ──
    # DuckDuckGo Lite 的搜索结果结构：
    # <a rel="nofollow" class="result-link" href="...">标题</a>
    # <span class="result-snippet">摘要</span>
    results: list[dict[str, str]] = []

    # 匹配 result-link 中的链接和标题
    link_pattern = re.compile(
        r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<span[^>]*class="result-snippet"[^>]*>(.*?)</span>',
        re.DOTALL | re.IGNORECASE,
    )

    links = link_pattern.findall(html)
    snippets = [s.strip() for s in snippet_pattern.findall(html)]

    for i, (url, title) in enumerate(links):
        if i >= num:
            break
        # 清理标题中的 HTML 标签
        clean_title = re.sub(r'<[^>]+>', '', title).strip()
        snippet = _clean_html(snippets[i]) if i < len(snippets) else ""
        results.append({
            "title": clean_title,
            "url": url,
            "snippet": snippet[:300],
        })

    return results


def _clean_html(text: str) -> str:
    """去掉 HTML 标签和实体编码。"""
    import re
    import html as _html
    text = re.sub(r'<[^>]+>', '', text)
    text = _html.unescape(text)
    return text.strip()


# ── 注册工具 ──
web_search_tool = ToolDefinition(
    name="web_search",
    description=(
        "搜索网络并返回结果列表（标题、URL、摘要）。"
        "用于查找最新信息、文档或事实时使用。"
        "注意：仅支持美国地区的英文搜索结果。"
    ),
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询词（最少 2 字符）。",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "只返回这些域名的结果。与 blocked_domains 互斥。",
            },
            "blocked_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "排除这些域名的结果。与 allowed_domains 互斥。",
            },
            "num_results": {
                "type": "integer",
                "description": f"返回结果数量，默认 {DEFAULT_NUM_RESULTS}，最大 {MAX_NUM_RESULTS}。",
            },
        },
        "required": ["query"],
    },
)
