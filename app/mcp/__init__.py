"""MCP (Model Context Protocol) 协议扩展模块。

提供:
    - StdioMcpClient: 通过 stdio 与本地 MCP Server 子进程通信
    - HttpMcpClient:  通过 HTTP POST 与远程 MCP Server 通信
    - McpManager:     MCP Server 生命周期管理（add/remove/list/dispose）
    - config:         .mcp.json 配置文件读写（支持 stdio 和 HTTP 两种传输）
"""
