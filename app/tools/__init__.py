"""工具装配入口，负责把各个内置工具和 MCP 远程工具注册到统一注册表。
   工具集已从旧 15 个工具收敛为 8 核心工具 + memory，
   语义对齐 Claude Code 核心工作流。"""

from app.agent.tooling import ToolRegistry
from app.mcp.manager import McpManager
from app.tools.run_command import run_command_tool
from app.tools.read_file import read_file_tool
from app.tools.edit_file import edit_file_tool
from app.tools.write_file import write_file_tool
from app.tools.glob_files import glob_files_tool
from app.tools.grep_files import grep_files_tool
from app.tools.ask_user import ask_user_tool
from app.tools.agent_dispatch import agent_dispatch_tool
from app.memory.memory_tool import memory_tool


# 8 个核心工具 + memory（对齐 Claude Code 工具集）
# 旧分析工具（find_symbols/find_references/get_ast_info/file_overview/locate_symbol/codebase_map/repo_overview/list_files/make_dirs）已移除
_LOCAL_TOOLS = [
    run_command_tool,
    read_file_tool,
    edit_file_tool,
    write_file_tool,
    glob_files_tool,
    grep_files_tool,
    ask_user_tool,
    agent_dispatch_tool,
    memory_tool,
]


def build_tool_registry(
    cwd: str = "",
    mcp_config: dict[str, dict] | None = None,
    *,
    start_mcp: bool = True,
) -> tuple[ToolRegistry, McpManager]:
    """构建工具注册表，包含本地工具和 MCP 远程工具。

    Args:
        cwd: 项目根目录（用于 MCP 配置读写）
        mcp_config: load_mcp_config() 返回的 MCP 配置，
                    留空或为 None 则不加载任何 MCP Server
        start_mcp: 是否立即启动 MCP Server（设为 False 可异步启动）

    Returns:
        (tool_registry, mcp_manager):
            tool_registry - 包含本地 + MCP 工具的 ToolRegistry
            mcp_manager   - MCP 生命周期管理器（用于 /mcp 命令和退出清理）
    """
    tool_registry = ToolRegistry(tools=list(_LOCAL_TOOLS))
    mcp_manager = McpManager(cwd=cwd, tool_registry=tool_registry)

    # 启动时加载 MCP 配置
    mcp_config = mcp_config or {}
    if mcp_config and start_mcp:
        failed = mcp_manager.bootstrap(mcp_config)
        if failed:
            import sys
            for item in failed:
                print(f"[MCP] 启动失败: {item}", file=sys.stderr)

    return tool_registry, mcp_manager
