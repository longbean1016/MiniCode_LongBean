"""工具装配入口，负责把各个内置工具和 MCP 远程工具注册到统一注册表。"""

from app.agent.tooling import ToolRegistry
from app.mcp.manager import McpManager
from app.tools.codebase_map import codebase_map_tool
from app.tools.edit_file import edit_file_tool
from app.tools.file_overview import file_overview_tool
from app.tools.find_references import find_references_tool
from app.tools.find_symbols import find_symbols_tool
from app.tools.get_ast_info import get_ast_info_tool
from app.tools.grep_files import grep_files_tool
from app.tools.list_files import list_files_tool
from app.tools.make_dirs import make_dirs_tool
from app.tools.locate_symbol import locate_symbol_tool
from app.tools.read_file import read_file_tool
from app.tools.repo_overview import repo_overview_tool
from app.tools.run_command import run_command_tool
from app.tools.write_file import write_file_tool


# 本地工具清单
_LOCAL_TOOLS = [
    list_files_tool,
    read_file_tool,
    grep_files_tool,
    run_command_tool,
    write_file_tool,
    edit_file_tool,
    make_dirs_tool,
    repo_overview_tool,
    codebase_map_tool,
    file_overview_tool,
    find_symbols_tool,
    find_references_tool,
    locate_symbol_tool,
    get_ast_info_tool,
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
    # 先创建空的 ToolRegistry，后续动态注册 MCP 工具
    tool_registry = ToolRegistry(tools=list(_LOCAL_TOOLS))
    # 创建 MCP 管理器
    mcp_manager = McpManager(cwd=cwd, tool_registry=tool_registry)

    # 启动时加载 MCP 配置
    mcp_config = mcp_config or {}
    if mcp_config and start_mcp:
        failed = mcp_manager.bootstrap(mcp_config)
        if failed:
            # 启动失败不阻塞，输出日志提示
            import sys
            for item in failed:
                print(f"[MCP] 启动失败: {item}", file=sys.stderr)

    return tool_registry, mcp_manager
