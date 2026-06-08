"""工具装配入口，负责把各个内置工具注册到统一注册表。"""

from app.agent.tooling import ToolRegistry
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


def build_tool_registry() -> ToolRegistry:
    """
    构建默认工具注册表。

    当前分两类：
    1. 基础文件与命令工具
    2. 代码理解与代码导航工具
    """

    return ToolRegistry(
        tools=[
            # 注册列出目录内容的工具
            list_files_tool,
            # 注册读取文件内容的工具
            read_file_tool,
            # 注册按文本搜索文件内容的工具
            grep_files_tool,
            # 注册执行命令的工具
            run_command_tool,
            # 注册写入完整文件的工具
            write_file_tool,
            # 注册局部编辑文件内容的工具
            edit_file_tool,
            # 注册创建目录的工具
            make_dirs_tool,
            # 注册仓库级结构概览工具
            repo_overview_tool,
            # 注册代码库第一层地图工具
            codebase_map_tool,
            # 注册单文件结构概览工具
            file_overview_tool,
            # 注册 Python 符号定义扫描工具
            find_symbols_tool,
            # 注册 Python 符号引用搜索工具
            find_references_tool,
            # 注册符号定位组合工具
            locate_symbol_tool,
            # 注册 Python AST 结构摘要工具
            get_ast_info_tool,
        ]
    )
