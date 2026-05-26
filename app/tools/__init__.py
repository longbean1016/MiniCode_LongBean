from app.tools.write_file import write_file_tool
from app.tools.list_files import list_files_tool
from app.tools.read_file import read_file_tool
from app.tools.grep_files import grep_files_tool
from app.tools.run_command import run_command_tool
from app.tools.edit_file import edit_file_tool
from app.tools.make_dirs import make_dirs_tool
from app.tooling import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    """
    构建默认工具注册表。

    第一版先只注册最基础的工具，
    这样方便一点点调试，不要一开始就塞太多工具。
    """
    return ToolRegistry(
        tools=[
            # 注册列出目录工具
            list_files_tool,
            # 注册读取文件工具
            read_file_tool,
            # 注册搜索文件内容工具
            grep_files_tool,
            # 注册运行命令工具
            run_command_tool,
            # 注册写入文件工具
            write_file_tool,
            # 注册局部修改文件内容工具
            edit_file_tool,
            # 注册创建目录与父目录工具
            make_dirs_tool
        ]
    )
