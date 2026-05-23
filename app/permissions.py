

from pathlib import Path


class PermissionManager:
    """
    权限管理器：负责对工具操作做最基础的安全校验。

    当前第一版只做两类检查：
    1. 路径访问是否超出 workspace_root
    2. 命令是否属于明显危险命令
    """

    def __init__(self,workspace_root: str) -> None:
         # 将工作根目录转成绝对路径，后续所有路径校验都基于它
        self._workspace_root = Path(workspace_root).resolve()

        # 第一版先维护一个简单的危险命令黑名单
        self.dangerous_commands = {
            "rm",
            "del",
            "rmdir",
            "format",
            "shutdown",
            "reboot",
            "mkfs",
        }

    def _resolve_path(self,target_path:str)-> Path: # type: ignore
        """
        将目标路径转成绝对路径。

        如果传入的是相对路径，就按 workspace_root 进行拼接。
        如果传入的是绝对路径，就直接解析。
        """
        path = Path(target_path)
        if not path.is_absolute():
            path = self._workspace_root / path
        return path.resolve()
    

    def ensure_path_access(self,target_path:str)-> Path:
        """
        检查目标路径是否在允许的工作目录范围内。

        返回值：
        - 如果合法，返回解析后的绝对路径

        异常：
        - 如果目标路径超出 workspace_root，抛出 PermissionError
        """

        resolved_path = self._resolve_path(target_path)

        # 判断目标路径是否位于 workspace_root 内
        # 如果不在，就拒绝访问
        try:
            resolved_path.relative_to(self._workspace_root)
        except ValueError as error:
            raise PermissionError(f"访问被拒绝：{resolved_path} 超出工作目录范围") from error
        
        return resolved_path
    
    def ensure_command_allowed(self,command:str)-> None:
        """
        检查命令是否属于危险命令。

        当前第一版做最简单的命令名拦截：
        - 如果命令在危险命令黑名单中，直接拒绝执行
        """

        normalized_command=command.strip().lower()
        if not normalized_command:
            raise PermissionError("命令不能为空")
        
        # 取命令的第一个单词作为主命令
        main_command = normalized_command.split()[0]
        if main_command in self.dangerous_commands:
            raise PermissionError(f"命令被拒绝:{main_command} 是危险命令")
