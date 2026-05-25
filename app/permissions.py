

from pathlib import Path
import re


class PermissionManager:
    """工具权限管理器：负责路径、命令和输出安全边界。"""

    def __init__(
            self,
            workspace_root: str,
            command_timeout_seconds:int=15,
            max_output_chars:int=8000,
            ) -> None:
        
        # 统一使用绝对路径，避免相对路径绕过校验
        self._workspace_root = Path(workspace_root).resolve()

        # 命令执行超时时间（秒）
        self.command_timeout_seconds=command_timeout_seconds

         # 工具输出最大字符数，超出后截断
        self.max_output_chars = max_output_chars
        
        # 只拦高危命令关键字（可后续继续加）
        self._dangerous_patterns = [
            r"\brm\b",
            r"\bdel\b",
            r"\brmdir\b",
            r"\bformat\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bmkfs\b",
            r"\bdd\b",
            r"\bpoweroff\b",
            r"\bhalt\b",
            r"\bchmod\s+777\b",
            r"\bchown\b",
        ]

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
        
        for pattern in self._dangerous_patterns:
            if re.search(pattern, normalized_command):
                raise PermissionError(f"命令被拒绝：命中危险规则 `{pattern}`")
    def get_command_timeout(self) -> int:
        """返回命令超时秒数。"""
        return self.command_timeout_seconds

    def truncate_output(self, text: str) -> str:
        """按最大字符数截断输出，避免超长响应。"""
        safe_text = text if isinstance(text, str) else str(text)
        if len(safe_text) <= self.max_output_chars:
            return safe_text

        clipped = safe_text[: self.max_output_chars]
        remain = len(safe_text) - self.max_output_chars
        return (
            f"{clipped}\n\n"
            f"[输出已截断：超出 {remain} 个字符，最大保留 {self.max_output_chars} 字符]"
        )