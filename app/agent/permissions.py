"""权限控制模块（向后兼容重导出）。

   实际实现已迁移至 app/permissions/manager.py。
   此文件保留以确保现有导入路径（如 app.agent.permissions）不受影响。
"""

from app.permissions.manager import (
    PathAccessStatus,
    PathCheckResult,
    PermissionDecision,
    PermissionManager,
)

__all__ = [
    "PathAccessStatus",
    "PathCheckResult",
    "PermissionDecision",
    "PermissionManager",
]
