
from typing import Any

from app.permissions import PermissionManager
from app.tooling import ToolDefinition
from app.types import ToolContext, ToolResult

DEFAULT_MAX_ENTRIES = 200
MAX_MAX_ENTRIES = 1_000


def _validate(input_data: Any) -> dict[str, int | str]:
    """
    校验并规范化工具输入。

    支持的输入格式：
    1. {"path": "app"}
    2. {} 或 None，默认使用当前目录 "."
    """
    if input_data is None:
        return {"path": ".", "max_entries": DEFAULT_MAX_ENTRIES}

    if not isinstance(input_data, dict):
        raise ValueError("输入必须是一个字典，包含 'path' 键")

    path = input_data.get("path", ".")
    if not isinstance(path, str):
        raise ValueError("路径必须是一个字符串")

    raw_max_entries = input_data.get("max_entries", DEFAULT_MAX_ENTRIES)
    try:
        max_entries = int(raw_max_entries)
    except (TypeError, ValueError) as error:
        raise ValueError("max_entries 必须是整数") from error

    if max_entries < 1 or max_entries > MAX_MAX_ENTRIES:
        raise ValueError(f"max_entries 必须在 1 到 {MAX_MAX_ENTRIES} 之间")

    return {"path": path, "max_entries": max_entries}


def _run(validated_input: dict[str, int | str], context: ToolContext) -> ToolResult:
    """
    执行列出文件工具。

    功能：
    1. 检查目标路径是否在允许的工作目录内
    2. 判断路径是否存在
    3. 如果是文件，直接返回文件信息
    4. 如果是目录，返回目录下的文件和子目录列表
    """

    # 第一步：创建权限管理器
    # 这里把 context.cwd 当作当前允许操作的工作根目录
    permission_manager = PermissionManager(context.cwd)

    # 第二步：拿到用户想查看的原始路径
    raw_path = validated_input["path"]

    # 第三步：权限管理器检查路径访问是否合法
    # 如果路径越界，比如跑到工作目录外面，会直接抛出 PermissionError
    target_path = permission_manager.ensure_path_access(raw_path)
    max_entries = int(validated_input["max_entries"])

    # 第四步：检查路径是否存在
    if not target_path.exists():
        return ToolResult(
            ok=False,
            output=f"路径不存在: {target_path}"
        )

    # 第五步：如果目标本身就是文件，
    # 那就没必要再迭代目录了，直接告诉调用方这是一个文件
    if target_path.is_file():
        return ToolResult(
            ok=True,
            output=(
                f"ROOT: {raw_path}\n"
                "TOTAL_ENTRIES: 1\n"
                "RETURNED_ENTRIES: 1\n"
                "TRUNCATED: no\n\n"
                f"file {target_path.name}"
            ),
            meta={
                "search_root": raw_path,
                "total_entries": 1,
                "returned_entries": 1,
                "truncated": False,
            },
        )

    # 第六步：如果是目录，就读取目录下所有子项
    # 用名字的小写排序，保证输出稳定，便于测试和调试

    entries = sorted(target_path.iterdir(), key=lambda item: item.name.lower())

    # 第七步：如果目录是空的，返回一个明确提示
    if not entries:
        return ToolResult(
            ok=True,
            output=(
                f"ROOT: {raw_path}\n"
                "TOTAL_ENTRIES: 0\n"
                "RETURNED_ENTRIES: 0\n"
                "TRUNCATED: no\n\n"
                "(empty)"
            ),
            meta={
                "search_root": raw_path,
                "total_entries": 0,
                "returned_entries": 0,
                "truncated": False,
            },
        )

    # 第八步：把每个子项转换成文本行
    # 目录前面标记 dir，文件前面标记 file
    lines: list[str] = []
    for entry in entries:
        prefix = "dir " if entry.is_dir() else "file"
        lines.append(f"{prefix} {entry.name}")

    returned_lines = lines[:max_entries]
    truncated = len(lines) > max_entries
    header_lines = [
        f"ROOT: {raw_path}",
        f"TOTAL_ENTRIES: {len(lines)}",
        f"RETURNED_ENTRIES: {len(returned_lines)}",
        f"TRUNCATED: {'yes' if truncated else 'no'}",
        "",
    ]

    # 第九步：把结果拼成一个多行字符串返回。
    # 这里先做一次工具自身限额，避免目录很大时原始输出直接失控。
    return ToolResult(
        ok=True,
        output="\n".join(header_lines + returned_lines),
        meta={
            "search_root": raw_path,
            "total_entries": len(lines),
            "returned_entries": len(returned_lines),
            "truncated": truncated,
        },
    )

# 第十步：把上面的校验函数和执行函数组装成一个正式工具定义
# 后面 ToolRegistry 注册的就是这个对象
list_files_tool = ToolDefinition(
    name="list_files",
    description="列出指定目录下的文件和文件夹",
    validator=_validate,
    runner=_run, # type: ignore
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要列出的目录路径，默认当前目录",
            },
            "max_entries": {
                "type": "integer",
                "description": f"最多返回多少条目录项，默认 {DEFAULT_MAX_ENTRIES}，最大 {MAX_MAX_ENTRIES}。",
            }
        },
        "required": [],
    },
)
