from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

"""代码导航公共辅助模块，复用路径、AST 和输出格式化逻辑。"""

from app.permissions import PermissionManager


@dataclass(slots=True)
class ResolvedPath:
    """
    保存一次路径解析后的结果。

    raw_path:
        用户原始传入的路径。
    abs_path:
        经过权限校验后的绝对路径。
    display_path:
        用于展示给模型看的相对路径。
    """

    raw_path: str
    abs_path: Path
    display_path: str


@dataclass(slots=True)
class SymbolRecord:
    """
    表示一个从 Python 语法树中抽取出的符号。
    """

    kind: str
    name: str
    line: int
    column: int
    parent: str | None = None
    signature: str = ""
    doc: str = ""


def resolve_safe_path(raw_path: str, workspace_root: str) -> ResolvedPath:
    """
    解析并校验路径，确保工具只能访问当前工作目录范围内的内容。
    """

    permission_manager = PermissionManager(workspace_root)
    abs_path = permission_manager.ensure_path_access(raw_path)

    workspace_path = Path(workspace_root).resolve()
    try:
        display_path = str(abs_path.relative_to(workspace_path))
    except ValueError:
        display_path = abs_path.name

    if display_path == "":
        display_path = "."

    return ResolvedPath(
        raw_path=raw_path,
        abs_path=abs_path,
        display_path=display_path,
    )


def iter_python_files(target_path: Path) -> list[Path]:
    """
    把单文件或目录统一展开成 Python 文件列表。
    """

    if target_path.is_file():
        return [target_path] if target_path.suffix == ".py" else []

    return sorted(
        [
            file_path
            for file_path in target_path.rglob("*.py")
            if file_path.is_file()
        ],
        key=lambda item: str(item).lower(),
    )


def read_text_file(file_path: Path) -> str:
    """
    统一读取文本文件。

    优先按 utf-8 读取。
    如果文件里混入了异常字符，就用 replace 兜底，避免整次工具调用失败。
    """

    try:
        # 先用 utf-8-sig，兼容带 BOM 的 Python 文件。
        # 对普通 utf-8 文件也能正常读取，不会影响现有逻辑。
        return file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="utf-8-sig", errors="replace")


def shorten_doc(doc: str | None) -> str:
    """
    把多行文档字符串压成首行，避免输出太长。
    """

    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def build_function_signature(node: ast.AST) -> str:
    """
    为函数或方法节点构建一个简短签名。
    """

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""

    arguments: list[str] = []

    for argument in node.args.posonlyargs:
        arguments.append(argument.arg)
    if node.args.posonlyargs:
        arguments.append("/")

    for argument in node.args.args:
        arguments.append(argument.arg)

    if node.args.vararg is not None:
        arguments.append(f"*{node.args.vararg.arg}")
    elif node.args.kwonlyargs:
        arguments.append("*")

    for argument in node.args.kwonlyargs:
        arguments.append(argument.arg)

    if node.args.kwarg is not None:
        arguments.append(f"**{node.args.kwarg.arg}")

    return f"({', '.join(arguments)})"


def parse_python_symbols(file_path: Path) -> tuple[list[SymbolRecord], str | None]:
    """
    解析一个 Python 文件，并抽取主要符号。

    返回：
    - symbols: 符号列表
    - error: 解析失败时的错误描述；成功时为 None
    """

    source = read_text_file(file_path)
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as error:
        return [], f"语法解析失败: {error}"

    symbols: list[SymbolRecord] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(
                SymbolRecord(
                    kind="class",
                    name=node.name,
                    line=node.lineno,
                    column=node.col_offset,
                    doc=shorten_doc(ast.get_docstring(node)),
                )
            )

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        SymbolRecord(
                            kind="method",
                            name=child.name,
                            line=child.lineno,
                            column=child.col_offset,
                            parent=node.name,
                            signature=build_function_signature(child),
                            doc=shorten_doc(ast.get_docstring(child)),
                        )
                    )
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            symbols.append(
                                SymbolRecord(
                                    kind="class_variable",
                                    name=target.id,
                                    line=target.lineno,
                                    column=target.col_offset,
                                    parent=node.name,
                                )
                            )
                elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    symbols.append(
                        SymbolRecord(
                            kind="class_variable",
                            name=child.target.id,
                            line=child.target.lineno,
                            column=child.target.col_offset,
                            parent=node.name,
                        )
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                SymbolRecord(
                    kind="function",
                    name=node.name,
                    line=node.lineno,
                    column=node.col_offset,
                    signature=build_function_signature(node),
                    doc=shorten_doc(ast.get_docstring(node)),
                )
            )

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.append(
                        SymbolRecord(
                            kind="variable",
                            name=target.id,
                            line=target.lineno,
                            column=target.col_offset,
                        )
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.append(
                SymbolRecord(
                    kind="variable",
                    name=node.target.id,
                    line=node.target.lineno,
                    column=node.target.col_offset,
                )
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                symbols.append(
                    SymbolRecord(
                        kind="import",
                        name=alias.asname or alias.name,
                        line=node.lineno,
                        column=node.col_offset,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                symbols.append(
                    SymbolRecord(
                        kind="import_from",
                        name=alias.asname or alias.name,
                        line=node.lineno,
                        column=node.col_offset,
                    )
                )

    return symbols, None


def format_symbol_record(display_path: str, symbol: SymbolRecord) -> str:
    """
    把符号记录整理成便于阅读的单行文本。
    """

    pieces = [f"{display_path}:{symbol.line}", symbol.kind, symbol.name]

    if symbol.parent:
        pieces.append(f"(parent={symbol.parent})")
    if symbol.signature:
        pieces.append(symbol.signature)
    if symbol.doc:
        pieces.append(f"- {symbol.doc}")

    return " ".join(pieces)


def to_relative_display(file_path: Path, root_path: Path) -> str:
    """
    把文件路径转换成相对于扫描根目录的展示路径。
    """

    try:
        relative_path = str(file_path.relative_to(root_path))
    except ValueError:
        relative_path = file_path.name

    if relative_path in {"", "."}:
        return file_path.name

    return relative_path
