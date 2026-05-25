 
from dataclasses import dataclass
from typing import Any, Callable
 

from app.types import ToolContext, ToolResult




# 工具执行函数类型
# 输入任意类型，返回检验/转换后的数据(也就是对输入的格式进行统一)，如果输入不合法则抛出异常
Validator=Callable[[Any], Any]

# 工具执行函数类型
# 接收处理后的输入和工具上下文，返回ToolResult(拿到处理后的输入格式后，工具执行函数就不需要关心输入是否合法了，只需要专注于工具的核心功能实现)
Runner=Callable[[Any, ToolContext], ToolResult]


@dataclass(slots=True)
class ToolDefinition:
    """
    表示一个工具的定义信息
    """

    name: str # 工具名称，必须唯一
    description: str # 工具描述信息，供模型参考使用
    validator: Validator # 参数校验函数
    runner: Runner # 工具执行函数
    input_schema:dict[str, Any] # 工具的参数结构，给 function call 使用


class ToolRegistry:
    """
    工具注册表：负责统一管理和执行所有工具。
    """

    def __init__(
            self,
            tools:list[ToolDefinition],
            max_output_lines:int=120, #超长输出最大保留行数
            )->None:
        # 保存工具列表，便于后续遍历展示
        self._tools=tools
        # 使用字典建立工具名到工具定义的映射，方便后续根据工具名快速找到对应的工具定义和执行函数
        self._tool_index:dict[str,ToolDefinition]={tool.name: tool for tool in tools}
        self._max_output_lines = max_output_lines  # 输出摘要阈值

    def list_tools(self)->list[ToolDefinition]:
        """
        返回当前注册的所有工具定义列表
        """
        return list(self._tools)
    
    def list_tool_name(self)->list[str]:
        """
        返回当前注册的所有工具名称列表
        """
        return list(self._tool_index.keys())
    

    def find_tool(self,name:str)->ToolDefinition|None:
        """
        根据工具名称查找对应的工具定义，如果找不到则返回 None
        """
        return self._tool_index.get(name)
    
    def _normalize_output(self, text: Any) -> str:
        """把输出统一转成字符串，并处理空输出。"""
        if text is None:
            return "工具执行完成，但没有输出。"
        s = str(text).strip()
        if not s:
            return "工具执行完成，但没有输出。"
        return s
    
    def _summarize_output(self, text: str) -> tuple[str, bool, int]:
        """按行截断超大输出，返回(文本, 是否截断, 原始行数)。"""
        lines = text.splitlines()
        total = len(lines)
        if total <= self._max_output_lines:
            return text, False, total

        kept = lines[: self._max_output_lines]
        remain = total - self._max_output_lines
        kept.append(f"[输出已截断：共 {total} 行，仅保留前 {self._max_output_lines} 行，省略 {remain} 行]")
        return "\n".join(kept), True, total

    def _normalize_result(self, tool_name: str, result: ToolResult) -> ToolResult:
        """统一返回结构：空输出兜底 + 超大输出摘要 + meta 填充。"""
        # 保证 output 永远是字符串
        output = self._normalize_output(result.output)

        # 超大输出做摘要
        summarized_output, truncated, total_lines = self._summarize_output(output)

        # 统一错误字段（失败但没填 error 时自动补）
        error = result.error
        if not result.ok and not error:
            error = f"工具 {tool_name} 执行失败"

        # 合并 meta，不覆盖已有键
        meta = dict(result.meta)
        meta.setdefault("tool_name", tool_name)
        meta.setdefault("truncated", truncated)
        meta.setdefault("total_lines", total_lines)
        meta.setdefault("max_output_lines", self._max_output_lines)

        # 返回标准化后的 ToolResult
        return ToolResult(
            ok=result.ok,
            output=summarized_output,
            error=error,
            meta=meta,
        )
    
    def execute_tool(self,tool_name:str,input_data:Any,context:ToolContext)->ToolResult:
        """
        统一执行一个工具。

        执行流程：
        1. 先根据名称找到工具
        2. 调用 validator 校验输入
        3. 调用 run 执行工具
        4. 返回 ToolResult

        如果发生问题：
        - 工具不存在：返回失败结果
        - 输入校验失败：返回失败结果
        - 工具执行异常：返回失败结果
        """

        tool=self.find_tool(tool_name)
        if not tool:
            return self._normalize_result(
                tool_name,
                ToolResult(
                ok=False,
                output=f"不存在名称为 {tool_name} 的工具",
                error="TOOL_NOT_FOUND",
                meta={"tool_name": tool_name},
                ),
            )
        try:
            # 先做输入校验与规范化
            parsed_input = tool.validator(input_data)
        except (ValueError, TypeError, KeyError) as error:
            # 输入不合法：返回失败结果，不抛异常
            return self._normalize_result(
                tool_name,
                ToolResult(
                    ok=False,
                    output=f"工具参数错误：{error}",
                    error="INVALID_INPUT",
                    meta={"tool_name": tool_name},
                ),
            )

        try:
            # 调用工具核心逻辑
            raw_result = tool.runner(parsed_input, context)
        except Exception as error:
            # 工具运行异常：兜底返回，避免主循环崩溃
            return self._normalize_result(
                tool_name,
                ToolResult(
                    ok=False,
                    output=f"工具运行异常：{error}",
                    error="TOOL_RUNTIME_ERROR",
                    meta={"tool_name": tool_name},
                ),
            )

        # runner 若返回 None（防御式兜底）
        if raw_result is None:
            raw_result = ToolResult(
                ok=False,
                output="工具未返回结果。",
                error="EMPTY_RESULT",
                meta={"tool_name": tool_name},
            )

        # runner 返回类型不正确时兜底，避免后续访问属性报错
        if not isinstance(raw_result, ToolResult):
            raw_result = ToolResult(
                ok=False,
                output=f"工具 {tool_name} 返回了非法结果类型: {type(raw_result).__name__}",
                error="INVALID_TOOL_RESULT",
                meta={"tool_name": tool_name},
            )

        # 最后统一标准化结果结构
        return self._normalize_result(tool_name, raw_result)

    
    





