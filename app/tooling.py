 
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
    validator: Validator # 输入检验/转换函数
    runner: Runner # 工具执行函数


class ToolRegistry:
    """
    工具注册表：负责统一管理和执行所有工具。
    """

    def __init__(self,tools:list[ToolDefinition])->None:
        # 保存工具列表，便于后续遍历展示
        self._tools=tools
        # 使用字典建立工具名到工具定义的映射，方便后续根据工具名快速找到对应的工具定义和执行函数
        self._tool_index:dict[str,ToolDefinition]={tool.name: tool for tool in tools}

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
            return ToolResult(
                ok=False,
                output=f"不存在名称为 {tool_name} 的工具",
            )
        try:
            # 先检验或规范化输入
            parsed_input=tool.validator(input_data)

            # 再执行工具核心功能
            result=tool.runner(parsed_input,context)

            # 防止工具返回None输出，保证output始终是字符串，避免后续处理出现类型问题
            if result.output is None:
                result.output=""
            
            return result
        except (ValueError,TypeError,KeyError) as error:
            # 输入格式错误或缺少必要字段等问题，都会抛出这些类型的异常，我们捕获后返回一个失败的 ToolResult
            return ToolResult(
                ok=False,
                output=f"工具{tool_name}执行失败，原因：{str(error)}",
            )
        except Exception as error:
            # 捕获其他未知异常，避免工具执行时崩溃整个系统
            return ToolResult(
                ok=False,
                output=f"工具{tool_name}执行过程中发生错误，原因：{str(error)}",
            )
        

    
    





