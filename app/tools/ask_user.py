"""ask_user 结构化提问工具，让模型主动向用户发起多选项问题。
   参考 Claude Code AskUserQuestionTool 语义实现。"""

from typing import Any

from app.agent.tooling import ToolDefinition
from app.types import ToolContext, ToolResult


def _validate(input_data: Any) -> dict[str, Any]:
    """校验 ask_user 输入：1-4 个问题，每题 2-4 个选项。"""
    if not isinstance(input_data, dict):
        raise ValueError("ask_user 输入必须是字典，包含 questions 字段。")

    questions = input_data.get("questions")
    if not isinstance(questions, list) or len(questions) < 1 or len(questions) > 4:
        raise ValueError("questions 必须是包含 1-4 个问题的列表。")

    validated_questions: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            raise ValueError(f"questions[{i}] 必须是字典。")

        question_text = q.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"questions[{i}].question 必须是非空字符串。")

        header = q.get("header")
        if not isinstance(header, str) or not header.strip():
            raise ValueError(f"questions[{i}].header 必须是非空字符串（最多 12 字符）。")

        options = q.get("options")
        if not isinstance(options, list) or len(options) < 2 or len(options) > 4:
            raise ValueError(f"questions[{i}].options 必须是包含 2-4 个选项的列表。")

        validated_options: list[dict[str, str]] = []
        for j, opt in enumerate(options):
            if not isinstance(opt, dict):
                raise ValueError(f"questions[{i}].options[{j}] 必须是字典。")

            label = opt.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"questions[{i}].options[{j}].label 必须是非空字符串。")

            desc = opt.get("description", "")
            if not isinstance(desc, str):
                raise ValueError(f"questions[{i}].options[{j}].description 必须是字符串。")

            validated_options.append({
                "label": label.strip(),
                "description": desc.strip(),
            })

        multi_select = bool(q.get("multiSelect", False))

        validated_questions.append({
            "question": question_text.strip(),
            "header": header.strip()[:12],  # 限制 header 最大长度
            "options": validated_options,
            "multiSelect": multi_select,
        })

    return {"questions": validated_questions}


def _run(validated_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """暂存问题列表，等待用户交互层填充答案。
       第一版返回问题结构，实际交互由上层 tui/app.py 处理。"""
    questions = validated_input["questions"]

    # 第一版将 questions 序列化后返回，
    # 答案由上层交互系统（AskUserQuestion 审批流）回填到 input_data.answers 中。
    output_lines = ["需要用户回答以下问题：", ""]
    for i, q in enumerate(questions, 1):
        output_lines.append(f"Q{i}: {q['question']}")
        for j, opt in enumerate(q["options"]):
            suffix = " [可多选]" if q["multiSelect"] else ""
            output_lines.append(f"  {chr(65 + j)}. {opt['label']} — {opt['description']}{suffix}")

    return ToolResult(
        ok=True,
        output="\n".join(output_lines),
        meta={
            "questions": questions,
            "awaits_user_response": True,
        },
    )


# 按 MiniCode 现有 ToolDefinition 模式注册工具
ask_user_tool = ToolDefinition(
    name="ask_user",
    description=(
        "向用户发起结构化多选项提问。"
        "当你需要让用户在多个方案中做选择时使用此工具。"
    ),
    validator=_validate,
    runner=_run,
    input_schema={
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "要提问的问题列表（1-4 题），每道题包含 question/header/options/multiSelect 字段。",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "完整的提问内容，以问号结尾。如 '我们应该使用哪个库来格式化日期？'",
                        },
                        "header": {
                            "type": "string",
                            "description": "简短标签，显示为 chip/tag（最多 12 字符）。如 '日期库'、'方案'。",
                        },
                        "options": {
                            "type": "array",
                            "description": "可选答案列表（2-4 个选项）。不包含 'Other' 选项，系统会自动补充。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "选项显示文本（1-5 词）。",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "该选项的含义或后果说明。",
                                    },
                                },
                                "required": ["label", "description"],
                            },
                        },
                        "multiSelect": {
                            "type": "boolean",
                            "description": "是否允许多选。默认 false。",
                        },
                    },
                    "required": ["question", "header", "options"],
                },
            },
        },
        "required": ["questions"],
    },
)
