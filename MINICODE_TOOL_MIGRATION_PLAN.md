# MiniCode 工具改造方案

## 目标

把 `MiniCode-ByMyself` 当前内置工具集，收敛成一套更接近 `claude-code` 核心工作流的工具集。

约束如下：

- 不处理 MCP，MCP 继续外置
- 保留 `memory`
- 不再依赖 `find_symbols / find_references / get_ast_info / file_overview` 这类分析型工具
- 工具逻辑参考 `D:\claudecode-resource\claude-code` 的真实实现语义
- 工具封装方式继续使用 MiniCode 现有的 `ToolDefinition / ToolRegistry / ToolResult / ToolContext`

## 最终工具集

推荐保留为 `8 个核心工具 + memory`：

1. `run_command`
2. `read_file`
3. `edit_file`
4. `write_file`
5. `glob_files`
6. `grep_files`
7. `ask_user`
8. `agent_dispatch`
9. `memory`

说明：

- `memory` 单独保留，不并入 8 个核心工具统计
- 如果后续强制要求“总数严格只有 8 个且包含 memory”，可把 `write_file` 合并进 `edit_file`

## 工具映射

| MiniCode 目标工具 | Claude Code 参考工具 | 处理方式 |
| --- | --- | --- |
| `run_command` | `BashTool` / `PowerShellTool` | 保留名字，升级内部逻辑 |
| `read_file` | `FileReadTool` | 保留名字，升级输入协议和读取策略 |
| `edit_file` | `FileEditTool` | 保留名字，补文件变更保护和 diff 语义 |
| `write_file` | `FileWriteTool` | 保留名字，补自动建目录、原子写入、diff |
| `glob_files` | `GlobTool` | 新增 |
| `grep_files` | `GrepTool` | 保留名字，补高级检索参数 |
| `ask_user` | `AskUserQuestionTool` | 新增 |
| `agent_dispatch` | `AgentTool` + `SendMessageTool` + `TaskStopTool` | 新增，MiniCode 内聚封装 |
| `memory` | 无需对齐 | 保留现有实现 |

## 需要移除的本地核心工具

这些工具不再作为主工具链的一部分：

- `make_dirs`
- `repo_overview`
- `codebase_map`
- `file_overview`
- `find_symbols`
- `find_references`
- `locate_symbol`
- `get_ast_info`

说明：

- `make_dirs` 的能力应并入 `write_file`
- 旧分析工具不建议再作为默认工具暴露给模型

## 当前代码与 Claude Code 逻辑的主要差距

### 1. `run_command`

当前文件：

- `app/tools/run_command.py`

当前问题：

- 只有 `subprocess.run(shell=True)` 的最小实现
- 只有统一审批，没有 Bash / PowerShell 分路
- 没有后台任务能力
- 没有复杂命令组合、重定向、危险模式的细粒度检查
- 没有长输出持久化与摘要预览

第一版应补：

- `shell=auto|bash|powershell`
- `timeout_ms`
- 针对 Windows 走 PowerShell 语义
- 保留现有审批流
- 补最基础的只读命令 / 高风险命令分类

第二版再补：

- 后台任务
- 输出持久化
- 更细的命令 AST / 组合命令风险判断

### 2. `read_file`

当前文件：

- `app/tools/read_file.py`

当前问题：

- 现在是 `path + char-based offset/limit`
- Claude Code 更偏 `file_path + line-based offset/limit`
- 没有特殊文件类型分支处理
- 没有更完整的相似路径建议和设备文件阻断

第一版应补：

- 新入参主字段改为 `file_path`，兼容旧 `path`
- 保留现有分段读取，但向行级协议靠拢
- 保留路径权限判断
- 补更稳的路径建议

第二版再补：

- PDF / 图片 / notebook / 二进制文件处理
- token 预算更细化

### 3. `edit_file`

当前文件：

- `app/tools/edit_file.py`

当前问题：

- 只是简单文本替换
- 没有 stale-check
- 没有 diff 输出
- 没有编码 / 换行风格保留

第一版应补：

- 输入协议向 `old_string / new_string` 靠拢，兼容旧字段
- 编辑前检查文件是否在 read 之后被修改
- 返回最小 diff 摘要

第二版再补：

- 更完整 patch 输出
- 编码 / 换行风格保留

### 4. `write_file`

当前文件：

- `app/tools/write_file.py`

当前问题：

- 父目录不存在就失败
- 没有原子写入
- 没有 stale-check
- 没有 diff / patch 输出

第一版应补：

- 自动创建父目录
- 入参改为 `file_path`，兼容旧 `path`
- 覆盖前做最小 stale-check
- 返回 create / update 语义

第二版再补：

- 原子写入
- 更完整 diff / git diff 输出

### 5. `glob_files`

当前状态：

- 不存在真正等价实现
- 现有 `list_files` 不是 `GlobTool`

第一版应新增：

- 支持 `pattern`
- 支持可选 `path`
- 返回匹配文件列表
- 返回 `truncated`
- 相对路径输出

### 6. `grep_files`

当前文件：

- `app/tools/grep_files.py`

当前问题：

- 当前实现偏 fixed-string
- 只支持目录，不支持单文件
- 缺少 `glob`、`output_mode`、`head_limit`、`offset`、`context`、`type`、`-i`

第一版应补：

- 支持单文件和目录
- 支持 `glob`
- 支持 `output_mode=content|files_with_matches|count`
- 支持 `head_limit + offset`

第二版再补：

- `-A/-B/-C/context`
- `type`
- `multiline`
- 更完整 regex 语义

### 7. `ask_user`

当前状态：

- 不存在
- 现有 `ApprovalRequest` 只能做授权，不等于结构化提问

第一版应新增：

- 模型主动提问
- 多选项问题
- 单题或多题
- 返回结构化答案

第二版再补：

- 预览内容
- 更复杂的 UI 展示

### 8. `agent_dispatch`

当前状态：

- 不存在 Claude Code 对应的子代理工具链

第一版应新增：

- `spawn`
- `result`

说明：

- 第一版可以先做同步子任务调度
- 先不强求完整的并发消息总线

第二版再补：

- `send`
- `stop`
- 子代理状态管理
- 后台任务和消息队列

## 第一版必须修改的文件

### 工具注册与工具实现

- `app/tools/__init__.py`
- `app/tools/run_command.py`
- `app/tools/read_file.py`
- `app/tools/edit_file.py`
- `app/tools/write_file.py`
- `app/tools/grep_files.py`
- `app/tools/glob_files.py` 新增
- `app/tools/ask_user.py` 新增
- `app/tools/agent_dispatch.py` 新增

### agent 侧耦合清理

- `app/agent/prompt.py`
- `app/agent/loop.py`
- `app/agent/analysis_guard.py`
- `app/context/runtime.py`
- `app/agent/tooling.py`

### 可能需要补的交互层

- `app/tui/app.py`
- `app/tui/widgets/input_area.py`
- `app/types.py`

说明：

- `ask_user` 需要新的交互事件，不应复用高风险审批事件
- `agent_dispatch` 如果带异步状态，可能要补状态展示

## 第一版实施顺序

1. 先替换 `app/tools/__init__.py` 的核心工具注册
2. 新增 `glob_files`
3. 升级 `grep_files`
4. 升级 `read_file`
5. 升级 `write_file`
6. 升级 `edit_file`
7. 升级 `run_command`
8. 新增 `ask_user`
9. 新增 `agent_dispatch`
10. 清理 `prompt / loop / analysis_guard / runtime` 中对旧分析工具的耦合
11. 补测试

## 第二版增强范围

第二版不是另一个方案，而是第一版跑通之后的增强迭代，目标是更接近 Claude Code 的真实实现细节。

主要包含：

- `run_command` 后台任务、长输出持久化、复杂命令安全分析
- `read_file` 多类型文件处理
- `edit_file / write_file` 更完整 diff、原子写、换行风格保留
- `grep_files` 更完整高级参数
- `agent_dispatch` 的 `send / stop / async lifecycle`

## 兼容性策略

- 第一版对旧字段做兼容，避免 prompt 和旧调用立刻全部失效
- 新字段优先采用 Claude Code 风格命名
- `memory` 保持原状
- MCP 不改
- 审批流继续保留，但只负责授权，不负责结构化提问

## Git 说明

当前仓库分支信息需要单独确认。

如果目标仓库默认分支仍然是 `master`，后续建分支应基于 `master`，而不是假设为 `main`。
