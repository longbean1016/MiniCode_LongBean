# CodingAgent 项目学习文档

> 面试准备用 — 理解项目链路、设计价值与不足

---

## 一、项目概述

### 1.1 这是什么项目

一个**面向本地代码分析与开发任务协同**的 AI Agent 系统。核心思路是：用户用自然语言描述代码任务（分析调用链路、搜索符号、修改代码等），Agent 通过 **Query Loop + Tool Use** 机制自主完成"理解意图 → 调用工具 → 分析结果 → 再决策"的闭环，直到产出最终答案。

### 1.2 一句话定位

> 用 OpenAI API 驱动一个能自主操作本地代码仓库的 Agent，通过工具系统、记忆系统和上下文压缩治理来保证多轮复杂任务下的**任务连续性、上下文稳定性和执行安全性**。

### 1.3 整体数据流

```
用户输入
  │
  ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  TUI 交互层   │───▶│  Agent 主循环    │───▶│ 模型适配层    │
│ (app/tui)    │    │ (app/agent/loop) │    │ (infra/model) │
└──────────────┘    └───────┬─────────┘    └──────┬───────┘
                            │                      │
                            │ ① 准备上下文          │ OpenAI API
              ┌─────────────┼──────────────┐       │
              ▼             ▼              ▼       │
      ┌──────────┐  ┌────────────┐  ┌──────────┐   │
      │ 上下文压缩 │  │ 记忆系统    │  │ 工具系统  │   │
      │ (context)│  │ (memory)   │  │ (tools)  │   │
      └──────────┘  └────────────┘  └────┬─────┘   │
              │             │             │         │
              ▼             ▼             ▼         │
         Snip/Micro/   短期WM/长期   ToolRegistry  │
         Full Compact  JSON+Qdrant   本地+MCP工具  │
                                           │       │
                                           ▼       │
                                    ┌────────────┐  │
                                    │ 权限 & 安全  │  │
                                    │ (permission)│  │
                                    └────────────┘  │
```

---

## 二、技术栈一览

| 技术 | 在项目中的作用 | 对应代码位置 |
|------|--------------|-------------|
| **Python** | 全项目语言，面向对象 + Protocol + dataclass | `app/` |
| **OpenAI API** | 驱动 Agent 推理的工具调用和文本生成 | `app/infra/model_registry.py` |
| **MCP (Model Context Protocol)** | 外部工具服务接入协议，支持子进程和 HTTP 双模 | `app/mcp/` |
| **Tool Calling** | Agent 通过 `function calling` 调用本地代码工具 | `app/tools/`, `app/agent/tooling.py` |
| **JSON-RPC** | MCP 客户端与服务端之间的通信协议 | `app/mcp/client.py` |
| **Qdrant** | 长期记忆语义向量检索 | `app/memory/vector_index.py` |
| **Memory System** | 短期工作记忆 + 长期 JSON + 向量检索 | `app/memory/`, `app/state/` |
| **Docker** | 容器化部署环境 | Dockerfile（项目根目录） |

---

## 三、核心架构：Query Loop + Tool Use 闭环

### 3.1 链路

```
每轮循环 step_index++
    │
    ├─ ① 准备上下文 (prepare_agent_context)
    │     ├─ 滑动窗口切分历史 (select_history_window)
    │     ├─ 三层压缩治理 (Snip → Microcompact → 必要时 Full)
    │     ├─ 长期记忆检索注入 (memory_pipeline.build_prompt_context)
    │     ├─ 工作记忆注入 (WorkingMemory.format_for_prompt)
    │     └─ 组装最终 system + messages
    │
    ├─ ② 调用模型 (model.next / model.stream_chat)
    │     ├─ 协议翻译 (ChatMessage → OpenAI format)
    │     ├─ 携带 tools 定义
    │     └─ 返回: assistant 文本 / tool_calls / approval
    │
    ├─ ③ 处理模型返回
    │     ├─ type="assistant" + kind="final" → 结束，返回答案
    │     ├─ type="assistant" + kind="progress" → 注入 nudge 继续
    │     ├─ type="tool_calls" → 逐个执行工具
    │     └─ type="approval" → 挂起，等用户确认
    │
    └─ ④ 后处理
          ├─ 工具结果写回 builder (tool_result 消息)
          ├─ 更新工作记忆 (paths, decisions, failures)
          ├─ 分析证据沉淀 (analysis_tracker)
          └─ 回到 ① 继续下一轮
```

### 3.2 关键设计

- **步数上限 `max_steps=20`**：防止模型在工具调用中来回打转
- **Nudge 机制**：主循环会在必要时注入临时 user 引导（如 "你已经拿到了工具结果，请直接给答案"），这些引导**不写回历史**，只影响当前一步的模型请求
- **分析护栏 `analysis_tracker`**：对代码分析类任务，专门维护一份"证据账本"——记录哪些函数名是实际观察到的、哪些文件已读过，最终回答前做事实校验，防止模型脑补

### 3.3 对项目的作用

- 让 Agent 能像人类开发者一样"边看边想边做"，而不是一次性回答
- 工具调用失败后能自动感知错误上下文、调整下一步策略
- 流式输出 (`stream_agent`) 让用户实时看到思考过程和工具执行进度

### 3.4 不足与改进

- 步数上限是硬编码的 20，没有根据任务复杂度动态调整
- Nudge 机制目前是纯规则触发，复杂场景下可能需要模型自身判断"我是否该继续探索"
- `analysis_tracker` 只覆盖代码分析类任务，对重构/生成类任务没有类似护栏

---

## 四、工具封装

### 4.1 链路

```
ToolDefinition（数据类）
    ├─ name, description          ← 模型的 function calling schema 来源
    ├─ validator: Callable        ← 输入校验（如路径合法性、参数类型）
    ├─ runner: Callable           ← 实际执行（文件读写、AST 解析、命令执行等）
    └─ input_schema: dict         ← JSON Schema，传给模型做参数识别

ToolRegistry（注册 & 执行）
    │
    ├─ register_tool() / unregister_tool()  ← 支持运行时热插拔（MCP 用）
    │
    └─ execute_tool(tool_name, input, context)
          │
          ├─ ① find_tool()         → TOOL_NOT_FOUND
          ├─ ② validator(input)    → INVALID_INPUT
          ├─ ③ runner(input, ctx)  → TOOL_RUNTIME_ERROR / ToolResult
          └─ ④ _normalize_result() → 统一的 ToolResult出口
                │
                ├─ _smart_truncate_output()  ← 主输出截断（给模型的）
                └─ _build_context_output()   ← 上下文输出（写历史的，更短）
```

### 4.2 按工具类型差异化截断策略

```python
# 每种工具配置了独立的输出上限
_TOOL_OUTPUT_LIMITS = {
    "read_file": 40_000,      # 代码文件最吃上下文
    "run_command": 30_000,    # 命令输出可能很长
    "grep_files": 16_000,
    "list_files": 12_000,
    "file_overview": 10_000,
    "find_references": 8_000,
}
# 写历史的 context_output 更激进（6,000 ~ 2,000）
```

截断策略也按工具类型不同：
- **read_file**：保留头部 60% + 尾部，因为文件头和尾往往信息密度最高
- **grep_files / find_references / list_files**：保统计头 + 正文首尾样本
- **run_command**：额外提取 error/warning 行，优先保留在截断结果中

### 4.3 内置工具清单（14 个）

| 工具 | 作用 | 属于简历中的哪类 |
|------|------|----------------|
| `read_file` | 分段读取文件 | 内容搜索 |
| `grep_files` | 正则搜索文件内容 | 内容搜索 |
| `list_files` | 列出目录结构 | 内容搜索 |
| `find_symbols` | 搜索符号定义 | AST 解析 |
| `locate_symbol` | 定位符号位置 | AST 解析 |
| `find_references` | 查找符号引用 | AST 解析 |
| `get_ast_info` | 获取 AST 结构信息 | AST 解析 |
| `file_overview` | 文件概览 | AST 解析 |
| `codebase_map` | 代码库结构地图 | AST 解析 |
| `repo_overview` | 仓库概览 | 内容搜索 |
| `write_file` | 写入文件 | 代码编辑 |
| `edit_file` | 精确编辑文件 | 代码编辑 |
| `make_dirs` | 创建目录 | 代码编辑 |
| `run_command` | 执行 shell 命令 | 命令执行 |

### 4.4 对项目的作用

- 上层 Agent 主循环完全不感知单个工具的实现细节——统一走 `execute_tool()`，拿到统一定义的 `ToolResult`
- 工具输出截断在 Registry 层面统一做，不会出现"某个工具忘截断导致上下文爆掉"的问题
- 建立了**双通道输出**：主通道给模型推理用（宽松），context 通道写历史用（紧凑），这为后面的上下文压缩打下了基础

### 4.5 不足与改进

- `validator` 目前只是一个 Callable，校验逻辑完全由各工具自己实现，缺少统一的 Schema 校验框架（如 Pydantic）
- 截断策略是基于字符数的规则截断，不是基于语义的理解截断——可能把代码中间关键逻辑截掉
- `_build_context_output()` 和 `_smart_truncate_output()` 逻辑有大量重复，可以合并抽象
- 工具错误码（TOOL_NOT_FOUND / INVALID_INPUT 等）是字符串枚举，不是结构化类型，后续扩展容易拼写错误

---

## 五、MCP 协议扩展

### 5.1 链路

```
用户 /mcp add <name> -- <command> [args...]
    │
    ▼
McpManager.handle_command()
    │
    ├─ Stdio 模式: 解析 command + args
    │     └─ StdioMcpClient.start()
    │           ├─ 安全校验 (_validate_mcp_command, _validate_mcp_args)
    │           ├─ 子进程启动 (subprocess.Popen)
    │           ├─ 协议协商: content-length 优先 → newline-json 回退
    │           ├─ JSON-RPC initialize 握手 (协议版本 2024-11-05)
    │           └─ notifications/initialized 通知
    │
    ├─ HTTP 模式: /mcp add <name> --url <url>
    │     └─ HttpMcpClient.start()
    │           └─ HTTP POST + JSON-RPC 通信
    │
    └─ 工具发现
          ├─ client.list_tools() → tools/list 请求
          ├─ 为每个远程工具生成 ToolDefinition
          │     name: mcp__<server>__<tool>
          │     validator: 透传（远端校验）
          │     runner: → client.call_tool(tool_name, arguments)
          └─ tool_registry.register_tool() → 动态注册
```

### 5.2 JSON-RPC 通信细节

```
┌─────────────┐        stdin/stdout         ┌─────────────┐
│  MiniCode   │ ◄═══════════════════════════▶│  MCP Server │
│ (Client)    │   JSON-RPC 2.0 over pipe    │  (外部进程)  │
└─────────────┘                              └─────────────┘

请求: {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
响应: {"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}
通知: {"jsonrpc":"2.0","method":"notifications/initialized","params":{}}

两种传输协议自动协商:
  - content-length: HTTP 风格头部 + body（先尝试）
  - newline-json:   每行一个 JSON 消息（回退方案）
```

### 5.3 安全校验

- **命令白名单**：只有 node/npm/python/uv/go/cargo 等已知安全命令允许作为 MCP command
- **系统 Shell 禁止**：cmd.exe / powershell.exe / pwsh.exe 直接拒绝
- **路径遍历拦截**：command 中不允许 `..` 和 `~`
- **Shell 元字符检查**：args 中不允许 `|&;$(){}<>\n\r` 等危险字符
- **Payload 上限**：50MB 上限防止恶意 MCP Server 制造 OOM

### 5.4 对项目的作用

- Agent 的工具能力从"固定 14 个内置工具"扩展为"可运行时接入任意 MCP 生态工具"
- 工具注册对主循环完全透明——MCP 工具和本地工具一样走 `ToolRegistry.execute_tool()`
- 支持运行时热插拔：`/mcp add` 和 `/mcp remove` 不需要重启 Agent

### 5.5 不足与改进

- 启动阶段 MCP Server 连接失败只是打印日志，不阻塞进程，但也没有自动重连机制
- MCP 工具的 `validator` 是透传（`lambda value: value`），依赖远端校验，本地无法提前拦截明显非法参数
- `list_tools()` 的结果缓存是进程生命期内永久的——如果 MCP Server 后端新增了工具，需要重启 Agent 才能发现
- 缺少 MCP Server 健康检查：如果子进程意外退出，下次 `call_tool` 才会发现并报错

---

## 六、分层记忆系统

### 6.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                   MemoryPipeline                    │
│                   （总编排层）                        │
│                                                     │
│  ┌──────────────────┐   ┌──────────────────────┐    │
│  │  MemoryReadPipeline│  │ MemoryWritePipeline    │    │
│  │  （读链路）         │  │ （写链路）              │    │
│  │                    │  │                        │    │
│  │  ① 构建检索 query  │  │  ① 显式记忆处理         │    │
│  │  ② 固定记忆优先    │  │  ② 回合反思触发         │    │
│  │  ③ 向量/词面检索   │  │  ③ 抽取候选记忆         │    │
│  │  ④ Rerank + 去重  │  │  ④ Guard 护栏过滤      │    │
│  │  ⑤ 注入 prompt    │  │  ⑤ Verifier 去重验证    │    │
│  │  ⑥ 反馈闭环       │  │  ⑥ Curator 增量整理     │    │
│  └──────────────────┘   │  ⑦ Decay 衰减归档       │    │
│                          └──────────────────────┘    │
│                                                     │
│  ┌──────────────────┐   ┌──────────────────────┐    │
│  │  WorkingMemory    │   │  MemoryStore          │    │
│  │  （短期/运行时）    │   │  （长期/持久化）        │    │
│  │                    │   │                        │    │
│  │  TTL 滑动窗口      │   │  JSON (权威源)          │    │
│  │  类型限额 + Token  │   │  + Qdrant (语义索引)    │    │
│  │  + 话题切换清理    │   │  + USER.md (分级注入)   │    │
│  └──────────────────┘   └──────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### 6.2 短期记忆：WorkingMemory

**链路：**

```
工具调用发生
  │
  ├─ 文件路径 → WorkingMemory "active_task" (TTL=1800s, importance=0.8)
  ├─ 工具失败 → WorkingMemory "error_context" (TTL=1800s, importance=0.9)
  └─ 每轮结束 → WorkingMemory "key_decision" / "reflection_decision"

模型回复
  │
  └─ extract_decisions_from_assistant() → key_decision

话题切换
  └─ _is_topic_shift() → 清理 active_task/key_decision/risk
     保留 user_preference/project_constraint
```

**核心约束：**
- 总条目上限 15 条，总 token 上限 420
- 每种 entry_type 独立限额（如 user_preference 最多 2 条）
- 超限时按 `importance` + `created_at` 淘汰低价值条目
- 带 TTL 过期自动清理

### 6.3 长期记忆：MemoryStore + Qdrant

**写入链路（MemoryWritePipeline）：**

```
一轮任务完成
  │
  ├─ ① 判断是否值得反思 (_should_attempt_reflection)
  │     异常收尾 / progress / 空回复 → 跳过
  │
  ├─ ② 收集本轮证据
  │     key_decisions（模型做出的关键决策）
  │     failures（工具执行失败摘要）
  │     files_touched（涉及的项目文件列表）
  │
  ├─ ③ reflection_policy 判断是否需要反思
  │     规则过滤 + 模型判断双保险
  │
  ├─ ④ LongTermMemoryExtractor 抽取候选记忆
  │     调用模型把本轮经验提炼为结构化 MemoryEntry
  │
  ├─ ⑤ MemoryWriteGuard 护栏过滤
  │     检查内容是否为空、是否过于临时、是否命中拒存规则
  │
  ├─ ⑥ MemoryVerifier 去重验证
  │     ├─ 语义召回相似旧记忆 (Qdrant → 词面回退)
  │     ├─ 调用 verifier 模型判定: store/supersede_store/duplicate/conflict/reject
  │     └─ 模型失败时退回本地规则兜底
  │
  ├─ ⑦ MemoryCurator 增量整理
  │     新记忆写入后触发增量 curator
  │     达到阈值时触发全量 curator（清理重复、合并相似、归档旧版本）
  │
  └─ ⑧ MemoryDecay 衰减
       新的写入/访问会刷新 decay_score
       长时间不访问的记忆分数逐渐降低
       低于阈值 + 足够陈旧 → 自动归档
```

**读取链路（MemoryReadPipeline）：**

```
每轮上下文准备
  │
  ├─ ① 构建检索 query（用户输入 + 会话摘要 + WM 活跃线索）
  │
  ├─ ② 固定记忆优先 (pinned entries — 用户显式固定的长期记忆)
  │
  ├─ ③ 向量语义搜索 (Qdrant search → 词面回退)
  │
  ├─ ④ Rerank 排序 (overlap + tag + domain + confidence + decay + recency)
  │
  ├─ ⑤ 注入预算分配 (top_k=4, 固定记忆 > 用户记忆 ≥ 项目记忆)
  │
  ├─ ⑥ 去重 (检查已覆盖的 active context 和 working memory，避免重复注入)
  │
  └─ ⑦ 格式化注入 prompt
        ## 固定用户长期记忆
        ## 相关用户长期记忆
        ## 项目长期记忆
        ## 当前工作记忆
```

### 6.4 反馈闭环

```
注入的记忆 ID 被追踪 → 回合结束时判断任务成功/失败
    │
    ├─ 成功 → injected memory 的 usage_count += 2
    └─ 失败 → injected memory 的 usage_count -= 1

长期来看:
  - 高 usage_count → 更可能被召回和注入
  - 低 usage_count → 逐步被 decay 归档
```

### 6.5 对项目的作用

- 短期记忆让 Agent 在同一个会话内"记得刚才在干什么"，不会每轮都重新探索相同路径
- 长期记忆让 Agent 跨会话"回忆起用户偏好和项目约定"，比如"上次你说这个项目用 Poetry 管理依赖"
- 写链路的抽取-校验-去重-衰减四层治理，避免了"记忆库越来越脏、注入内容越来越杂"
- 反馈闭环把"有没有注入"和"有没有帮上忙"关联起来，不帮上忙的记忆逐步降权

### 6.6 不足与改进

- **写链路触发时机过晚**：只在回合完全结束时才做反思抽取，过程中产生的关键发现（如 "找到了根因"）可能被后续轮次冲淡
- **依赖模型做抽取和验证**：extractor 和 verifier 都靠 LLM，模型不稳定时记忆质量会明显下降，虽然有本地规则兜底，但兜底规则比较粗糙（Jaccard + 关键词）
- **短记忆和长记忆的边界模糊**：WorkingMemory 中的 key_decision 和长期记忆可能记录同一件事，但两个体系不做互认去重
- **向量检索只在写入和检索时使用**：记忆更新（如 usage_count 变化）需要手动 sync，没有自动化的增量索引更新
- **缺少记忆版本管理的可视化**：哪些记忆被替代了、为什么被替代，目前只能在 JSON 里看 `supersedes_memory_id` 字段

---

## 七、上下文压缩治理

### 7.1 三层递进式压缩架构

```
准备上下文时：每轮都过这一套 pipeline
    │
    ├─ 第一层：Snip（纯规则裁剪——轻量，安全，零模型成本）
    │
    │   ① 删除 assistant_progress 消息（中间进度说明）
    │   ② read_file 去重：同路径 + 同内容 hash → 替换为占位说明
    │   ③ list/grep 去重：按头部统计 + 结果集签名去重
    │   ④ 过滤纯确认类 tool_result（"文件写入成功"无额外事实）
    │   ⑤ 合并连续重复 assistant 回复（恢复后重复答复）
    │   ⑥ 超预算时删低优先级旧消息
    │
    │   关键特征：不删 tool_call / tool_result 协议对，
    │   不调模型，不做语义理解，只做精确匹配裁剪。
    │
    ├─ 第二层：Microcompact（清理旧 tool_result——保留协议结构）
    │
    │   触发条件：有工具调用的轮数 > 保留阈值（3）
    │   节流机制：两次 microcompact 之间至少间隔 1 小时
    │
    │   处理逻辑：
    │   ① 保护最近 3 轮的全部 tool_result（最近证据不能丢）
    │   ② 将较旧的 tool_result 正文替换为占位说明：
    │      "[旧 tool_result 内容已由 microcompact 清理]
    │       工具: read_file
    │       路径: src/main.py"
    │   ③ 可选：调用轻量模型抽取被清理内容的关键语义
    │      - tool_findings → 回写到 WorkingMemory
    │      - open_issues    → 回写到 WorkingMemory
    │      - key_decisions  → 回写到 WorkingMemory
    │
    │   关键特征：不删消息，不破坏配对结构，
    │   只清正文，保留后续恢复推理的可能性。
    │
    └─ 第三层：Full Compact（达到阈值时触发——折叠旧对话为摘要）
         │
         触发条件：上下文使用率 ≥ 92% (AUTO_COMPACT_TRIGGER_RATIO)
         │
         策略选择：
         ├─ session compact（激进清理但不到底）
         │    目标：使用率降到 58%
         │    方法：折叠部分旧对话 → 结构化摘要
         │
         └─ full compact（折叠到压缩基线）
              目标：使用率降到 35%
              方法：全部旧对话合并为 active context 摘要
              │
              ├─ 规则摘要（build_older_history_summary）
              │   兜底：基于 user/assistant/tool 统计生成
              │
              └─ 模型摘要（OlderHistorySummarizer）
                   优先：调模型生成结构化摘要
                   格式：## 当前任务 / 关键决策 / 未解决问题 /
                        关键工具发现 / 项目约束
```

### 7.2 上下文状态持久化

为了让压缩在跨轮次间具备连续性：

```
ContextStateData（持久化到磁盘）
    ├─ source_message_count（上次完整历史的长度）
    ├─ source_history_fingerprint（历史指纹，用于判断是否可增量恢复）
    ├─ compacted_messages（压缩后的消息列表）
    ├─ active_context_summary（当前压缩基线摘要）
    ├─ compaction_history（最近 10 次压缩记录）
    └─ last_microcompact_at（最近一次 microcompact 时间）

跨轮次复用逻辑：
    如果当前 full_history 的前 N 条指纹与上次一致
    → 直接复用上次 compacted_messages + 增量追加新消息
    → 避免每轮重复压缩同一批旧历史
```

### 7.3 对项目的作用

- **Snip**：零成本清理，处理了占上下文 30%-40% 的"纯冗余"（重复读文件、空确认信息、连续重复回复）
- **Microcompact**：释放了旧 tool_result 占用的空间，但保留了配对结构（tool_call + tool_result 不能单方面删除）和语义承接，让模型在压缩后仍能理解之前的工具执行链路
- **Full Compact**：防止长对话必然导致的上下文爆窗，折叠为结构化摘要后保留跨轮次核心信息

### 7.4 不足与改进

- **三层是顺序耦合的**：Snip → Microcompact → Full 是固定的。如果 Snip 已经释放了大量空间，Microcompact 可能白跑一套（但因为节流机制不会频繁触发）
- **Microcompact 的语义承接依赖模型**：如果轻量模型抽取失败（熔断 / 超时），被清掉的关键事实就永久丢了
- **Full Compact 的模型摘要偶尔会丢失精确信息**：比如 "read_file 第 42 行的报错" 可能被压缩为 "某处有报错"，后续轮次无法精确定位
- **压缩阈值是全局常量**：92% 触发、78%/58%/35% 目标，没有根据任务类型（分析 vs 生成 vs 重构）做差异化配置
- **缺少压缩效果的可视化反馈**：不知道哪次压缩清掉了什么，排查"为什么 Agent 忘了这件事"比较困难

---

## 八、权限与安全控制

### 8.1 链路

```
工具执行前
  │
  ├─ 文件操作
  │     └─ PermissionManager.ensure_path_access(path)
  │           ├─ 相对路径 → 拼接 workspace_root → 绝对路径
  │           ├─ 判断是否在 workspace_root 子树内
  │           ├─ 在范围内 → 放行，返回绝对路径
  │           └─ 越界 → 抛出 PermissionError，直接拒绝
  │
  ├─ 命令执行 (run_command)
  │     └─ PermissionManager.check_command_permission(command)
  │           │
  │           ├─ ① 检查 approved_actions（会话级已批准集合）
  │           │     命中 → 直接 allow（授权复用）
  │           │
  │           ├─ ② 检查 deny_patterns（绝对禁止）
  │           │     format / shutdown / reboot / mkfs / dd /
  │           │     poweroff / halt / chown / chmod 777
  │           │     命中 → deny，不给授权机会
  │           │
  │           ├─ ③ 检查 ask_patterns（高风险需确认）
  │           │     rm / del / rmdir
  │           │     命中 → ask，提交 ApprovalRequest
  │           │
  │           └─ ④ 默认 → allow
  │
  └─ 输出处理
        └─ PermissionManager.truncate_output(text)
              超过 max_output_chars → 截断 + 附加截断说明
```

### 8.2 审批流程

```
主循环收到 tool_calls
  │
  ├─ tool_registry.execute_tool()
  │     │
  │     ├─ 正常执行 → ToolResult(ok=True)
  │     │
  │     └─ 命中审批 → ToolResult(ok=False, error="PERMISSION_REQUIRED")
  │
  ├─ 主循环检测到 PERMISSION_REQUIRED
  │     ├─ 先写入占位 tool_result（保证消息协议完整）
  │     ├─ 构建 ApprovalRequest（tool_name, command, action_key, message）
  │     └─ 返回 AgentStep(type="approval", approval=...)
  │
  ├─ TUI 展示审批对话框
  │     └─ "该操作需要用户授权。工具: run_command, 命令: rm -rf ..."
  │
  └─ 用户确认后
        ├─ action_key 加入 ToolContext.approved_actions
        ├─ 同一条命令后续自动放行（会话级复用）
        └─ 继续执行主循环
```

### 8.3 命令超时控制

- `command_timeout_seconds` 默认 15 秒
- 通过 `subprocess.run(timeout=...)` 实现
- 超时后进程被 kill，返回命令执行失败

### 8.4 对项目的作用

- **路径沙箱**将 Agent 的文件操作彻底锁在项目工作区，即使模型幻觉产出了 `/etc/passwd` 这样的路径也会被拦截
- **三级权限体系**（allow/ask/deny）让危险命令不会"静默执行"——必须用户亲自确认
- **授权复用**让用户确认一次后，同一条命令在当次会话内不再重复弹出确认框
- **MCP 侧也做了安全校验**：命令白名单、Shell 元字符拦截、路径遍历拦截、Payload 上限

### 8.5 不足与改进

- 权限策略只区分了 deny（不可覆盖）和 ask（可授权），缺少"允许但记录审计日志"模式
- 路径沙箱只做了 `resolve().relative_to()` 检查，未防范符号链接绕过
- 命令危险检测用正则——`rm` 匹配了 `rm`，也匹配了 `xterm`（不过目前 `rm` 是 ask 而非 deny，影响较小）
- 命令超时时长是全局统一的 15 秒，无法根据命令类型（编译 vs 简单 ls）差异化配置
- MCP 侧的安全校验只作用于 command/args，不检查 MCP Server 返回的内容是否安全

---

## 九、可靠性机制（串联其他模块）

### 9.1 重试机制 (RetryPolicy)

```
主模型 / 辅助模型 / 向量操作 → 共享同一套重试框架

RetryPolicy:
    max_attempts=3
    base_delay_seconds=0.8
    backoff_multiplier=2.0     → 实际等待: 0.8s → 1.6s → 3.2s
    max_delay_seconds=4.0

should_retry_model_error():
    网络超时、服务端 5xx → 重试
    认证失败、权限不足 → 不重试（重试也没用）
```

### 9.2 熔断机制 (CircuitBreaker)

```
三个独立熔断器：
    ├─ "chat_model"          ← 主模型调用
    ├─ "history_summarizer"  ← 历史摘要 / Full Compact 模型调用
    ├─ "memory_verifier"     ← 记忆验证模型调用
    ├─ "microcompact_extractor"  ← Microcompact 语义抽取
    └─ "assistant_reply_extractor" ← Assistant 回复记忆抽取

逻辑：
    连续失败 N 次 → 熔断打开 → 后续请求直接拒绝（不等待超时）
    冷却 T 秒后 → 半开 → 允许一次试探 → 成功则恢复，失败继续熔断
```

为什么分拆熔断器：Microcompact 抽取失败不应该连带阻断历史摘要，独立熔断让"每个模型用途的可用性"各自由各自的电路保护。

### 9.3 上下文溢出恢复

```
模型调用抛异常
  │
  ├─ is_context_overflow_error() 判断是否上下文过长
  │
  └─ recover_from_context_overflow()
        ├─ 策略 1: 激进裁剪 tool_result
        ├─ 策略 2: 强制触发 Full Compact
        └─ 用压缩后的 messages 重试模型调用
```

---

## 十、核心设计决策 & 面试可能追问

### Q1: 为什么选择 Tool Calling 而不是固定流程？

固定流程无法处理代码场景的多样性——用户可能要求分析链路、修改代码、搜索符号、执行命令，每种任务的工具调用顺序和组合都不一样。Tool Calling 让模型自己决策"下一步该调什么、用什么参数"，Agent 主循环只负责"调模型 → 执行工具 → 喂结果 → 再调模型"的执行框架。

### Q2: 记忆系统的核心难点在哪？

不在"存"，而在**"什么时候该存、存了之后什么时候该用、用了之后怎么知道有没有用"**。所以设计了抽取-校验-去重-衰减-反馈五段流水线：用模型判断值不值得存（extractor），用规则+模型去重（verifier），用使用频次 + 时间做衰减（decay），用反馈调整权重（feedback）。

### Q3: 三层压缩看起来设计很重，为什么不直接调模型做摘要？

纯模型摘要有两个问题：一是贵（每次摘要都等于一次 API 调用），二是不稳定（模型有时候会改写关键信息）。Snip 处理了 30-40% 的纯规则可裁冗余（零成本），Microcompact 在保留协议结构的前提下清理旧结果（只可选调模型承接语义），只有最后真的逼近窗口上限时才做 Full Compact。这是**成本递增、破坏性递增**的递进设计。

### Q4: 如果有 200 轮对话，上下文怎么撑住？

200 轮对话中，只有最近 6-8 轮保留原始消息，更早的轮次通过以下机制处理：
- 历史摘要（OlderHistorySummarizer）把旧轮次压成结构化基线
- WorkingMemory 保留了跨轮次的活跃线索（当前任务、关键决策、风险）
- 长期记忆持续沉淀项目级知识
- 压缩状态持久化让跨会话恢复不需要重压

### Q5: 安全性方面最可能出问题的点？

MCP 外部工具——因为 MCP Server 是外部进程，虽然对 command/args 做了安全校验，但无法完全控制 MCP Server 的行为。比如 MCP Server 的 tools/call 返回结果中包含恶意指令或者诱导信息，目前没有对内容做安全审查。

---

## 附录：关键文件索引

| 模块 | 文件 | 职责 |
|------|------|------|
| 入口 | `app/main.py` | 启动依赖组装、会话管理 |
| 主循环 | `app/agent/loop.py` | Query Loop 主循环、Nudge、护栏 |
| 工具注册 | `app/agent/tooling.py` | ToolDefinition, ToolRegistry, 截断策略 |
| 工具清单 | `app/tools/__init__.py` | 14 个内置工具装配 |
| 权限 | `app/agent/permissions.py` | PermissionManager, 三级权限 |
| MCP 管理 | `app/mcp/manager.py` | McpManager, 生命周期 |
| MCP 客户端 | `app/mcp/client.py` | StdioMcpClient, JSON-RPC |
| MCP HTTP | `app/mcp/http_client.py` | HttpMcpClient |
| 记忆编排 | `app/memory/pipeline.py` | MemoryPipeline, 反馈闭环 |
| 记忆写链 | `app/memory/write_pipeline.py` | 抽取→验证→整理→衰减 |
| 记忆读链 | `app/memory/read_pipeline.py` | 检索→排序→注入 |
| 记忆验证 | `app/memory/verifier.py` | 去重/冲突/替代判定 |
| 记忆存储 | `app/memory/store.py` | JSON + Qdrant 双写 |
| 工作记忆 | `app/state/working_memory.py` | TTL 滑动窗口 WM |
| 上下文压缩 | `app/context/compactor.py` | Snip + Microcompact |
| 自动压缩 | `app/context/auto_compact.py` | Full Compact 调度 |
| 上下文装配 | `app/context/runtime.py` | prepare_agent_context |
| 历史摘要 | `app/context/history_summarizer.py` | 模型摘要 + 规则兜底 |
| 模型适配 | `app/infra/model_registry.py` | OpenAI API 调用、流式 |
| 消息桥接 | `app/agent/message_bridge.py` | 内部消息 → OpenAI 格式 |
| 熔断 | `app/agent/circuit_breaker.py` | 电路熔断器 |
| 重试 | `app/agent/retry.py` | 指数退避重试 |
| 类型定义 | `app/types.py` | AgentStep, ChatMessage, ToolResult... |
| 配置 | `app/config.py` | 环境变量 → AppConfig |
