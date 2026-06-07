# 上下文压缩体系重构设计

**日期**: 2026-06-07  
**状态**: 待实现

## 背景与问题

当前压缩体系存在以下问题：

1. **阶段边界模糊**：`compact_recent_messages` 混合了规则裁剪和语义折叠，`_semantic_compact_old_tool_interactions` 用规则模拟了一个伪摘要，与 Collapse AI 职责重叠
2. **规则抽取太脆**：`build_active_context_event_snapshot` 通过关键词匹配和语义评分从 tool_result 中提取关键信息写入 working memory，换一个表述就漏掉
3. **WM 写入只有规则**：6 个 working memory 写入路径全部是规则驱动，`extract_decisions_from_assistant` 靠关键词匹配 decision_tokens，模型自然语言表达的决策经常漏
4. **保存时机有窗口期**：模型返回后 WM 写入的内容（key_decision、risk、constraint）要到下一轮才落盘到 context_state，进程意外退出则丢失
5. **话题切换无清理**：旧话题的 WM 条目在新话题下仍残留，仅靠 TTL 过期

## 设计目标

- 三阶段职责清晰：Snip Compact（纯规则裁剪）→ Microcompact（结构性清理 + 轻量 AI 承接）→ Collapse AI（摘要折叠）
- 关键语义抽取节点用轻量模型替代规则
- 双保存点保证记忆不丢
- 话题切换时自动清理旧 WM 条目

## 一、压缩三阶段

```
Snip Compact  →  Microcompact  →  Collapse AI
 (规则裁剪)       (微压缩+AI承接)    (AI摘要折叠)
  每轮必跑        tool_result超阈值     预算92%触发
  纯规则          有时间节流           模型优先/规则兜底
```

### 1.1 Snip Compact（规则裁剪）

**触发**：每轮必跑  
**模型调用**：无  
**改动类型**：修改现有逻辑

```
阶段顺序（均在 compact_recent_messages 内）：
  1. 删除 assistant_progress
  2. 超大 tool_result 落盘 + 首尾预览
  3. read_file / list_files / grep_files 去重
  4. [新增] 过滤纯确认类 tool_result：当 tool_result 内容仅包含 "文件写入成功""目录创建成功""命令执行成功" 等无信息量的确认短语时，将整条 tool_result 替换为 "[工具执行成功，内容无额外信息]" 占位
  5. [新增] 连续重复 assistant 消息去重（内容完全相同则保留最新一条）
  6. 按优先级裁剪旧消息（兜底）
  
  移除：_semantic_compact_old_tool_interactions（规则伪摘要，删掉）
```

### 1.2 Microcompact（微压缩 + 轻量 AI 承接）

**触发**：tool_result 数量超过 `keep_recent_tool_results` 阈值，且通过时间节流（冷却 1 小时）  
**模型调用**：有（轻量，每次处理一小批 tool_result）  
**改动类型**：修改 + 新增

```
流程：
  1. 判断哪些 tool_result 需要清正文（窗口外 N 轮）
  2. [改为] 调轻量模型从被清 tool_result 原文中抽取：
     - tool_findings：关键工具发现/结论
     - open_issues：风险/未解决问题
     - key_decisions：重要决策
  3. 模型抽取结果写入 working memory
  4. 清掉 tool_result 正文，替换为占位标记

  移除：build_active_context_event_snapshot 规则抽取
  移除：_promote_microcompact_carryovers 整条链路
  移除：carried_tool_findings / carried_open_issues 字段
```

**轻量模型调用约定**：
- 共用 `OlderHistorySummarizer` 的模型实例（同一套 api_key、base_url、model_name）
- 独立 circuit breaker（避免与 Collapse AI 的熔断互相影响），命名 `microcompact_extractor`
- 输入：被清 tool_result 的原文 + tool_name + 对应的 tool_call input
- 输出：结构化 JSON（tool_findings[]、open_issues[]、key_decisions[]）
- 失败回退：模型调用失败或熔断时，跳过抽取，直接清正文（不写入 WM）

### 1.3 Collapse AI（摘要折叠）

**触发**：上下文 token 达到 usable_budget × 0.92  
**模型调用**：有  
**改动类型**：不变

```
策略递进：
  Session Memory Compact → Full Compact
  
  保持不变：
  - 模型优先、规则兜底
  - structured snapshot 产出
  - active_context_summary 更新
```

## 二、Working Memory 写入改造

### 2.1 现有 6 个路径的处理

| # | 路径 | 改动 |
|---|------|------|
| 1 | `remember_user_intent` | **不动**，关键词提取 preference/constraint 够用 |
| 2 | `record_tool_call`（提取路径） | **不动**，纯结构化操作 |
| 3 | `record_tool_failure` | **不动**，tool_name + error 拼装足够 |
| 4 | `record_assistant_reply` | **改为轻量模型**，从 assistant 回复中抽取 key_decision / risk / constraint |
| 5 | agent_loop 异常 | **不动**，直接拼字符串 |
| 6 | Microcompact AI 承接 | **改为轻量模型**（见 1.2） |

### 2.2 #4 record_assistant_reply 改造

**当前**：`extract_decisions_from_assistant()` 用 `decision_tokens` 关键词匹配  
**改为**：轻量模型从 assistant 回复中抽取

```
共用 OlderHistorySummarizer 的模型实例，独立 circuit breaker（命名 "assistant_reply_extractor"）
输入：assistant 回复内容，取前 1200 字符（超出截断）
输出：结构化 JSON（key_decisions[]、recent_risks[]、preferences[]、constraints[]）
失败回退：模型失败时回退到现有的 extract_decisions_from_assistant / extract_recent_risks 规则函数
```

### 2.3 话题切换清理

```
逻辑：
  当新的 user_intent 写入时：
    1. 对当前和上一轮 user_intent 分别做中文分词（取 2-4 字 ngram）
    2. 计算 ngram 交集数
    3. 交集数 <= 1 → 判定为话题切换
    4. 清除非持久类别的旧 WM 条目：
       - 保留：user_preference、project_constraint
       - 清理：active_task、key_decision、recent_risk、error_context、tool_finding
```

## 三、Context State 双保存点

### 3.1 保存点A（请求前，现有）

```
时机：prepare_agent_context() 末尾，pipeline 完成后
内容：完整 state
  - active_context_snapshot
  - active_context_summary
  - compacted_messages
  - source_history_fingerprint
  - compaction_history
  - microcompact 节流时间
```

### 3.2 保存点B（请求后，新增）

```
时机：模型返回 + WM 写入完成后（MemoryWritePipeline 执行后）
内容：增量合并 snapshot

流程：
  1. 从当前 WM 构建 protected_snapshot
  2. 读取磁盘 context_state
  3. 将 protected_snapshot 合并进 active_context_snapshot：
     - 同 key 的条目：新覆盖旧（去重）
     - 不同 key 的条目：保留
  4. 写回磁盘
```

### 3.3 保证

```
每轮流程：
  保存点A                   保存点B（新增）
  ↓                        ↓
  存完整压缩机基线          存增量 WM 快照

  → 进程在保存点A前挂：state 有保存点B上次写入的最新快照
  → 进程在保存点B前挂：state 有保存点A写入的压缩机基线
  → 两个保存点之间的窗口期：最多丢失当轮 WM 增量，不影响基线
```

## 四、上下文注入确认

所有产出的数据均进入模型上下文：

| 数据 | 注入位置 | 形式 |
|------|---------|------|
| Snip/Microcompact 处理后的消息 | `pipeline_result.messages` | 对话历史 |
| Collapse AI 摘要 | `pipeline_result.messages` 中的 system marker | `[全量压缩] 已折叠较早消息数：N...` |
| active_context_summary | system prompt 中的 `当前会话压缩基线` | 结构化快照文本 |
| working_memory 条目 | system prompt 中的 `当前工作记忆` | 当前任务/决策/风险/错误上下文 |
| 长期记忆 | system prompt 中的 `固定/相关/项目长期记忆` | user/project scoped 记忆条目 |

## 五、涉及文件

| 文件 | 改动 |
|------|------|
| `app/context_compactor.py` | Snip：新增过滤逻辑，删除 `_semantic_compact_old_tool_interactions`；Microcompact：AI 承接替换规则抽取 |
| `app/context_compactor_pipeline.py` | 移除 `_promote_microcompact_carryovers` 合并逻辑 |
| `app/context_compact_memory.py` | 可能简化，移除不再需要的 `build_active_context_event_snapshot` |
| `app/context_runtime.py` | 新增保存点B；话题切换清理 |
| `app/memory_write_pipeline.py` | `record_assistant_reply` 改为轻量模型 |
| `app/history_summarizer.py` | 新增轻量 AI 承接方法（Microcompact 和 record_assistant_reply 共用） |
| `app/working_memory.py` | 新增话题切换清理方法 |
| `app/context_state.py` | 新增增量合并 snapshot 方法 |

## 六、不变的部分

- Collapse AI 整体逻辑
- `remember_user_intent`、`record_tool_call`、`record_tool_failure`、agent_loop 异常处理
- 上下文注入链路（`MemoryReadPipeline.build_context` + `_build_compacted_request`）
- context_state 文件格式（保持向后兼容）
- 各阶段常量阈值（可后续调参）

## 七、风险与回退

- **模型调用增加**：Microcompact 和 record_assistant_reply 各增加一次轻量模型调用，需要关注延迟和成本
- **回退策略**：所有模型调用点均有「失败回退到不写入」或「失败回退到规则」的兜底
- **向后兼容**：context_state 新增字段不破坏旧格式解析；旧 snapshot 仍可被新逻辑读取
