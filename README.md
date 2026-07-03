# BeanCode Agent

基于 Python 的本地命令行 AI Agent，支持代码分析、工具调用、多轮会话和持久记忆。

## 主要能力

- TUI 终端多轮对话界面（基于 Textual）
- 11 个内置工具：文件读写、搜索、命令执行、网页抓取、网页搜索、子代理调度
- 工具并行执行（只读工具线程池并发）
- Prompt Cache 优化（DeepSeek 缓存命中率 ~95%，token 费用降低约 90%）
- 支持 /model 命令切换模型
- 会话恢复（--resume latest）
- 持久记忆：MEMORY.md + USER.md 文件型注入
- 参考 Claude Code 的上下文压缩机制（Microcompact + Auto Compact）

## 运行环境

- Python `3.11+`
- 可用的 OpenAI-compatible API 接口

## 安装依赖

```bash
pip install -r requirements.txt
```

## 首次配置

```bash
python -m app.main --setup
```

按提示输入 API Key、Base URL、选择模型，配置自动保存到 `~/.bean/settings.json`。

## 启动

```bash
python -m app.main
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `python -m app.main` | 新建会话 |
| `python -m app.main --resume latest` | 恢复最近会话 |
| `python -m app.main --resume list` | 查看会话列表 |
| `python -m app.main --session <id>` | 按 ID 恢复指定会话 |
| `python -m app.main --setup` | 重新配置 API 和模型 |
| `/model` | 查看/切换模型 |
| `/mcp` | 管理 MCP Server |

## 配置文件

所有配置统一在 `~/.bean/settings.json`：

```json
{
  "api_key": "sk-xxx",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "models": ["deepseek-v4-flash", "deepseek-v4-pro"]
}
```

## 内置工具

| 工具 | 说明 |
|------|------|
| `run_command` | 执行系统命令（Bash/PowerShell） |
| `read_file` | 读取文件内容 |
| `edit_file` | 精确字符串替换编辑文件 |
| `write_file` | 写入/创建文件 |
| `glob_files` | 文件名通配符匹配 |
| `grep_files` | 文件内容搜索（ripgrep） |
| `web_search` | 网页搜索（DuckDuckGo） |
| `web_fetch` | 网页内容抓取 |
| `ask_user` | 结构化用户提问 |
| `agent_dispatch` | 子代理调度 |
| `memory` | 持久记忆读写 |

## 运行后生成的文件

- `.sessions/`：会话历史
- `.bean/`：权限规则持久化
- `debug.log`：运行日志
- `MEMORY.md`：Agent 持久记忆
