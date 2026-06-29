# LongBean MiniCode Agent

一个基于 Python 的本地命令行 Agent，主要用于代码库分析、工具调用、多轮会话和持久记忆。

## 主要能力

- 命令行多轮对话（TUI 终端界面）
- 支持恢复历史会话
- 支持代码库浏览、文件读取、符号分析等工具调用
- 持久记忆：MEMORY.md + USER.md 文件型全量注入，后台异步反思自动提取
- 针对代码分析场景做了上下文压缩和证据保真

## 运行环境

- Python `3.11+`
- 可用的 OpenAI 或 OpenAI-compatible 接口

## 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 必须配置的文件

### 1. 准备 `.env`

先复制示例配置：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

### 2. 最少必填配置

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
WORKSPACE_ROOT=.
```

说明：

- `OPENAI_API_KEY`：必填，模型 API Key
- `OPENAI_BASE_URL`：模型服务地址
- `OPENAI_MODEL`：主对话模型名
- `WORKSPACE_ROOT`：Agent 允许操作的工作区根目录

## 如何启动

```bash
python -m app.main
```

启动后会进入 TUI 交互界面。

## 常用启动参数

### 新建会话

```bash
python -m app.main
```

### 恢复最近一次会话

```bash
python -m app.main --resume latest
```

### 查看会话列表

```bash
python -m app.main --resume list
```

### 按指定会话 ID 恢复

```bash
python -m app.main --session <session_id>
```

## 持久记忆

Agent 支持跨会话的持久记忆，基于 MEMORY.md 和 USER.md 两个 Markdown 文件：

- **MEMORY.md**：项目规范、环境约束、经验教训等 Agent 自身笔记
- **USER.md**：用户身份、偏好、工作风格等用户信息

**记忆来源：**
- 对话中 Agent 主动判断后调用 `memory` 工具写入
- 每 10 轮用户对话后，后台异步反思线程自动提取新信息写入
- 用户直接编辑 `.memory/MEMORY.md` 或 `.memory/USER.md` 文件

**注入方式：** 会话启动时全量读取两个文件，作为 system prompt 的持久记忆段注入。

**安全机制：** 写入前经过审批开关、注入安全扫描、文件漂移检测三道保护。

## 首次使用建议

1. 复制 `.env.example` 为 `.env`
2. 填写 `OPENAI_API_KEY`
3. 确认 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`
4. 运行 `python -m app.main`

## 运行后会生成的目录

- `.sessions/`：会话历史
- `.memory/`：持久记忆文件（MEMORY.md、USER.md）
- `debug.log`：运行日志

## 最小可运行示例

`.env` 最少可先写成：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
WORKSPACE_ROOT=.
```

启动：

```bash
python -m app.main
```

然后输入：

```text
帮我分析一下当前项目 app/logger.py 的启动链路
```

## 说明

- 当前项目是本地 CLI Agent，不是 HTTP 服务
- 启动失败时，优先检查 `.env`、模型接口可达性、`WORKSPACE_ROOT` 是否正确
