# LongBean MiniCode Agent

一个基于 Python 的本地命令行 Agent，主要用于代码库分析、工具调用、多轮会话和长期记忆实验。

## 主要能力

- 命令行多轮对话
- 支持恢复历史会话
- 支持代码库浏览、文件读取、符号分析等工具调用
- 支持长期记忆落盘
- 可选接入 Qdrant 做语义向量检索
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
QDRANT_ENABLED=false
```

说明：

- `OPENAI_API_KEY`：必填，主模型 API Key
- `OPENAI_BASE_URL`：模型服务地址；如果使用兼容接口，需要改成对应地址
- `OPENAI_MODEL`：主对话模型名
- `WORKSPACE_ROOT`：Agent 允许操作的工作区根目录
- `QDRANT_ENABLED=false`：先关闭向量库，优先确认主链路能跑通

## 可选配置

### Embedding 配置

如果要启用长期记忆的语义检索，可以补充：

```env
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_DIMENSIONS=1024
```

规则：

- `EMBEDDING_API_KEY` 留空时，默认回退到 `OPENAI_API_KEY`
- `EMBEDDING_BASE_URL` 留空时，默认回退到 `OPENAI_BASE_URL`
- `EMBEDDING_DIMENSIONS=0` 表示不显式传入维度

### Qdrant 配置

如果要启用 Qdrant：

```env
QDRANT_ENABLED=true
QDRANT_URL=http://127.0.0.1:6333
QDRANT_PATH=
QDRANT_API_KEY=
QDRANT_COLLECTION=project_memories
```

说明：

- 默认长期记忆权威存储仍然是本地 JSON
- 打开 Qdrant 后，会额外建立向量索引
- 启动时会尝试把本地记忆和向量索引做一次收敛同步

## 如何启动

CLI 入口在 [app/main.py]

当前建议使用下面两种方式启动：

### 方式 1：直接运行脚本

```bash
python app/main.py
```

### 方式 2：按模块方式启动

```bash
python -m app.main
```

启动后会进入交互式命令行：

```text
LongBean MiniCode Agent 已启动，输入 quit 或 exit 退出。
You>
```

## 常用启动参数

### 新建会话

```bash
python app/main.py
```

### 恢复最近一次会话

```bash
python app/main.py --resume latest
```

### 查看会话列表

```bash
python app/main.py --resume list
```

### 按指定会话 ID 恢复

```bash
python app/main.py --session <session_id>
```

例如：

```bash
python app/main.py --session 81d81acc8c97
```

## 显式记忆命令

启动后，除了普通提问，也可以手动把偏好或规则写入长期记忆。

当前推荐的显式命令格式：

```text
/memory add project: <内容>
/memory add <内容>
```

例如：

```text
/memory add project: 修改 session 相关逻辑时优先走 repository/service 层
/memory add 新增接口时优先补测试
```

说明：

- `project`：项目规则，例如架构约束、代码约定、协作规范
- 不写作用域时，默认按 `project` 处理
- `/memory add ...` 当前会写入工作区下的 `.memory/memory.json`

这类输入会直接写入长期记忆，不会继续走一轮普通 Agent 对话。

## USER.md 命令

如果你想维护工作区级的用户偏好文件，而不是长期记忆 JSON，可以使用 `/user` 命令。它会直接读写工作区根目录的 `USER.md`。

常见用法：

```text
/user
/user add identity 我的名字是长豆角，我给你取名字为小多
/user add preferences 默认中文回答，回答尽量直接
/user add custom 在 tmp 目录新建的代码不用帮我测试
/user set preferences.language zh-CN
/user set preferences.verbosity concise
/user set coding_style.comments 修改代码时加中文注释
/user paths
/user reset
```

说明：

- `/user add <内容>` 兼容旧写法，默认会把内容追加到 `Custom Instructions`
- `/user add identity <内容>` 会追加到 `## Identity`
- `/user add preferences <内容>` 会追加到 `## Preference Instructions`
- `/user add custom <内容>` 会追加到 `## Custom Instructions`
- `/user set <key> <value>` 会把结构化字段写入 `USER.md`
- `/user` 可以直接查看当前 `USER.md` 摘要
- `/user paths` 可以查看当前 `USER.md` 的真实路径
- `/user reset` 会删除当前工作区的 `USER.md`

当前支持的常用 `set key`：

- `preferences.language`
- `preferences.verbosity`
- `preferences.response_style`
- `coding_style.comments`
- `identity_instructions`
- `preference_instructions`
- `custom_instructions`

当前两类入口的区别：

- `/memory add ...` -> 写 `.memory/memory.json`
- `/user add ...` / `/user set ...` -> 写 `USER.md`

## 首次使用建议

建议按这个顺序检查：

1. 复制 `.env.example` 为 `.env`
2. 填写 `OPENAI_API_KEY`
3. 确认 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`
4. 先保持 `QDRANT_ENABLED=false`
5. 运行 `python app/main.py`

## 运行后会生成的目录

运行过程中通常会生成这些目录或文件：

- `.sessions/`：会话历史
- `.memory/`：长期记忆本地存储
- `.context_state/`：上下文状态与压缩快照
- `.qdrant_storage/`：本地 Qdrant 持久化目录
- `debug.log`：运行日志

一般不需要手动编辑这些内容。

## 最小可运行示例

`.env` 最少可先写成：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
WORKSPACE_ROOT=.
QDRANT_ENABLED=false
```

启动：

```bash
python app/main.py
```

然后输入：

```text
帮我分析一下 app/main.py 的启动链路
```




## 说明

- 当前项目是本地 CLI Agent，不是 HTTP 服务
- 启动失败时，优先检查 `.env`、模型接口可达性、`WORKSPACE_ROOT` 是否正确
