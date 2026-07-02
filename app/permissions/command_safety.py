"""命令安全分类模块：只读命令白名单、高风险命令检测、危险模式识别。
   对标 Claude Code readOnlyValidation.ts + bashSecurity.ts 的核心逻辑。"""

import platform
from typing import Literal

# ── 命令风险等级 ──
RiskLevel = Literal["read_only", "caution", "high_risk", "critical"]

# ── 纯只读命令（对标 Claude Code READONLY_COMMANDS + EXTERNAL_READONLY_COMMANDS）──
_READ_ONLY_COMMANDS: set[str] = {
    # 文件查看
    "cat", "head", "tail", "less", "more", "nl",
    # 目录浏览
    "ls", "dir", "tree", "pwd",
    # 文本搜索
    "grep", "rg", "ack", "ag", "find",
    # 文本处理（只读）
    "wc", "sort", "uniq", "cut", "tr", "paste", "column",
    "diff", "comm", "cmp", "sdiff",
    # 文件信息
    "stat", "file", "strings", "du", "df",
    # 路径工具
    "which", "whereis", "where", "type", "realpath", "readlink",
    "basename", "dirname",
    # 系统信息
    "uname", "hostname", "uptime", "date", "cal",
    "whoami", "id", "groups", "who", "w",
    "free", "nproc", "getconf", "locale",
    # 进程/网络查看
    "ps", "top", "htop", "pgrep", "lsof", "ss", "netstat",
    "ip", "ifconfig", "ping", "traceroute",
    # 开发工具（纯查看/版本类子命令）
    "echo", "printf", "true", "false",
    "sleep", "seq", "expr", "test",
    "man", "info", "help", "whatis", "apropos",
    "history", "alias",
    "tar", "gzip", "gunzip", "bzip2", "xz", "zipinfo",
    # 版本管理（纯查看子命令单独判断）
    "git",
    # 包管理器（纯查看子命令单独判断）
    "npm", "yarn", "pnpm", "pip", "pip3",
    "python", "python3", "node", "go", "cargo", "rustc",
    # 数据库 CLI（纯查询操作单独判断）
}

# ── Git 只读子命令 ──
_GIT_READ_ONLY_SUBCMDS: set[str] = {
    "status", "log", "show", "diff", "branch", "tag",
    "remote", "ls-files", "ls-tree", "rev-list", "rev-parse",
    "blame", "describe", "stash list", "config", "cat-file",
    "for-each-ref", "reflog", "shortlog", "whatchanged",
    "checkout --", "checkout -b",  # -- 查看文件，-b 创建分支（需授权）
}

# ── Git 危险子命令 ──
_GIT_DANGEROUS_SUBCMDS: set[str] = {
    "push", "commit", "merge", "rebase", "reset", "clean",
    "gc", "prune", "filter-branch", "filter-repo",
}

# ── 包管理器只读子命令 ──
_PM_READ_ONLY_SUBCMDS: set[str] = {
    "ls", "list", "view", "info", "outdated", "why", "explain",
    "audit", "config list", "config get", "cache list",
}

# ── 绝对危险命令（对标 Claude Code deny patterns）──
_CRITICAL_COMMANDS: set[str] = {
    "format", "fdisk", "mkfs",
    "shutdown", "reboot", "halt", "poweroff", "init",
    "dd",  # 磁盘直接写入
}

# ── 高风险命令关键词 ──
_HIGH_RISK_KEYWORDS: set[str] = {
    "rm", "del", "erase", "rmdir", "rd",
    "sudo", "su", "runas", "doas", "pkexec",
    "chmod", "chown", "chgrp", "cacls", "icacls",
    "kill", "taskkill", "pkill", "killall",
    "mv", "cp", "rename", "ren",
    "mount", "umount",
    "systemctl", "service", "sc",
    "iptables", "netsh", "firewall",
    "wget", "curl",  # 网络下载
    "nc", "telnet", "ssh",  # 远程连接
    "git push", "git commit",  # git 危险操作
}

# ── 命令注入检测模式 ──
_SHELL_INJECTION_PATTERNS: list[str] = [
    r"\$\(.*\)",       # $() 命令替换
    r"`[^`]*`",        # 反引号命令替换
    r"\$\{[^}]*\}",    # ${} 变量替换
    r";\s*\w",         # 分号后跟命令
    r"&&\s*\w",        # && 后跟命令
    r"\|\|\s*\w",      # || 后跟命令
    r">\s*/",          # 输出重定向到绝对路径
    r"<\s*/",          # 输入重定向从绝对路径
]

# ── Windows PowerShell 高风险动词 ──
_PS_HIGH_RISK_VERBS: set[str] = {
    "remove-item", "stop-process", "stop-service",
    "set-executionpolicy", "clear-content", "disable-",
    "uninstall-", "unregister-", "set-acl",
    "restart-computer", "stop-computer",
}


def classify_command_risk(command: str) -> RiskLevel:
    """对单条命令做安全风险分类。

       返回 read_only / caution / high_risk / critical 四个级别。

       Args:
           command: 要分类的命令文本

       Returns:
           风险级别字符串
    """
    if not command or not command.strip():
        return "caution"

    lowered = command.strip().lower()
    tokens = lowered.split()
    main_cmd = _extract_base_command(tokens)

    # ── 绝对危险（对标 Claude Code deny）──
    if main_cmd in _CRITICAL_COMMANDS:
        return "critical"

    # ── 高风险检查 ──
    for kw in _HIGH_RISK_KEYWORDS:
        if lowered.startswith(kw) or f" {kw}" in lowered:
            return "high_risk"

    # ── PowerShell 高风险动词 ──
    if platform.system() == "Windows":
        for verb in _PS_HIGH_RISK_VERBS:
            if lowered.startswith(verb):
                return "high_risk"

    # ── 只读命令判断 ──
    if main_cmd in _READ_ONLY_COMMANDS:
        # git 子命令细化
        if main_cmd == "git":
            return _classify_git_command(tokens)
        # 包管理器子命令细化
        if main_cmd in ("npm", "yarn", "pnpm", "pip", "pip3"):
            return _classify_pm_command(tokens)
        return "read_only"

    # ── 默认：谨慎 ──
    return "caution"


def has_command_injection_risk(command: str) -> bool:
    """检测命令是否包含潜在的注入/命令替换模式。

       对标 Claude Code bashSecurity.ts 的 validateDangerousPatterns。

       Args:
           command: 要检测的命令文本

       Returns:
           True 表示存在注入风险
    """
    import re
    for pattern in _SHELL_INJECTION_PATTERNS:
        if re.search(pattern, command):
            return True
    return False


def is_git_dangerous(command: str) -> bool:
    """判断 git 命令是否包含危险操作。"""
    tokens = command.strip().lower().split()
    if len(tokens) < 2 or tokens[0] != "git":
        return False
    subcmd = " ".join(tokens[1:3]) if len(tokens) >= 3 else tokens[1]
    return subcmd in _GIT_DANGEROUS_SUBCMDS


# ── 内部辅助函数 ──

def _extract_base_command(tokens: list[str]) -> str:
    """从 token 列表中提取基础命令名。

       跳过环境变量前缀（VAR=value）和安全包装器（timeout/nice/nohup 等）。
    """
    # 跳过环境变量赋值（如 NODE_ENV=production npm ...）
    i = 0
    while i < len(tokens) and "=" in tokens[i]:
        i += 1
    if i >= len(tokens):
        return ""
    # 跳过安全包装器
    cmd = tokens[i]
    # 去掉路径前缀
    if "/" in cmd or "\\" in cmd:
        cmd = cmd.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return cmd


def _classify_git_command(tokens: list[str]) -> RiskLevel:
    """对 git 子命令做细化安全分类。"""
    if len(tokens) < 2:
        return "read_only"
    subcmd = tokens[1]
    full = " ".join(tokens[1:3]) if len(tokens) >= 3 else subcmd
    if full in _GIT_DANGEROUS_SUBCMDS or subcmd in _GIT_DANGEROUS_SUBCMDS:
        return "high_risk"
    if full in _GIT_READ_ONLY_SUBCMDS or subcmd in _GIT_READ_ONLY_SUBCMDS:
        return "read_only"
    # git checkout -b 是创建分支，需要授权
    if subcmd == "checkout" and "-b" in tokens:
        return "caution"
    return "read_only"


def _classify_pm_command(tokens: list[str]) -> RiskLevel:
    """对包管理器子命令做细化安全分类。"""
    if len(tokens) < 2:
        return "caution"
    subcmd = tokens[1]
    full = " ".join(tokens[1:3]) if len(tokens) >= 3 else subcmd
    if full in _PM_READ_ONLY_SUBCMDS or subcmd in _PM_READ_ONLY_SUBCMDS:
        return "read_only"
    # install / add / remove / update 等修改操作
    write_subcmds = {"install", "add", "remove", "uninstall", "update", "upgrade", "publish"}
    if subcmd in write_subcmds:
        return "caution"
    return "read_only"
