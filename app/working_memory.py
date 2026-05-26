


from dataclasses import dataclass, field


def _dedupe_keep_last(items: list[str],max_items: int)->list[str]:
    """
    列表去重，并保留最近一次出现的顺序。
    """

    # seen 用来记录已经保留过的文本，避免重复
    seen: set[str]=set()

    # result 保存去重后的结果
    result: list[str]=[]

    # 倒序遍历，表示优先保留“最近加入”的内容
    for item in reversed(items):
        # 先去掉首尾空格，避免空字符串和纯空白内容混入
        text=item.strip()

        # 空内容直接跳过
        if not text:
            continue

        # 如果这条内容已经保留过，就不再重复加入
        if text in seen:
            continue

        # 记录这条内容已经出现过
        seen.add(text)

        # 把当前内容加入结果
        result.append(text)
    # 因为前面是倒序遍历，这里再翻回正常顺序
    result.reverse()
    # 只保留最后 max_items 条，避免列表无限增长
    return result[-max_items:]


@dataclass(slots=True)
class WorkingMemory:
    """
    短期工作记忆：保存当前任务最重要、最不能丢的上下文状态。
    """

    # 当前用户正在让Agent做的最主要目标
    current_goal: str=""

    # 最近几条关键决策，例如“先做规则摘要，再做工作记忆”
    recent_decisions: list[str] = field(default_factory=list)

    # 最近几条失败或异常信息，方便后面注入 prompt 提醒模型避坑
    recent_failures: list[str] = field(default_factory=list)

    # 当前正在频繁操作的文件或目录路径
    active_paths: list[str] = field(default_factory=list)

    # recent_decisions 最多保留多少条
    max_decisions: int = 5

    # recent_failures 最多保留多少条
    max_failures: int = 5

    # active_paths 最多保留多少条
    max_paths: int = 8

    def set_current_goal(self,goal: str)->None:
        """
        设置当前任务目标
        """

        # goal就是用户当前最核心的任务描述
        self.current_goal=goal.strip()

    def add_decision(self,decision: str)->None:
        """
        记录一条关键决策
        """

        # decision表示一条已经确认的执行方向或约束
        text=decision.strip()

        # 空内容不记录
        if not text:
            return 
        
        # 追加到原列表
        self.recent_decisions.append(text)

        # 再统一做去重和数量限制
        self.recent_decisions=_dedupe_keep_last(
            self.recent_decisions,
            self.max_decisions
        )

    def add_failure(self,failure: str)->None:
        """
        记录一条最近失败的信息。
        """

        # failure表示最近一次报错、异常或失败结论
        text=failure.strip()
        # 空内容不记录
        if not text:
            return

        # 先追加到失败列表
        self.recent_failures.append(text)

        # 再统一做去重和数量限制
        self.recent_failures = _dedupe_keep_last(
            self.recent_failures,
            self.max_failures,
        )

    def add_active_path(self,path: str)->None:
        """
        记录当前活跃路径，例如最近读写过的文件或目录。
        """
        # path 表示当前任务里比较关键、最近在频繁操作的路径
        text=path.strip()

        # 空路径不记录
        if not text:
            return

        # 先追加到活跃路径列表
        self.active_paths.append(text)

        # 再统一做去重和数量限制
        self.active_paths = _dedupe_keep_last(
            self.active_paths,
            self.max_paths,
        )

    def clear_failures(self) -> None:
        """
        清空最近失败记录。
        """
        # 当错误已经处理完，或者想重新开始时可以调用
        self.recent_failures.clear()

    def to_dict(self) -> dict[str, object]:
        """
        转成可序列化字典，后面如果要持久化可以直接复用。
        """
        # 这里返回的是普通字典，方便后续写入 JSON
        return {
            "current_goal": self.current_goal,
            "recent_decisions": list(self.recent_decisions),
            "recent_failures": list(self.recent_failures),
            "active_paths": list(self.active_paths),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "WorkingMemory":
        """
        从字典恢复 WorkingMemory。
        """
        # current_goal 是单个字符串字段
        current_goal = str(data.get("current_goal", "")).strip()

        # recent_decisions 要从字典里恢复成字符串列表
        recent_decisions = [
            str(item).strip()
            for item in data.get("recent_decisions", [])
            if str(item).strip()
        ]

        # recent_failures 要从字典里恢复成字符串列表
        recent_failures = [
            str(item).strip()
            for item in data.get("recent_failures", [])
            if str(item).strip()
        ]

        # active_paths 要从字典里恢复成字符串列表
        active_paths = [
            str(item).strip()
            for item in data.get("active_paths", [])
            if str(item).strip()
        ]

        # 用恢复出的字段创建一个 WorkingMemory 实例
        memory = cls(
            current_goal=current_goal,
            recent_decisions=recent_decisions,
            recent_failures=recent_failures,
            active_paths=active_paths,
        )

        # 恢复后顺手做一次去重和数量限制，避免旧数据太脏
        memory.recent_decisions = _dedupe_keep_last(
            memory.recent_decisions,
            memory.max_decisions,
        )
        memory.recent_failures = _dedupe_keep_last(
            memory.recent_failures,
            memory.max_failures,
        )
        memory.active_paths = _dedupe_keep_last(
            memory.active_paths,
            memory.max_paths,
        )

        return memory
    
    def format_for_prompt(self)->str:
        """
        把短期工作记忆格式化成可注入prompt的文本。
        """
        # parts 用来逐段收集最终要注入 prompt 的文本
        parts: list[str] = []

        # 当前目标单独作为第一段
        if self.current_goal:
            parts.append(f"当前目标：{self.current_goal}")

        # 最近关键决策按列表输出
        if self.recent_decisions:
            parts.append("最近关键决策：")
            for item in self.recent_decisions:
                parts.append(f"- {item}")

        # 最近失败按列表输出
        if self.recent_failures:
            parts.append("最近失败：")
            for item in self.recent_failures:
                parts.append(f"- {item}")

        # 当前活跃路径按列表输出
        if self.active_paths:
            parts.append("当前活跃路径：")
            for item in self.active_paths:
                parts.append(f"- {item}")

        # 拼成一段完整文本，后面直接给 prompt 使用
        return "\n".join(parts).strip()




