"""数据类型：样品、状态机、两类错误"""
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum

class Status(str,Enum):
    """状态机"""
    PLANNED = "planned" # 优化器刚推荐，还没开始做
    PREPARING = "preparing" # 正在配制（移液、混合、固化……）
    PREPARED = "prepared" # 配制完成，等待测量
    MEASURING = "measuring" # 正在测量
    DONE = "done" # 测量完成，等待分析
    FAILED = "failed" # 设备故障导致本次实验失败，不作为配方低分

#状态机：每个状态只允许走到这些下一个状态，流程错会报错，而不是悄悄产生脏数据
ALLOWED = {
    Status.PLANNED: {Status.PREPARING,Status.FAILED},
    Status.PREPARING: {Status.PREPARED, Status.DONE, Status.FAILED},
    # done:配方本身不可行，直接记录结果
    Status.PREPARED: {Status.MEASURING,Status.FAILED},
    Status.MEASURING: {Status.PREPARED, Status.DONE, Status.FAILED},
    # prepared：测量失败，退回等待重测
    Status.DONE: set(),
    Status.FAILED: set(),
}


class InvalidTransition(Exception):
    """状态机错误,非法状态跳转，调度代码可能写错了"""

class DeviceError(Exception):
    """设备错误，测量失败，设备问题导致的测量失败,与配方无关，可重试"""

class RecipeInfeasible(Exception):
    """配方本身不可行，是有效实验结论，不重试设备。"""

@dataclass
class Sample:
    sample_id : str
    requested : dict #优化器要求的配方参数
    actual : dict | None = None # 实际配制的配方参数，可能和requested不一样
    status : Status = Status.PLANNED
    result : float | None = None # 交给优化器的目标值
    feasible : bool = True # 交给优化器的可行性标记
    raw : dict = field(default_factory=dict) #原始测量数据的摘要，比如 Rb,t
    events : list = field(default_factory=list) # 状态变更事件记录，用来追溯
    attempts : int = 0 # 测量尝试次数，首次测量也计入

    def move_to(self, new : Status, t : float, note : str = " ") -> None:
        """状态机跳转"""
        if new not in ALLOWED[self.status]:
            raise InvalidTransition(f"{self.sample_id} : {self.status.value} -> {new.value} 不允许")
        self.events.append({ "t": round(t, 1), "from": self.status.value, "to": new.value, "note": note })
        self.status = new

    def record(self) -> dict:
        """写入 JSONL 的独立快照，后续改动样品不会改变已有记录。"""
        return deepcopy({
            "sample_id": self.sample_id,
            "requested": self.requested,
            "actual": self.actual,
            "status": self.status.value,
            "result": self.result,
            "feasible": self.feasible,
            "raw": self.raw,
            "events": self.events,
            "attempts": self.attempts,
        })
