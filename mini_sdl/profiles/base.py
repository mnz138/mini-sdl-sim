"""“实验配置”（profile）的公共部分。
调度器和优化器只认识这里定义的三样东西：
参数（Param）、流程步骤（Step）、以及每一步要做的动作。
换一个学科，只需要新写一个 profile。
"""
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

@dataclass(frozen=True)
class Param:
    name: str     # 代码里的名字，如 "c_salt"
    low: float    # 下限
    high: float   # 上限
    unit: str     # 单位
    label: str    # 中文说明


@dataclass(frozen=True)
class Step:
    name: str                 # 步骤名，如 "配液"
    device: str               # 占用哪台设备
    seconds: float            # 固定耗时（模拟秒）；动作自己计时的话可以写 0
    kind: str                 # "prepare" 或 "measure"，决定样品状态怎么变
    action: Optional[Callable[..., Awaitable[None]]] = None   # 这一步真正要做的事


class Profile:
    name: str = ""
    target: str = ""          # 优化目标的列名
    target_unit: str = ""
    params: list = []
    capacities: dict = {}     # 设备名 -> 容量
    workflow: list = []       # Step 列表，按顺序执行

    def truth(self, actual: dict) -> float:
        """仿真用的“真实”目标值（只有上帝视角知道），用来画图和评估优化效果。"""
        raise NotImplementedError

    def setup_lab(self, lab) -> None:
        """需要额外设备（如移液工作站）时在这里创建。"""
