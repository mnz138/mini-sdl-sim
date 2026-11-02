"""实验室：设备资源、时间线记录、用pylabrobot仿真的移液工作站"""
import contextlib
import io
import math
from contextlib import asynccontextmanager

import asyncio
import numpy as np
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import LiquidHandlerChatterboxBackend
from pylabrobot.liquid_handling.errors import ChannelizedError
from pylabrobot.resources import (PLT_CAR_L5AC_A00, STARLetDeck, TIP_CAR_480_A00, agenbio_4_troughplate_75mL_Vb,cor_96_wellplate_360uL_Fb, hamilton_96_tiprack_300uL_filter, set_tip_tracking,set_volume_tracking)
from pylabrobot.resources.errors import TooLittleLiquidError

from .clock import SimClock
from .models import DeviceError

class Lab:
    def __init__(self, capacities:dict, clock:SimClock, seed:int=0, p_timeout:float=0.0):
        # 每台设备是一个 带容量的资源 ： 容量 1 = 同一时间只能服务1个样品
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in capacities.values()):
            raise ValueError("设备容量必须是正整数")
        if not 0 <= p_timeout <= 1:
            raise ValueError("p_timeout 必须在 0 和 1 之间")
        self.devices = {name: asyncio.Semaphore(n) for name, n in capacities.items()}
        self.clock = clock
        self.rng = np.random.default_rng(seed)
        self.p_timeout = p_timeout # 测量超时的注入概率，用来演练异常处理
        self.timeline = [] # 记录时间线，记录每个操作的开始时间、结束时间、设备名称、操作名称、样品编号
        self.incidents = [] # 记录异常事件，记录时间、设备名称、操作名称、样品编号、异常类型、异常信息
        self.liquid = None # 用于记录液体的状态，后续可以扩展为一个液体状态管理器;需要移液工作站的实验才创建

    @asynccontextmanager
    async def use_device(self, device:str, sample_id:str, step:str):
        """占用一个设备，设备忙时在这里排队，进入后记录开始时间和退出时记录结束时间"""
        async with self.devices[device]:
            start_time = self.clock.now()
            entry = {"sample": sample_id, "step": step, "device": device, "start": start_time, "end": None}
            self.timeline.append(entry)
            try:
                yield
            finally:
                entry["end"] = self.clock.now()

    def incident(self, kind:str, detail:str)-> None:
        """记录异常事件"""
        self.incidents.append({"t": round(self.clock.now(),1), "kind":kind, "detail": detail})

    async def close(self) -> None:
        """结束后保留最终实验耗时，并释放移液后端。重复关闭也安全。"""
        self.clock.pause()
        if self.liquid is not None:
            await self.liquid.stop()

class LiquidStation:
    """把pylabrobot的仿真移液工作站封装成一个实验室设备，能够 管枪头、管补液、算时间
    这里只连接 Chatterbox 仿真后端。接入真机还需校准、耗材适配和安全验证。
    """
    SECONDS_PER_TRANSFER = 12.0   # 一次“取枪头-吸-吐-弃枪头”的模拟耗时
    REFILL_SECONDS = 90.0         # 人工或自动补液耗时
    RACK_SWAP_SECONDS = 60.0      # 换一盒新枪头耗时

    def __init__(self, lab: Lab, stock_names: list, stock_volume_uL: float, quiet: bool = True):
        if not 1 <= len(stock_names) <= 4 or len(set(stock_names)) != len(stock_names):
            raise ValueError("储液名称需要 1 至 4 个，且不能重复")
        if not math.isfinite(stock_volume_uL) or stock_volume_uL <= 0:
            raise ValueError("储液体积必须是有限正数")
        set_tip_tracking(True)      # 打开枪头追踪：用过的位置不能再取
        set_volume_tracking(True)   # 打开液量追踪：吸多了会报 TooLittleLiquidError
        self.lab = lab
        self.quiet = quiet
        self.lh = LiquidHandler(backend=LiquidHandlerChatterboxBackend(num_channels=8), deck=STARLetDeck())
        tip_car = TIP_CAR_480_A00("tip_carrier")
        tip_car[0] = self.tips = hamilton_96_tiprack_300uL_filter("tips")
        plate_car = PLT_CAR_L5AC_A00("plate_carrier")
        plate_car[0] = self.stocks = agenbio_4_troughplate_75mL_Vb("stocks")   # 4 个储液槽
        plate_car[1] = self.plate = cor_96_wellplate_360uL_Fb("samples")      # 96 孔样品板
        self.lh.deck.assign_child_resource(tip_car, rails=3)
        self.lh.deck.assign_child_resource(plate_car, rails=15)
        self.stock_volume_uL = stock_volume_uL
        self.stock = dict(zip(stock_names, self.stocks.children))   # 名字 -> 储液槽
        for trough in self.stock.values():
            if stock_volume_uL > trough.tracker.max_volume:
                raise ValueError("储液体积超过槽容量")
            trough.tracker.set_volume(stock_volume_uL)
        self._next_well = 0

    async def setup(self) -> None:
        with self._mute():
            await self.lh.setup()

    async def stop(self) -> None:
        if self.lh.setup_finished:
            with self._mute():
                await self.lh.stop()

    def _mute(self):
        return contextlib.redirect_stdout(io.StringIO()) if self.quiet else contextlib.nullcontext()

    def new_well(self):
        if self._next_well >= self.plate.num_items:
            raise DeviceError("96 孔样品板已用完，请换板后开始新的实验")
        well = self.plate.get_all_items()[self._next_well]
        self._next_well += 1
        return well

    def _next_tip(self):
        spots = [s for s in self.tips.get_all_items() if s.has_tip()]
        if not spots:   # 异常 1：枪头用完 -> 换新盒
            self.tips.fill()
            self.lab.incident("tip_rack_swap", "枪头用完，换新枪头盒")
            return None
        return spots[0]

    async def transfer(self, stock_name: str, well, volume_uL: float) -> None:
        # 本项目单次移液使用 300 µL 枪头。2000 µL 必须拆分且分配到多个孔。
        if not math.isfinite(volume_uL) or not 0 < volume_uL <= 300:
            raise ValueError("单次移液体积必须在 0 至 300 µL 之间（不含 0）")
        if volume_uL > self.stock_volume_uL:
            raise DeviceError("单次移液量超过补满后的储液体积")
        if volume_uL > well.tracker.get_free_volume() + 1e-6:
            raise DeviceError("移液后会超过样品孔容量")
        trough = self.stock[stock_name]
        spot = self._next_tip()
        if spot is None:
            await self.lab.clock.sleep(self.RACK_SWAP_SECONDS)
            spot = self._next_tip()
        with self._mute():
            await self.lh.pick_up_tips([spot], use_channels=[0])
            try:
                try:
                    await self.lh.aspirate([trough], vols=[volume_uL], use_channels=[0])
                except (TooLittleLiquidError, ChannelizedError) as error:
                    # PLR 的部分后端把通道错误包装在 ChannelizedError 中。
                    if isinstance(error, ChannelizedError) and (
                        not error.errors or not all(isinstance(e, TooLittleLiquidError)
                                                    for e in error.errors.values())
                    ):
                        raise
                    trough.tracker.set_volume(self.stock_volume_uL)
                    self.lab.incident("refill", f"{stock_name} 储液不足，补液后重试")
                    await self.lab.clock.sleep(self.REFILL_SECONDS)
                    await self.lh.aspirate([trough], vols=[volume_uL], use_channels=[0])
                await self.lh.dispense([well], vols=[volume_uL], use_channels=[0])
            finally:
                # 即使吸液或吐液失败，也移除已经占用的枪头。
                if self.lh.head[0].has_tip:
                    await self.lh.discard_tips(use_channels=[0])
        await self.lab.clock.sleep(self.SECONDS_PER_TRANSFER)
