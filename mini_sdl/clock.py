"""模拟时钟：把实验室里的“秒”压缩成很短的真实时间，方便在电脑上快速跑完一整天的实验。"""
import asyncio
import math
import time

class SimClock:
    """模拟时钟类"""

    def __init__(self, scale: float = 1e-3):
        # scale = 1 个模拟秒对应多少真实秒。1e-3 表示 1 小时实验约 3.6 秒跑完。
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale 必须是有限正数")
        self.scale = scale
        self._t0 = None
        self._paused_at = 0.0

    def start(self) -> None:
        """启动模拟时钟"""
        self._paused_at = 0.0
        self._t0 = time.monotonic()

    def now(self) -> float:
        """经过的时间"""
        if self._t0 is None:
            return self._paused_at
        return self._paused_at + (time.monotonic() - self._t0) / self.scale

    async def sleep(self, sim_seconds: float) -> None:
        """模拟一台设备工作 sim_seconds 秒，await 期间其他样品可以继续。"""
        if not math.isfinite(sim_seconds) or sim_seconds < 0:
            raise ValueError("模拟耗时必须是有限非负数")
        await asyncio.sleep(sim_seconds * self.scale)

    def pause(self) -> None:
        """暂停计时，优化器思考下一批样品的实验方案，实验室的时钟不走"""
        self._paused_at = self.now()
        self._t0 = None

    def resume(self) -> None:
        """恢复计时"""
        if self._t0 is None:
            self._t0 = time.monotonic()
