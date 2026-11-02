"""闭环：推荐 -> 配制 -> 测量 -> 回流，循环若干轮。每个样品写一行 JSONL 方便追溯。"""
import json
from pathlib import Path

from .clock import SimClock
from .lab import Lab
from .models import Sample, Status
from .scheduler import run_batch


async def make_lab(profile, seed: int = 0, p_timeout: float = 0.1, scale: float = 1e-3) -> Lab:
    if not 0 <= p_timeout <= 1:
        raise ValueError("p_timeout 必须在 0 和 1 之间")
    lab = Lab(profile.capacities, SimClock(scale), seed=seed, p_timeout=p_timeout)
    try:
        profile.setup_lab(lab)
        if lab.liquid is not None:
            await lab.liquid.setup()
    except BaseException:
        await lab.close()
        raise
    # 硬件布局创建和后端初始化不属于样品流程，不放大为模拟实验耗时。
    lab.clock.start()
    return lab


async def run_campaign(profile, optimizer, rounds: int = 6, batch_size: int = 8, seed: int = 0,
                       p_timeout: float = 0.1, mode: str = "pipeline", log_path=None, verbose: bool = True,
                       on_round=None, scale: float = 1e-3):
    """on_round(r, history, lab)：每轮结束后调用一次，前端用它刷新进度；回调期间实验室时钟暂停。"""
    if rounds < 1 or batch_size < 1:
        raise ValueError("rounds 和 batch_size 必须是正整数")
    if mode not in {"pipeline", "sequential"}:
        raise ValueError(f"未知调度模式：{mode}")
    if profile.name == "electrolyte" and rounds * batch_size > 96:
        raise ValueError("电解液单次实验最多使用 96 个孔，请减小轮数或批量")
    history = []
    log = None
    lab = None
    try:
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open("x", encoding="utf-8")  # 已有记录不能被覆盖。
        lab = await make_lab(profile, seed=seed, p_timeout=p_timeout, scale=scale)
        for r in range(1, rounds + 1):
            lab.clock.pause()                      # 优化器思考时实验室时钟暂停
            try:
                recs = optimizer.ask(batch_size)
            finally:
                lab.clock.resume()
            if len(recs) != batch_size:
                raise ValueError("优化器返回的配方数量与 batch_size 不一致")
            samples = [Sample(f"R{r:02d}-S{i:02d}", {k: float(v) for k, v in rec.items()})
                       for i, rec in enumerate(recs, 1)]
            await run_batch(samples, profile, lab, mode)
            done = [s for s in samples if s.status == Status.DONE]
            # 先保存原始实验结果，即使优化器回流失败也可追溯本轮样品。
            for s in samples:
                history.append({"round": r, **s.record()})
                if log:
                    log.write(json.dumps(history[-1], ensure_ascii=False, allow_nan=False) + "\n")
            if log:
                log.flush()
            lab.clock.pause()
            try:
                # 设备故障不当作配方低分；仅回流已完成的样品。
                optimizer.tell([{**s.actual, profile.target: s.result} for s in done])
            finally:
                lab.clock.resume()
            if verbose:
                best = max((h["result"] for h in history if h["status"] == "done"), default=float("nan"))
                print(f"第 {r} 轮：完成 {len(done)}/{len(samples)}，本轮最好 "
                      f"{max((s.result for s in done), default=float('nan')):.2f}，迄今最好 {best:.2f} "
                      f"{profile.target_unit}，实验室时钟 {lab.clock.now() / 3600:.2f} 小时")
            if on_round:
                lab.clock.pause()
                try:
                    on_round(r, history, lab)
                finally:
                    lab.clock.resume()
    finally:
        if log is not None:
            log.close()
        if lab is not None:
            await lab.close()
    return history, lab
