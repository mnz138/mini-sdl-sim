"""调度器：每个样品是一个协程，按流程一步步走；设备是带容量的资源，忙就排队。
流水线不需要专门写：样品 1 在测量时，样品 2 已经能占用空出来的移液工作站，
asyncio 会自动让它们交替推进。顺序模式（一个做完再做下一个）用来对比。
"""
import asyncio

from .models import DeviceError, RecipeInfeasible, Sample, Status


async def _do_step(step, sample: Sample, lab) -> None:
    async with lab.use_device(step.device, sample.sample_id, step.name):
        if step.action:
            await step.action(sample, lab)
        if step.seconds:
            await lab.clock.sleep(step.seconds)


async def _measure_with_retry(step, sample: Sample, lab, max_retries: int) -> None:
    while True:
        sample.move_to(Status.MEASURING, lab.clock.now())
        sample.attempts += 1
        try:
            await _do_step(step, sample, lab)
        except DeviceError as e:
            lab.incident("measure_error", f"{sample.sample_id} 第 {sample.attempts} 次测量失败：{e}")
            if sample.attempts > max_retries:
                sample.move_to(Status.FAILED, lab.clock.now(), f"重试 {max_retries} 次仍失败：{e}")
                return
            sample.move_to(Status.PREPARED, lab.clock.now(), "等待重测")
            continue
        sample.move_to(Status.DONE, lab.clock.now())
        return


async def process_sample(sample: Sample, profile, lab, max_retries: int = 2) -> None:
    """一个样品从配制到测量的完整生命周期。"""
    if max_retries < 0:
        raise ValueError("max_retries 不能为负")
    for step in profile.workflow:
        if step.kind == "prepare":
            if sample.status == Status.PLANNED:
                sample.move_to(Status.PREPARING, lab.clock.now())
            try:
                await _do_step(step, sample, lab)
            except RecipeInfeasible as e:
                # 配方本身不可行：这是有效的实验结论，记一个低分交给优化器，让它别再往这里找
                sample.feasible = False
                sample.result = profile.infeasible_value
                sample.raw["infeasible"] = str(e)
                sample.move_to(Status.DONE, lab.clock.now(), f"配方不可行：{e}")
                return
            except DeviceError as e:
                sample.move_to(Status.FAILED, lab.clock.now(), f"配制失败：{e}")
                return
        elif step.kind == "measure":
            sample.move_to(Status.PREPARED, lab.clock.now())
            await _measure_with_retry(step, sample, lab, max_retries)
            return
        else:
            raise ValueError(f"未知步骤类型：{step.kind}")


async def run_batch(samples: list, profile, lab, mode: str = "pipeline") -> None:
    if mode == "pipeline":
        tasks = [asyncio.create_task(process_sample(s, profile, lab)) for s in samples]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # 某个样品意外出错或用户取消时，先结束其它任务，再关闭实验室。
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    elif mode == "sequential":
        for s in samples:
            await process_sample(s, profile, lab)
    else:
        raise ValueError(f"未知调度模式：{mode}")
