import asyncio
from types import SimpleNamespace
import pytest

from mini_sdl.campaign import make_lab
from mini_sdl.models import DeviceError, Sample, Status
from mini_sdl.profiles import PROFILES
from mini_sdl.profiles.base import Step
from mini_sdl.scheduler import run_batch

RECIPES = [{"c_salt": 0.5 + 0.15 * i, "x_ec": 0.3 + 0.05 * i} for i in range(8)]


async def _makespan(mode, n_cells=1):
    profile = PROFILES["electrolyte"](n_cells=n_cells)
    lab = await make_lab(profile, p_timeout=0.0)
    samples = [Sample(f"S{i}", r) for i, r in enumerate(RECIPES)]
    try:
        await run_batch(samples, profile, lab, mode)
        assert all(s.status == Status.DONE for s in samples)
        return lab.clock.now()
    finally:
        await lab.close()


def test_pipeline_is_faster_than_sequential():
    seq = asyncio.run(_makespan("sequential"))
    pipe = asyncio.run(_makespan("pipeline"))
    assert pipe < 0.85 * seq


def test_second_conductivity_cell_shortens_batch():
    one = asyncio.run(_makespan("pipeline", n_cells=1))
    two = asyncio.run(_makespan("pipeline", n_cells=2))
    assert two < 0.75 * one


def test_persistent_device_error_ends_in_failed_after_retries():
    async def go():
        profile = PROFILES["electrolyte"]()
        lab = await make_lab(profile, p_timeout=1.0)     # 每次测量都超时
        s = Sample("S0", RECIPES[0])
        try:
            await run_batch([s], profile, lab)
            assert len(lab.incidents) == 3
            return s
        finally:
            await lab.close()
    s = asyncio.run(go())
    assert s.status == Status.FAILED and s.attempts == 3 and s.result is None


def test_infeasible_recipe_is_reported_as_low_score():
    async def go():
        profile = PROFILES["tim"]()
        lab = await make_lab(profile, p_timeout=0.0)
        s = Sample("S0", {"phi": 0.75, "r_small": 0.0})  # 超过最大堆积分数
        try:
            await run_batch([s], profile, lab)
            assert s.attempts == 0
            return s
        finally:
            await lab.close()
    s = asyncio.run(go())
    assert s.status == Status.DONE and not s.feasible and s.result == 0.0


def test_measurement_can_succeed_on_second_retry():
    async def go():
        profile = PROFILES["cavity"]()

        async def measure(sample, lab):
            if sample.attempts < 3:
                raise DeviceError("短暂故障")
            sample.result = 1.0

        profile.workflow = [Step("prepare", "dispenser", 0, "prepare"),
                            Step("measure", "cavity", 0, "measure", measure)]
        lab = await make_lab(profile)
        sample = Sample("S0", {})
        try:
            await run_batch([sample], profile, lab)
            assert sample.status == Status.DONE and sample.attempts == 3
            assert sample.result == 1.0 and len(lab.incidents) == 2
        finally:
            await lab.close()
    asyncio.run(go())


def test_devices_never_exceed_capacity():
    async def go():
        profile = PROFILES["cavity"]()
        lab = await make_lab(profile)
        active, peak = 0, 0

        async def occupy(i):
            nonlocal active, peak
            async with lab.use_device("oven", str(i), "cure"):
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.002)
                finally:
                    active -= 1

        try:
            await asyncio.gather(*(occupy(i) for i in range(20)))
            assert peak == profile.capacities["oven"] == 8
            assert active == 0
            assert all(row["end"] >= row["start"] for row in lab.timeline)
        finally:
            await lab.close()
    asyncio.run(go())


def test_unexpected_error_cancels_pending_samples_and_releases_resources():
    async def go():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def prepare(sample, lab):
            if sample.sample_id == "bad":
                await started.wait()
                raise RuntimeError("unexpected failure")
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

        profile = SimpleNamespace(capacities={"device": 2}, setup_lab=lambda lab: None,
                                  workflow=[Step("prepare", "device", 0, "prepare", prepare)])
        lab = await make_lab(profile)
        try:
            samples = [Sample(name, {}) for name in ("bad", "waiting", "queued")]
            with pytest.raises(RuntimeError, match="unexpected failure"):
                await run_batch(samples, profile, lab)
            assert cancelled.is_set()
            assert all(row["end"] is not None for row in lab.timeline)
            async with asyncio.timeout(0.1):
                async with lab.use_device("device", "next1", "test"):
                    async with lab.use_device("device", "next2", "test"):
                        pass
        finally:
            await lab.close()
    asyncio.run(go())
