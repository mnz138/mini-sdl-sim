import asyncio

import pytest
from pylabrobot.liquid_handling.errors import ChannelizedError
from pylabrobot.resources.errors import TooLittleLiquidError

from mini_sdl.clock import SimClock
from mini_sdl.lab import Lab, LiquidStation
from mini_sdl.models import DeviceError


async def make_station(stock_volume=1000):
    lab = Lab({"liquid_handler": 1}, SimClock(scale=1e-6))
    lab.clock.start()
    lab.liquid = LiquidStation(lab, ["water"], stock_volume)
    await lab.liquid.setup()
    return lab


def test_stock_refill_preserves_liquid_balance():
    async def go():
        lab = await make_station(100)
        try:
            station = lab.liquid
            first, second = station.new_well(), station.new_well()
            await station.transfer("water", first, 75)
            await station.transfer("water", second, 75)
            assert first.tracker.get_used_volume() == 75
            assert second.tracker.get_used_volume() == 75
            assert station.stock["water"].tracker.get_used_volume() == 25
            assert [event["kind"] for event in lab.incidents] == ["refill"]
            assert not station.lh.head[0].has_tip
        finally:
            await lab.close()
        assert not lab.liquid.lh.setup_finished
    asyncio.run(go())


def test_used_up_tip_rack_is_replaced():
    async def go():
        lab = await make_station()
        try:
            station = lab.liquid
            well = station.new_well()
            for _ in range(97):
                await station.transfer("water", well, 1)
            assert well.tracker.get_used_volume() == 97
            assert station.stock["water"].tracker.get_used_volume() == 903
            assert sum(spot.has_tip() for spot in station.tips.get_all_items()) == 95
            assert [event["kind"] for event in lab.incidents] == ["tip_rack_swap"]
        finally:
            await lab.close()
    asyncio.run(go())


def test_channelized_stock_error_is_retried(monkeypatch):
    async def go():
        lab = await make_station()
        station = lab.liquid
        original = station.lh.backend.aspirate
        calls = 0

        async def once(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ChannelizedError({0: TooLittleLiquidError("test low stock")})
            await original(**kwargs)

        monkeypatch.setattr(station.lh.backend, "aspirate", once)
        try:
            well = station.new_well()
            await station.transfer("water", well, 50)
            assert calls == 2 and well.tracker.get_used_volume() == 50
            assert station.stock["water"].tracker.get_used_volume() == 950
            assert [event["kind"] for event in lab.incidents] == ["refill"]
        finally:
            await lab.close()
    asyncio.run(go())


def test_unrelated_channel_error_is_not_hidden_and_tip_is_released(monkeypatch):
    async def go():
        lab = await make_station()
        station = lab.liquid

        async def broken(**kwargs):
            raise ChannelizedError({0: RuntimeError("test motor failure")})

        monkeypatch.setattr(station.lh.backend, "aspirate", broken)
        try:
            with pytest.raises(ChannelizedError):
                await station.transfer("water", station.new_well(), 50)
            assert lab.incidents == []
            assert not station.lh.head[0].has_tip
            assert station.stock["water"].tracker.get_used_volume() == 1000
        finally:
            await lab.close()
    asyncio.run(go())


def test_plate_and_transfer_capacity_are_checked_before_moving_liquid():
    async def go():
        lab = await make_station()
        try:
            station = lab.liquid
            wells = [station.new_well() for _ in range(96)]
            assert len({well.name for well in wells}) == 96
            with pytest.raises(DeviceError, match="96"):
                station.new_well()
            with pytest.raises(ValueError, match="300"):
                await station.transfer("water", wells[0], 2000)
            await station.transfer("water", wells[0], 300)
            with pytest.raises(DeviceError, match="样品孔容量"):
                await station.transfer("water", wells[0], 100)
            assert wells[0].tracker.get_used_volume() == 300
            assert station.stock["water"].tracker.get_used_volume() == 700
        finally:
            await lab.close()
    asyncio.run(go())


@pytest.mark.parametrize("capacity", [0, -1, 1.5])
def test_invalid_device_capacity_does_not_create_a_deadlock(capacity):
    with pytest.raises(ValueError):
        Lab({"device": capacity}, SimClock())
