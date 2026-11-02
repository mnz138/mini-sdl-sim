"""检查真实项目函数，不在测试文件里复制业务实现。"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from mini_sdl.models import Sample
from mini_sdl.profiles import electrolyte as el


@pytest.mark.parametrize("c_salt,x_ec", [(0.2, 0.1), (1.1, 0.4), (2.0, 0.9)])
def test_recipe_reports_the_composition_of_transferred_volumes(c_salt, x_ec):
    volumes, actual = el.recipe_to_volumes(c_salt, x_ec)
    total = sum(volumes.values())
    assert set(volumes) == set(el.STOCKS)
    assert 0 < total <= el.TOTAL_UL + 0.2  # 四个体积各舍入到 0.1 µL
    assert all(v == 0 or v >= el.MIN_UL for v in volumes.values())
    assert actual["c_salt"] * total == pytest.approx(
        el.C_STOCK * (volumes["salt_EC"] + volumes["salt_DMC"])
    )
    assert actual["x_ec"] * total == pytest.approx(volumes["salt_EC"] + volumes["EC"])


@pytest.mark.parametrize("c_salt,x_ec", [(-0.1, 0.5), (2.1, 0.5), (1.0, -0.1),
                                        (1.0, 1.1), (np.nan, 0.5), (1.0, np.inf)])
def test_invalid_recipe_is_rejected(c_salt, x_ec):
    with pytest.raises(ValueError):
        el.recipe_to_volumes(c_salt, x_ec)


@pytest.mark.parametrize("r_bulk", [80.0, 150.0, 300.0, 600.0])
def test_zero_noise_eis_resistance_and_conductivity_units(r_bulk):
    f, z = el.simulate_eis(r_bulk, np.random.default_rng(1), noise=0)
    rb = el.extract_rb(f, z)
    # 谷值法是近似反演；零噪声也不要求数值恰好等于设定的 R_b。
    assert rb == pytest.approx(r_bulk, rel=0.01)
    assert 1000 * el.K_CELL / rb == pytest.approx(1000 * el.K_CELL / r_bulk, rel=0.01)


@pytest.mark.parametrize("f,z", [([], []), ([1, 2], [1 + 0j]),
                                 ([1, 2], [1 + 0j, np.nan + 1j]),
                                 ([2, 1], [1 - 1j, 2 - 2j])])
def test_invalid_eis_trace_is_rejected(f, z):
    with pytest.raises(ValueError):
        el.extract_rb(f, z)


def test_measure_converts_unrounded_bulk_resistance_to_millisiemens_per_cm(monkeypatch):
    """固定阻抗谱，只验证测量流程中的单位换算与记录，避免噪声掩盖错误。"""
    async def no_wait(_seconds):
        pass

    sample = Sample("S1", {"c_salt": 1.0, "x_ec": 0.5})
    sample.actual = sample.requested.copy()
    lab = SimpleNamespace(rng=np.random.default_rng(0), p_timeout=0,
                          clock=SimpleNamespace(sleep=no_wait))
    monkeypatch.setattr(el, "simulate_eis", lambda *args: (np.array([1., 2., 3.]),
                                                        np.array([60 - 4j, 83.456 - 1j, 90 - 3j])))
    asyncio.run(el.Electrolyte().measure(sample, lab))
    assert sample.raw["R_b_ohm"] == 83.46
    assert sample.result == pytest.approx(1000 * el.K_CELL / 83.456)
