import pytest

from mini_sdl.models import InvalidTransition, Sample, Status
from mini_sdl.profiles.electrolyte import MIN_UL, TOTAL_UL, recipe_to_volumes


def test_volumes_add_up_and_reproduce_recipe():
    vols, actual = recipe_to_volumes(1.0, 0.5)
    assert sum(vols.values()) == pytest.approx(TOTAL_UL)
    assert actual["c_salt"] == pytest.approx(1.0)
    assert actual["x_ec"] == pytest.approx(0.5)


def test_tiny_volumes_are_dropped_and_actual_recipe_reported():
    # c=1.96 时纯溶剂只需各 2 µL，低于最小移液量，被舍掉；实际浓度变成 2.0
    vols, actual = recipe_to_volumes(1.96, 0.5)
    assert vols["EC"] == 0 and vols["DMC"] == 0
    assert all(v == 0 or v >= MIN_UL for v in vols.values())
    assert actual["c_salt"] == pytest.approx(2.0)


def test_out_of_range_recipe_is_rejected():
    with pytest.raises(ValueError):
        recipe_to_volumes(2.5, 0.5)


def test_state_machine_blocks_illegal_jump():
    s = Sample("S1", {})
    with pytest.raises(InvalidTransition):
        s.move_to(Status.DONE, 0.0)        # 还没配制就宣布完成，不允许
    s.move_to(Status.PREPARING, 0.0)
    s.move_to(Status.PREPARED, 1.0)
    s.move_to(Status.MEASURING, 2.0)
    s.move_to(Status.PREPARED, 3.0)        # 测量失败退回
    s.move_to(Status.MEASURING, 4.0)
    s.move_to(Status.DONE, 5.0)
    assert [e["to"] for e in s.events][-1] == "done"
