"""测试实际样品类，不在测试里复制实现。"""
import json

import pytest

from mini_sdl.models import InvalidTransition, Sample, Status


def test_sample_lifecycle_and_json_record():
    sample = Sample("S1", {"c_salt": 1.0})
    for t, status in enumerate((Status.PREPARING, Status.PREPARED, Status.MEASURING, Status.DONE)):
        sample.move_to(status, float(t))
    sample.attempts = 1
    record = json.loads(json.dumps(sample.record()))
    assert record["status"] == "done"
    assert record["attempts"] == 1
    assert [event["to"] for event in record["events"]] == [
        "preparing", "prepared", "measuring", "done"
    ]


def test_cannot_skip_preparation():
    sample = Sample("S1", {})
    with pytest.raises(InvalidTransition):
        sample.move_to(Status.MEASURING, 0.0)
    assert sample.status == Status.PLANNED and sample.events == []


@pytest.mark.parametrize("terminal", [Status.DONE, Status.FAILED])
def test_terminal_status_cannot_restart(terminal):
    sample = Sample("S1", {}, status=terminal)
    with pytest.raises(InvalidTransition):
        sample.move_to(Status.PREPARING, 0.0)


def test_record_is_a_snapshot_and_sample_lists_are_independent():
    first, second = Sample("S1", {"x": 1}), Sample("S2", {})
    first.raw["volumes"] = {"stock": 50.0}
    record = first.record()
    first.raw["volumes"]["stock"] = 100.0
    first.move_to(Status.PREPARING, 0.0)
    assert record["raw"]["volumes"]["stock"] == 50.0
    assert record["events"] == []
    assert second.events == [] and second.raw == {}
