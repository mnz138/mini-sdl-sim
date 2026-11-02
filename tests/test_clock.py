import pytest

from mini_sdl.clock import SimClock


def test_pause_excludes_optimizer_and_readout_time(monkeypatch):
    real = [10.0]
    monkeypatch.setattr("mini_sdl.clock.time.monotonic", lambda: real[0])
    clock = SimClock(scale=0.1)
    clock.start()
    real[0] += 2.0
    assert clock.now() == pytest.approx(20.0)
    clock.pause()
    real[0] += 100.0
    assert clock.now() == pytest.approx(20.0)
    clock.pause()
    clock.resume()
    clock.resume()
    real[0] += 1.0
    assert clock.now() == pytest.approx(30.0)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        SimClock(scale)
