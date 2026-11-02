"""合成测量的反演精度和坏数据边界；不构成真实仪器标定。"""
import numpy as np
import pytest

from mini_sdl.profiles import cavity, tim


@pytest.mark.parametrize("alpha", [2e-7, 5e-7, 1.5e-6])
def test_zero_noise_lfa_recovers_thermal_conductivity(alpha):
    t, v = tim.simulate_lfa(alpha, np.random.default_rng(2), noise=0)
    alpha_est = tim.PARKER * tim.THICKNESS ** 2 / tim.extract_t_half(t, v)
    heat_capacity = tim.rho_cp(0.5)
    assert alpha_est * heat_capacity == pytest.approx(alpha * heat_capacity, rel=0.003)


@pytest.mark.parametrize("t,v", [([], []), ([0, 1], [0, 1]),
                                 (np.arange(20), np.zeros(20)),
                                 (np.arange(20), np.ones(20)),
                                 (np.arange(20), np.full(20, np.nan)),
                                 (np.arange(20)[::-1], np.linspace(0, 1, 20))])
def test_lfa_rejects_invalid_or_missing_half_rise(t, v):
    with pytest.raises(ValueError):
        tim.extract_t_half(t, v)


@pytest.mark.parametrize("eps1,tand", [(2.8, 0.002), (4.0, 0.015)])
def test_zero_noise_cavity_inversion(eps1, tand):
    f, amp = cavity.simulate_s21(eps1, tand, np.random.default_rng(3), noise=0)
    fr, q = cavity.extract_resonance(f, amp)
    recovered_eps, recovered_tand = cavity.eps_from_cavity(fr, q)
    assert recovered_eps == pytest.approx(eps1, rel=1e-9)
    assert recovered_tand == pytest.approx(tand, rel=1e-9)


@pytest.mark.parametrize("f,amp", [([], []), ([1, 2, 3], [0, 0, 0]),
                                   ([1, 2, 3], [1, 1, 1]),
                                   ([1, 2, 3], [1, np.nan, 1]),
                                   ([3, 2, 1], [0.5, 1, 0.5]),
                                   ([1, 2, 3], [0.1, 1, 0.1])])
def test_cavity_rejects_invalid_or_unresolved_peak(f, amp):
    with pytest.raises(ValueError):
        cavity.extract_resonance(f, amp)
