import numpy as np
import pytest

from mini_sdl.profiles import electrolyte as el
from mini_sdl.profiles import tim


@pytest.mark.parametrize("r_bulk", [80.0, 150.0, 300.0, 600.0])
def test_eis_extraction_recovers_bulk_resistance(r_bulk):
    f, z = el.simulate_eis(r_bulk, np.random.default_rng(1))
    assert el.extract_rb(f, z) == pytest.approx(r_bulk, rel=0.01)


@pytest.mark.parametrize("alpha", [2e-7, 5e-7, 1.5e-6])
def test_laser_flash_extraction_recovers_diffusivity(alpha):
    t, v = tim.simulate_lfa(alpha, np.random.default_rng(2))
    alpha_est = tim.PARKER * tim.THICKNESS ** 2 / tim.extract_t_half(t, v)
    assert alpha_est == pytest.approx(alpha, rel=0.03)


def test_composite_models_reduce_to_matrix_without_filler():
    assert tim.lewis_nielsen(0.0, 0.7) == pytest.approx(tim.K_MATRIX)
    assert tim.krieger_dougherty(0.0, 0.7) == pytest.approx(tim.ETA0)


def test_conductivity_peaks_at_mu():
    c = np.linspace(0.2, 2.0, 181)
    k = el.casteel_amis(c, kappa_max=10.0, mu=1.1)
    assert c[np.argmax(k)] == pytest.approx(1.1, abs=0.01)
    assert k.max() == pytest.approx(10.0, rel=1e-3)
