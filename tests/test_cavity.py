import numpy as np
import pytest
from mini_sdl.profiles import cavity as cv

@pytest.mark.parametrize("eps1,tand", [(2.8, 0.002), (3.2, 0.0008), (3.5, 0.005), (4.0, 0.015)])
def test_cavity_recovers_dk_df(eps1, tand):
    fr, q = cv.extract_resonance(*cv.simulate_s21(eps1, tand, np.random.default_rng(3)))
    e, t = cv.eps_from_cavity(fr, q)
    assert e == pytest.approx(eps1, rel=0.002)
    assert t == pytest.approx(tand, rel=0.05)
