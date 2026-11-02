import math

import pytest

from mini_sdl.optimizer import BayesOptimizer, RandomOptimizer
from mini_sdl.profiles.tim import ThermalInterface


def test_random_search_is_seeded_and_within_requested_bounds():
    profile = ThermalInterface()
    first = RandomOptimizer(profile.params, profile.target, seed=11).ask(5)
    second = RandomOptimizer(profile.params, profile.target, seed=11).ask(5)
    assert first == second
    assert all(p.low <= row[p.name] <= p.high for row in first for p in profile.params)


def test_baybe_preserves_actual_recipe_even_outside_recommendation_bounds():
    profile = ThermalInterface()
    optimizer = BayesOptimizer(profile.params, profile.target, seed=0)
    row = {"phi": 0.751, "r_small": 0.3, "score": 1.2}
    optimizer.tell([row])
    actual = optimizer.campaign.measurements.iloc[0]
    assert actual["phi"] == 0.751  # 不允许把实际 0.751 偷换成上限 0.75。
    assert actual["score"] == 1.2
    with pytest.raises(ValueError, match="有限数值"):
        optimizer.tell([{**row, "score": math.nan}])
