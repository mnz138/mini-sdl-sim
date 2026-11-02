"""验证闭环的结果回流、持久化和失败处理。"""
import asyncio
import json

import pytest

from mini_sdl.campaign import run_campaign
from mini_sdl.optimizer import RandomOptimizer
from mini_sdl.profiles import PROFILES


class RecordingOptimizer(RandomOptimizer):
    def __init__(self, profile):
        super().__init__(profile.params, profile.target, seed=7)
        self.feedback = []

    def tell(self, rows):
        self.feedback.extend(rows)


@pytest.mark.parametrize("profile_name", list(PROFILES))
def test_closed_loop_records_actual_recipes_and_feedback(profile_name, tmp_path):
    profile = PROFILES[profile_name]()
    optimizer = RecordingOptimizer(profile)
    path = tmp_path / "nested" / "samples.jsonl"
    history, lab = asyncio.run(run_campaign(
        profile, optimizer, rounds=2, batch_size=2, p_timeout=0,
        log_path=path, verbose=False, scale=1e-5,
    ))
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert saved == history
    assert len(saved) == 4
    assert len({row["sample_id"] for row in saved}) == 4
    assert all(row["status"] == "done" for row in saved)
    assert optimizer.feedback == [
        {**row["actual"], profile.target: row["result"]} for row in saved
    ]
    assert lab.timeline and all(event["end"] >= event["start"] for event in lab.timeline)


def test_failed_measurements_are_logged_without_fake_optimizer_scores(tmp_path):
    profile = PROFILES["electrolyte"]()
    optimizer = RecordingOptimizer(profile)
    path = tmp_path / "failed.jsonl"
    history, lab = asyncio.run(run_campaign(
        profile, optimizer, rounds=1, batch_size=2, p_timeout=1,
        log_path=path, verbose=False, scale=1e-5,
    ))
    assert len(history) == 2
    assert all(row["status"] == "failed" and row["attempts"] == 3 for row in history)
    assert all(row["result"] is None for row in history)
    assert optimizer.feedback == []
    assert len(lab.incidents) == 6
    assert len(path.read_text().splitlines()) == 2


def test_existing_experiment_log_is_not_overwritten(tmp_path):
    path = tmp_path / "existing.jsonl"
    original = '{"previous_experiment": true}\n'
    path.write_text(original)
    profile = PROFILES["electrolyte"]()
    with pytest.raises(FileExistsError):
        asyncio.run(run_campaign(profile, RecordingOptimizer(profile), log_path=path))
    assert path.read_text() == original


def test_results_survive_an_optimizer_feedback_error(tmp_path, monkeypatch):
    import mini_sdl.campaign as campaign

    closed = []
    original_make_lab = campaign.make_lab

    async def tracked_make_lab(*args, **kwargs):
        lab = await original_make_lab(*args, **kwargs)
        close = lab.close

        async def tracked_close():
            await close()
            closed.append(True)

        lab.close = tracked_close
        return lab

    monkeypatch.setattr(campaign, "make_lab", tracked_make_lab)
    profile = PROFILES["electrolyte"]()
    optimizer = RecordingOptimizer(profile)

    def broken_feedback(rows):
        raise RuntimeError("optimizer unavailable")

    optimizer.tell = broken_feedback
    path = tmp_path / "recoverable.jsonl"
    with pytest.raises(RuntimeError, match="optimizer unavailable"):
        asyncio.run(run_campaign(
            profile, optimizer, rounds=1, batch_size=2, p_timeout=0,
            log_path=path, verbose=False, scale=1e-5,
        ))
    assert len(path.read_text().splitlines()) == 2
    assert closed == [True]


@pytest.mark.parametrize("kwargs", [
    {"rounds": 0}, {"batch_size": 0}, {"p_timeout": 1.1},
    {"mode": "invalid"}, {"rounds": 13, "batch_size": 8},
])
def test_invalid_campaign_is_rejected(kwargs):
    profile = PROFILES["electrolyte"]()
    with pytest.raises(ValueError):
        asyncio.run(run_campaign(profile, RecordingOptimizer(profile), **kwargs))
