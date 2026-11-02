"""网页验收：不会自动运行，真实随机小闭环可显示，失败/负分不误导。"""
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from mini_sdl.plotting import convergence_figure
from mini_sdl.profiles import PROFILES

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_page_waits_for_user_and_runs_small_random_campaign():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "result" not in app.session_state
    app.number_input(key="rounds").set_value(1)
    app.number_input(key="batch").set_value(2)
    app.button[0].click().run()
    result = app.session_state["result"]
    try:
        assert not app.exception
        assert len(result["history"]) == 2
        assert all(row["status"] == "done" for row in result["history"])
        assert len(app.tabs) == 6
        assert result["timeline"]
        assert Path(result["log_path"]).exists()
        # 更改侧栏时，已完成结果仍保持电解液标签与单位。
        app.selectbox(key="profile").select("cavity").run()
        assert not app.exception
        assert any("mS/cm" in caption.value for caption in app.caption)
    finally:
        Path(result["log_path"]).unlink(missing_ok=True)


def _result(history):
    return {
        "config": {"profile": "cavity", "optimizer": "random", "mode": "pipeline", "seed": 0},
        "history": history, "timeline": [], "incidents": [], "elapsed_s": 0,
        "log_path": "test-only.jsonl",
    }


def test_empty_failed_and_negative_scores_display_without_zero_best():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.session_state["result"] = _result([])
    app.run()
    assert not app.exception
    assert app.metric[2].value == "暂无"
    row = dict(sample_id="test", round=1, requested={"phi": 0.3, "h": 0.2},
               actual=None, status="failed", result=None, raw={}, events=[], feasible=True, attempts=3)
    app.session_state["result"] = _result([row])
    app.run()
    assert not app.exception
    assert app.metric[1].value == "1"
    assert app.metric[2].value == "暂无"
    completed = {**row, "status": "done", "result": -12.5, "actual": row["requested"]}
    app.session_state["result"] = _result([row, completed])
    app.run()
    assert not app.exception
    assert app.metric[2].value == "-12.5"
    assert any("没有 96 孔" in item.value for item in app.info)
    fig = convergence_figure([row, completed], PROFILES["cavity"]())
    best = fig.axes[0].lines[0].get_ydata()
    assert np.isnan(best[0])
    assert best[1] == -12.5
