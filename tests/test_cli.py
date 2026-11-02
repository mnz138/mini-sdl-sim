"""以真实子进程验证命令行，包括所有样品测量失败的边界。"""
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_cli_all_failed_still_writes_records_and_summary(tmp_path):
    output = tmp_path / "failed.jsonl"
    result = subprocess.run([
        sys.executable, str(ROOT / "run_campaign.py"),
        "--optimizer", "random", "--rounds", "1", "--batch", "2",
        "--p-timeout", "1", "--output", str(output),
    ], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "没有成功完成测量的样品" in result.stdout
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2 and all(row["status"] == "failed" for row in rows)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["best"] is None
    assert summary["done"] == 0 and summary["failed"] == 2
    assert summary["simulation"] and summary["synthetic_objective"]


def test_cli_rejects_invalid_input_before_running(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "run_campaign.py"), "--rounds", "0",
    ], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "正整数" in result.stderr
    assert not (tmp_path / "data").exists()
