"""真实命令行验收：三种配置 × 两种优化器，并记录流水线耗时。

从项目目录运行：.venv/bin/python scripts/validate_project.py
加 --full 额外执行电解液 BayBE 6 轮 × 8 个样品。
"""
import argparse
import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from mini_sdl.campaign import make_lab
from mini_sdl.models import Sample
from mini_sdl.profiles import PROFILES
from mini_sdl.scheduler import run_batch


ROOT = Path(__file__).resolve().parents[1]


def cli_case(folder, profile, optimizer, rounds=2, batch=2):
    name = f"{profile}_{optimizer}_{rounds}x{batch}"
    output = folder / f"{name}.jsonl"
    command = [sys.executable, str(ROOT / "run_campaign.py"),
               "--profile", profile, "--optimizer", optimizer,
               "--rounds", str(rounds), "--batch", str(batch), "--seed", "0",
               "--p-timeout", "0", "--output", str(output)]
    print(f"验证 {name}", flush=True)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    (folder / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{name} 失败，请查看 {folder / (name + '.log')}")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == rounds * batch
    assert len({row["sample_id"] for row in rows}) == len(rows)
    assert all(row["status"] == "done" and math.isfinite(row["result"]) for row in rows)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    return {"case": name, "samples": len(rows), "best": summary["best"]["result"],
            "unit": summary["target_unit"], "wall_seconds": summary["wall_seconds"],
            "incidents": sorted({event["kind"] for event in summary["incidents"]}),
            "jsonl": str(output.relative_to(ROOT))}


async def scheduling_case(mode, n_cells):
    profile = PROFILES["electrolyte"](n_cells=n_cells)
    lab = await make_lab(profile, p_timeout=0)
    recipes = [{"c_salt": 0.5 + 0.15 * i, "x_ec": 0.3 + 0.05 * i} for i in range(8)]
    samples = [Sample(f"S{i}", recipe) for i, recipe in enumerate(recipes)]
    try:
        await run_batch(samples, profile, lab, mode)
        assert all(sample.status.value == "done" for sample in samples)
        elapsed = max(event["end"] for event in lab.timeline)
        return {"mode": mode, "conductivity_cells": n_cells,
                "samples": len(samples), "simulated_seconds": elapsed,
                "simulated_minutes": elapsed / 60}
    finally:
        await lab.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
    folder = ROOT / "data" / f"validation-{stamp}"
    folder.mkdir(parents=True)
    cases = [cli_case(folder, profile, optimizer)
             for profile in PROFILES for optimizer in ["random", "bayes"]]
    if args.full:
        cases.append(cli_case(folder, "electrolyte", "bayes", rounds=6, batch=8))
    scheduling = [asyncio.run(scheduling_case(mode, cells))
                  for mode, cells in [("sequential", 1), ("pipeline", 1), ("pipeline", 2)]]
    seq, pipe, two = [row["simulated_seconds"] for row in scheduling]
    assert pipe < 0.85 * seq and two < 0.75 * pipe
    report = {"scope": "synthetic simulation; fixed seed 0; not an optimizer benchmark",
              "cli_cases": cases, "scheduling": scheduling}
    path = folder / "validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"验收记录：{path}")


if __name__ == "__main__":
    main()
