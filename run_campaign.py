"""命令行入口。例：python run_campaign.py --profile electrolyte --optimizer bayes --rounds 6 --batch 8"""
import argparse
import asyncio
import json
import platform
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

from mini_sdl.campaign import run_campaign
from mini_sdl.optimizer import OPTIMIZERS
from mini_sdl.profiles import PROFILES


def main() -> None:
    ap = argparse.ArgumentParser(description="自驱动实验室仿真：闭环配方筛选")
    ap.add_argument("--profile", choices=PROFILES, default="electrolyte")
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="bayes")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-timeout", type=float, default=0.1, help="测量超时的注入概率")
    ap.add_argument("--mode", choices=["pipeline", "sequential"], default="pipeline")
    ap.add_argument("--output", type=Path, help="JSONL 保存位置；省略时生成带时间戳的文件名")
    args = ap.parse_args()
    if args.rounds < 1 or args.batch < 1:
        ap.error("--rounds 和 --batch 必须为正整数")
    if not 0 <= args.p_timeout <= 1:
        ap.error("--p-timeout 必须在 0 和 1 之间")
    if args.profile == "electrolyte" and args.rounds * args.batch > 96:
        ap.error("电解液单次实验最多 96 个样品")

    profile = PROFILES[args.profile]()
    optimizer = OPTIMIZERS[args.optimizer](profile.params, profile.target, seed=args.seed)
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
    log_path = args.output or Path("data") / f"{args.profile}_{args.optimizer}_{args.mode}_seed{args.seed}_{timestamp}.jsonl"
    summary_path = log_path.with_suffix(".summary.json")
    if log_path.exists() or summary_path.exists():
        ap.error("输出文件已存在，请选择新路径；已有实验记录不会被覆盖")
    started = time.perf_counter()
    history, lab = asyncio.run(run_campaign(profile, optimizer, args.rounds, args.batch, args.seed,
                                            args.p_timeout, args.mode, log_path))
    done = [h for h in history if h["status"] == "done"]
    best = max(done, key=lambda h: h["result"]) if done else None
    summary = {
        "simulation": True, "synthetic_objective": True,
        "profile": args.profile, "optimizer": args.optimizer, "mode": args.mode,
        "seed": args.seed, "rounds": args.rounds, "batch_size": args.batch,
        "p_timeout": args.p_timeout, "target": profile.target, "target_unit": profile.target_unit,
        "clock_scale": 1e-3, "python": platform.python_version(),
        "samples": len(history), "done": len(done), "failed": len(history) - len(done),
        "best": best,
        "simulated_seconds": max((e["end"] for e in lab.timeline), default=0.0),
        "wall_seconds": time.perf_counter() - started,
        "versions": {p: version(p) for p in ["pylabrobot", "baybe", "numpy", "torch", "pyarrow", "streamlit"]},
        "timeline": lab.timeline, "incidents": lab.incidents,
    }
    with summary_path.open("x", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, allow_nan=False, indent=2)
    if best is not None:
        print(f"\n最好配方 {best['sample_id']}：{best['actual']}，{profile.target} = {best['result']:.2f} {profile.target_unit}")
    else:
        print("\n没有成功完成测量的样品，请查看故障记录。")
    print(f"异常事件 {len(lab.incidents)} 次：" + "、".join(sorted({i['kind'] for i in lab.incidents})))
    print(f"记录已写入 {log_path}")
    print(f"运行摘要已写入 {summary_path}")


if __name__ == "__main__":
    main()
