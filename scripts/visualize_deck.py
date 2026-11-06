"""在浏览器里查看 PyLabRobot 仿真工作台（学习手册图 6-2）。

从项目目录运行：.venv/bin/python scripts/visualize_deck.py
浏览器打开终端打印的地址（默认 http://127.0.0.1:1337，端口被占用时会顺延）。按 Ctrl+C 退出。

流程：建电解液实验室 → 启动 Visualizer → 用真实调度器配制并测量一批样品 → 保持服务供浏览器查看。
显示的是软件资源树和枪头/液量 tracker 的状态，不是机器人运动或吸液物理仿真；数据均为合成。
"""
import argparse
import asyncio

from pylabrobot.visualizer import Visualizer

from mini_sdl.campaign import make_lab
from mini_sdl.models import Sample, Status
from mini_sdl.optimizer import RandomOptimizer
from mini_sdl.profiles import PROFILES
from mini_sdl.scheduler import run_batch


async def main(args) -> None:
    profile = PROFILES["electrolyte"]()
    # p_timeout=0：这里只演示配液后的工作台状态，不注入测量故障。
    lab = await make_lab(profile, seed=args.seed, p_timeout=0.0, scale=args.scale)
    vis = Visualizer(resource=lab.liquid.lh.deck, open_browser=not args.no_browser,
                     name="mini-sdl-sim 电解液工作台（仿真）")
    await vis.setup()
    try:
        await asyncio.sleep(args.wait)   # 给浏览器留出连接时间，便于看到配液过程
        recs = RandomOptimizer(profile.params, profile.target, seed=args.seed).ask(args.samples)
        samples = [Sample(f"V-S{i:02d}", {k: float(v) for k, v in rec.items()})
                   for i, rec in enumerate(recs, 1)]
        await run_batch(samples, profile, lab, "pipeline")
        for s in samples:
            print(f"{s.sample_id}  {s.raw.get('well', '-'):>16}  {s.status.value:<6}  "
                  f"{s.raw.get('volumes_uL')}")
        tips_left = sum(spot.has_tip() for spot in lab.liquid.tips.get_all_items())
        done = sum(s.status == Status.DONE for s in samples)
        print(f"完成 {done}/{len(samples)}；枪头剩余 {tips_left}/{lab.liquid.tips.num_items}；"
              f"异常事件 {len(lab.incidents)} 次")
        if args.hold > 0:
            print(f"工作台页面保留 {args.hold:g} s。")
            await asyncio.sleep(args.hold)
        else:
            print("工作台页面保持打开，按 Ctrl+C 退出。")
            await asyncio.Event().wait()
    finally:
        await vis.stop()
        await lab.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=8, help="配制的样品数（1–96），默认 8 个即孔板第一列")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.01,
                    help="1 个模拟秒对应的真实秒数；默认 0.01，便于在浏览器里看到配液过程")
    ap.add_argument("--wait", type=float, default=3.0, help="开始配液前等待浏览器连接的秒数")
    ap.add_argument("--hold", type=float, default=0.0, help="配完后保留页面的秒数；0 表示直到 Ctrl+C")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()
    if not 1 <= args.samples <= 96:
        ap.error("--samples 必须在 1 和 96 之间")
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("已退出。")
