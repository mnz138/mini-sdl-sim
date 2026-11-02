"""Streamlit 演示入口：.venv/bin/python -m streamlit run app.py。"""
import asyncio
import json
from pathlib import Path
from time import time_ns

import pandas as pd
import streamlit as st

from mini_sdl.campaign import run_campaign
from mini_sdl.optimizer import OPTIMIZERS
from mini_sdl.plotting import (
    configure_chinese_font,
    convergence_figure,
    parameter_figure,
    plate_figure,
    timeline_figure,
)
from mini_sdl.profiles import PROFILES

PROFILE_LABELS = {
    "electrolyte": "电解液 · 电导率",
    "tim": "导热胶 · 导热与粘度",
    "cavity": "微波谐振腔 · Dk/Df",
}


def render_result(result):
    """所有标签都使用本次结果的配置，侧栏修改不会重标旧数据。"""
    profile = PROFILES[result["config"]["profile"]]()
    history = result["history"]
    done = [row for row in history if row["status"] == "done" and row["result"] is not None]
    best = max((row["result"] for row in done), default=None)
    st.caption(
        f"当前结果：{PROFILE_LABELS[profile.name]} · {result['config']['optimizer']} · "
        f"{result['config']['mode']} · seed={result['config']['seed']}。"
        "修改侧栏后，点击运行才会生成新结果。"
    )
    columns = st.columns(4)
    columns[0].metric("已结束样品", len(history))
    columns[1].metric("设备失败", sum(row["status"] == "failed" for row in history))
    columns[2].metric("迄今最佳目标值", "暂无" if best is None else f"{best:.4g}")
    columns[3].metric("模拟流程时间 / min", f"{result['elapsed_s'] / 60:.2f}")
    st.caption(f"优化目标：{profile.target}，单位：{profile.target_unit}；所有配置均求最大值。")
    if profile.name == "tim":
        st.caption("目标值含粘度惩罚；反演的导热系数见样品记录 raw.k_W_mK。")
    elif profile.name == "cavity":
        st.caption("目标为 −1000 × Df 再扣除 Dk、粘度超标惩罚；负分正常，Dk/Df 见样品记录。")

    tabs = st.tabs(["收敛曲线", "参数空间", "96 孔板", "调度时间线", "样品记录", "异常事件"])
    with tabs[0]:
        if done:
            st.pyplot(convergence_figure(history, profile), width="stretch")
            st.caption("横轴按样品记录顺序排列；设备失败不作为低分回流。单次运行不用于证明优化器优劣。")
        else:
            st.info("尚无完成测量的目标值，不能绘制收敛曲线。")
    with tabs[1]:
        if history:
            st.pyplot(parameter_figure(history, profile), width="stretch")
            st.caption("优先绘制实际配方；尚无实际配方时使用请求配方。灰色叉号表示没有目标值。")
        else:
            st.info("暂无样品记录。")
    with tabs[2]:
        if profile.name == "electrolyte":
            st.pyplot(plate_figure(history, profile), width="stretch")
            st.caption("圆点颜色表示目标值；灰色叉号为已分配但测量失败的孔，空心圆为未使用孔。")
        else:
            st.info("该配置使用固体样品流程，没有 96 孔移液板。")
    with tabs[3]:
        if result["timeline"]:
            st.pyplot(timeline_figure(result["timeline"]), width="stretch")
            st.caption("时间轴为加速时钟中的模拟分钟，不是本机实际运行耗时；颜色表示设备。")
        else:
            st.info("暂无设备操作记录。")
    with tabs[4]:
        if history:
            records = pd.json_normalize(history)
            if "events" in records:
                records["events"] = records["events"].map(lambda value: json.dumps(value, ensure_ascii=False))
            st.dataframe(records, width="stretch", hide_index=True)
            st.download_button(
                "下载样品 JSONL",
                "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in history) + "\n",
                file_name=Path(result["log_path"]).name,
                mime="application/x-ndjson",
            )
            st.caption(f"本机记录：{result['log_path']}")
        else:
            st.info("暂无样品记录。")
    with tabs[5]:
        if result["incidents"]:
            st.dataframe(pd.DataFrame(result["incidents"]), width="stretch", hide_index=True)
            st.caption("t 的单位为模拟秒。补液、换枪头盒和测量重试均由实际运行产生。")
        else:
            st.info("本次运行没有异常事件。")


def main():
    st.set_page_config(page_title="mini-sdl-sim", layout="wide")
    configure_chinese_font()
    st.title("自驱动实验室仿真")
    st.caption("个人学习项目 · 基于 PyLabRobot 与 BayBE · 全程仿真，目标函数为合成。")
    with st.sidebar:
        st.subheader("运行配置")
        with st.form("campaign_config"):
            profile_name = st.selectbox("实验配置", list(PROFILES), format_func=PROFILE_LABELS.get, key="profile")
            optimizer_name = st.selectbox("优化器", ["random", "bayes"], key="optimizer")
            rounds = st.number_input("轮数", min_value=1, max_value=12, value=2, step=1, key="rounds")
            batch = st.number_input("每轮样品数", min_value=1, max_value=16, value=2, step=1, key="batch")
            seed = st.number_input("随机种子", min_value=0, max_value=2**32 - 1, value=0, step=1, key="seed")
            p_timeout = st.slider("每次测量故障概率", 0.0, 1.0, 0.0, 0.05, key="p_timeout")
            mode = st.selectbox("调度模式", ["pipeline", "sequential"], key="mode")
            submitted = st.form_submit_button("运行仿真", type="primary")
        st.caption("默认先运行 4 个随机样品。BayBE 首次导入和后续推荐可能需要更长时间。")

    if submitted:
        if profile_name == "electrolyte" and rounds * batch > 96:
            st.error("电解液单次最多使用 96 个孔，请减小轮数或每轮样品数。")
        else:
            config = dict(profile=profile_name, optimizer=optimizer_name, rounds=rounds,
                          batch_size=batch, seed=seed, p_timeout=p_timeout, mode=mode)
            log_path = Path(__file__).resolve().parent / "data" / f"web-{profile_name}-{optimizer_name}-{time_ns()}.jsonl"
            progress = st.progress(0, text="正在初始化仿真实验室…")
            try:
                profile = PROFILES[profile_name]()
                optimizer = OPTIMIZERS[optimizer_name](profile.params, profile.target, seed=seed)
                with st.spinner("正在运行仿真…"):
                    history, lab = asyncio.run(run_campaign(
                        profile, optimizer, rounds=rounds, batch_size=batch, seed=seed,
                        p_timeout=p_timeout, mode=mode, log_path=log_path, verbose=False,
                        on_round=lambda r, *_: progress.progress(r / rounds, text=f"已完成 {r}/{rounds} 轮"),
                    ))
                st.session_state["result"] = {
                    "config": config, "history": history, "timeline": lab.timeline,
                    "incidents": lab.incidents, "elapsed_s": lab.clock.now(), "log_path": str(log_path),
                }
                st.success("本次仿真已结束，结果如下。")
            except Exception as exc:
                st.error(f"本次运行未完成：{type(exc).__name__}: {exc}")
                if log_path.exists():
                    st.caption(f"已写出的样品记录保存在 {log_path}")
            finally:
                progress.empty()

    if "result" in st.session_state:
        render_result(st.session_state["result"])
    else:
        st.info("在侧栏选择配置并点击“运行仿真”。页面不会自动启动实验。")


if __name__ == "__main__":
    main()
