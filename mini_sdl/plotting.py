"""绘图只读取仿真记录；时间单位换算只影响显示。"""
import re

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure


def configure_chinese_font():
    """选择本机已有中文字体；不下载字体、不在导入时修改全局设置。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ["PingFang SC", "Heiti TC", "Songti SC", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"]
    fonts = [name for name in preferred if name in available]
    mpl.rcParams["font.sans-serif"] = fonts + ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False


def _figure(height=4.2):
    fig = Figure(figsize=(8, height), layout="constrained")
    return fig, fig.subplots()


def _label(profile):
    return f"{profile.target} / {profile.target_unit}"


def convergence_figure(history, profile):
    """失败样品留空；负目标的累计最大值不会被错误初始化为 0。"""
    fig, ax = _figure()
    x = np.arange(1, len(history) + 1)
    values = np.array([row["result"] if row["status"] == "done" and row["result"] is not None
                       else np.nan for row in history], dtype=float)
    best = np.maximum.accumulate(np.where(np.isfinite(values), values, -np.inf))
    best[~np.isfinite(best)] = np.nan
    ax.scatter(x, values, s=30, color="#26828e", label="样品目标值")
    ax.plot(x, best, color="#244a73", label="迄今最佳")
    ax.set(xlabel="已结束样品序号", ylabel=_label(profile))
    ax.grid(alpha=0.18)
    ax.legend()
    return fig


def parameter_figure(history, profile):
    fig, ax = _figure()
    p, q = profile.params[:2]
    measured, missing = [], []
    for row in history:
        params = row.get("actual") or row["requested"]
        if row["status"] == "done" and row["result"] is not None:
            measured.append((params[p.name], params[q.name], row["result"]))
        else:
            missing.append((params[p.name], params[q.name]))
    if measured:
        data = np.asarray(measured)
        points = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="viridis", s=65)
        fig.colorbar(points, ax=ax, label=_label(profile))
    if missing:
        data = np.asarray(missing)
        ax.scatter(data[:, 0], data[:, 1], marker="x", color="gray", label="无目标值")
        ax.legend()
    ax.set(xlabel=f"{p.name} / {p.unit}", ylabel=f"{q.name} / {q.unit}")
    ax.grid(alpha=0.18)
    return fig


def plate_figure(history, profile):
    fig, ax = _figure(4.4)
    xx, yy = np.meshgrid(np.arange(1, 13), np.arange(8))
    ax.scatter(xx.ravel(), yy.ravel(), s=180, facecolors="none", edgecolors="#cbd5df")
    measured, missing = [], []
    for row in history:
        match = re.search(r"(?:^|_)([A-H])(1[0-2]|[1-9])$", row.get("raw", {}).get("well", ""))
        if not match:
            continue
        location = (int(match[2]), ord(match[1]) - ord("A"))
        if row["status"] == "done" and row["result"] is not None:
            measured.append((*location, row["result"]))
        else:
            missing.append(location)
    if measured:
        data = np.asarray(measured)
        points = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="viridis", s=180)
        fig.colorbar(points, ax=ax, label=_label(profile))
    if missing:
        data = np.asarray(missing)
        ax.scatter(data[:, 0], data[:, 1], marker="x", color="gray", s=100)
    ax.set(xticks=np.arange(1, 13), yticks=np.arange(8), yticklabels=list("ABCDEFGH"),
           xlim=(0.3, 12.7), ylim=(7.7, -0.7), xlabel="孔列", ylabel="孔行")
    return fig


def timeline_figure(timeline):
    samples = list(dict.fromkeys(row["sample"] for row in timeline))
    devices = list(dict.fromkeys(row["device"] for row in timeline))
    fig, ax = _figure(max(3.6, min(20, 1.5 + len(samples) * 0.32)))
    colors = {name: mpl.colormaps["tab10"](i % 10) for i, name in enumerate(devices)}
    seen = set()
    positions = {name: i for i, name in enumerate(samples)}
    for row in timeline:
        if row.get("end") is None:
            continue
        device = row["device"]
        ax.barh(positions[row["sample"]], (row["end"] - row["start"]) / 60,
                left=row["start"] / 60, height=0.65, color=colors[device],
                label=device if device not in seen else None)
        seen.add(device)
    ax.set(yticks=range(len(samples)), yticklabels=samples, xlabel="模拟时间 / min", ylabel="样品")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    if seen:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(3, len(devices)), fontsize=8)
    return fig
