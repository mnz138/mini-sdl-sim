"""低损耗基板介电膜（封装基板材料）：二氧化硅填料含量 × 空心玻璃微珠比例 → 介电损耗 Df 尽量低。

测量：微波谐振腔微扰法（IPC-TM-650 2.5.5.13 的思路），网络分析仪测 S21 谐振峰的频率和 Q 值。
truth() 是合成函数：Lichtenecker 对数混合律算 Dk、体积加权算 Df，参数仅供演示。
"""
import numpy as np

from ..models import DeviceError
from .base import Param, Profile, Step
from .tim import krieger_dougherty

F0, Q0, FILL = 10.04e9, 20000.0, 0.01        # 空腔谐振频率 Hz、空腔 Q、样品填充因子（假设值）
EPS = {"resin": 3.6, "silica": 3.8, "hollow": 1.4}     # 各组分介电常数（演示值）
TAN = {"resin": 0.020, "silica": 0.0003, "hollow": 0.002}  # 各组分损耗角正切（演示值）
DK_MAX = 3.2        # 设计要求：介电常数不超过 3.2
ETA_MAX = 150.0     # 涂膜工艺能接受的最大粘度，Pa·s
PHI_M = 0.64        # 最大堆积分数


def dk_df(phi: float, h: float) -> tuple:
    v = {"resin": 1 - phi, "silica": phi * (1 - h), "hollow": phi * h}
    dk = float(np.exp(sum(v[k] * np.log(EPS[k]) for k in v)))
    df = float(sum(v[k] * TAN[k] for k in v))
    return dk, df


def score(dk: float, df: float, eta: float) -> float:
    """目标越大越好：-1000×Df，Dk 超标和粘度超标都扣分。"""
    return -1000 * df - 50 * max(0.0, dk - DK_MAX) - 3 * max(0.0, float(np.log(eta / ETA_MAX)))


def simulate_s21(eps1: float, tand: float, rng, noise: float = 0.002):
    fs = F0 * (1 - FILL * (eps1 - 1) / 2)
    qs = 1 / (1 / Q0 + FILL * eps1 * tand)
    f = np.linspace(fs - 6 * fs / qs, fs + 6 * fs / qs, 601)
    amp = 1 / np.sqrt(1 + 4 * qs ** 2 * ((f - fs) / fs) ** 2)
    return f, amp * (1 + noise * rng.standard_normal(f.size))


def extract_resonance(f, amp) -> tuple:
    f, amp = np.asarray(f, dtype=float), np.asarray(amp, dtype=float)
    if (f.ndim != 1 or amp.ndim != 1 or f.size != amp.size or f.size < 3
            or not np.all(np.isfinite(f)) or not np.all(np.isfinite(amp))
            or np.any(f <= 0) or np.any(np.diff(f) <= 0) or np.any(amp <= 0)):
        raise ValueError("S21 需要至少 3 个有限数据点，频率递增且频率和幅度均为正")
    m = amp >= 0.5 * amp.max()
    if np.count_nonzero(m) < 3 or np.ptp(amp) == 0:
        raise ValueError("S21 峰附近数据不足，无法拟合谐振频率和 Q")
    fc = f[m].mean()
    x = (f[m] - fc) / 1e6
    p2, p1, p0 = np.polyfit(x, 1 / amp[m] ** 2, 2)
    if not np.all(np.isfinite([p2, p1, p0])) or p2 <= 0:
        raise ValueError("S21 拟合曲率必须为正")
    xc = -p1 / (2 * p2)
    cmin = p0 - p1 ** 2 / (4 * p2)
    fr = fc + xc * 1e6
    if not np.all(np.isfinite([p2, cmin, fr])) or p2 <= 0 or cmin <= 0 or not f[0] < fr < f[-1]:
        raise ValueError("S21 曲线没有可辨识的正曲率谐振峰")
    return float(fr), float(fr / 2e6 * np.sqrt(p2 / cmin))


def eps_from_cavity(fr: float, q: float) -> tuple:
    eps1 = 1 + 2 * (F0 - fr) / (FILL * F0)
    return eps1, (1 / q - 1 / Q0) / (FILL * eps1)


class Cavity(Profile):
    name = "cavity"
    target = "score"
    target_unit = "（−1000×Df）"
    params = [Param("phi", 0.20, 0.62, "体积分数", "二氧化硅填料体积分数"),
              Param("h", 0.0, 0.6, "比例", "填料中空心玻璃微珠的比例")]
    infeasible_value = -40.0

    def __init__(self):
        self.capacities = {"dispenser": 1, "mixer": 1, "oven": 8, "cavity": 1}
        self.workflow = [Step("称量投料", "dispenser", 90, "prepare", self.prepare),
                         Step("行星搅拌", "mixer", 120, "prepare"),
                         Step("涂膜固化", "oven", 1800, "prepare"),
                         Step("谐振腔测 Dk/Df", "cavity", 0, "measure", self.measure)]

    def truth(self, actual):
        dk, df = dk_df(actual["phi"], actual["h"])
        return score(dk, df, krieger_dougherty(actual["phi"], PHI_M))

    async def prepare(self, sample, lab):
        phi = sample.requested["phi"] * (1 + 0.005 * lab.rng.standard_normal())
        sample.actual = {"phi": float(phi), "h": sample.requested["h"]}
        sample.raw["viscosity_Pa_s"] = round(float(krieger_dougherty(phi, PHI_M)), 1)

    async def measure(self, sample, lab):
        dk, df = dk_df(sample.actual["phi"], sample.actual["h"])
        if lab.rng.random() < lab.p_timeout:
            await lab.clock.sleep(60)
            raise DeviceError("网络分析仪通讯超时")
        fr, q = extract_resonance(*simulate_s21(dk, df, lab.rng))
        eps1, tand = eps_from_cavity(fr, q)
        await lab.clock.sleep(180)
        sample.raw.update(f_GHz=round(fr / 1e9, 5), Q=round(q), Dk=round(eps1, 3), Df=round(tand, 5))
        sample.result = score(eps1, tand, sample.raw["viscosity_Pa_s"])
