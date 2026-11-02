"""导热界面材料（TIM，一种常见的半导体封装材料）：填料含量 × 粗细粒径配比。
目标：导热系数尽量高，同时粘度不能太大（太稠就没法点胶）。
truth() 用 Lewis–Nielsen 导热模型和 Krieger–Dougherty 粘度模型合成，参数是演示用的。
"""
import numpy as np

from ..models import DeviceError, RecipeInfeasible
from .base import Param, Profile, Step

K_MATRIX, K_FILLER = 0.2, 30.0      # 树脂基体 / 氧化铝填料的导热系数，W/(m·K)
RHO_MATRIX, RHO_FILLER = 1200.0, 3950.0  # 密度，kg/m^3
CP_MATRIX, CP_FILLER = 1100.0, 880.0     # 比热容，J/(kg·K)
ETA0 = 5.0          # 不加填料时树脂的粘度，Pa·s
ETA_MAX = 150.0     # 点胶工艺能接受的最大粘度，Pa·s
PENALTY = 3.0       # 超过粘度上限时的惩罚系数
THICKNESS = 1e-3    # 激光闪射样品厚度，m
PARKER = 0.1388     # Parker 公式常数：alpha = 0.1388 d^2 / t_half


def phi_max(r_small: float) -> float:
    """最大堆积分数：粗细颗粒搭配时，细颗粒填进粗颗粒空隙，能堆得更密（合成函数）。"""
    return 0.637 + 0.15 * np.exp(-((r_small - 0.3) / 0.18) ** 2)


def lewis_nielsen(phi: float, phim: float, A: float = 1.5) -> float:
    """Lewis–Nielsen 模型：复合材料导热系数。A=1.5 对应球形颗粒。"""
    ratio = K_FILLER / K_MATRIX
    B = (ratio - 1) / (ratio + A)
    psi = 1 + (1 - phim) / phim ** 2 * phi
    return K_MATRIX * (1 + A * B * phi) / (1 - B * psi * phi)


def krieger_dougherty(phi: float, phim: float, intrinsic: float = 2.5) -> float:
    """Krieger–Dougherty 公式：填料越接近最大堆积分数，粘度越是急剧上升。"""
    return ETA0 * (1 - phi / phim) ** (-intrinsic * phim)


def rho_cp(phi: float) -> float:
    """体积比热 ρ·c_p（按混合规则）。"""
    rho = phi * RHO_FILLER + (1 - phi) * RHO_MATRIX
    w = phi * RHO_FILLER / rho                       # 填料质量分数
    return rho * (w * CP_FILLER + (1 - w) * CP_MATRIX)


def score(k: float, eta: float) -> float:
    """优化目标：导热系数，粘度超标按对数扣分，最低 0 分（和“配方不可行”同分，都是最差）。"""
    return max(0.0, k - PENALTY * max(0.0, float(np.log(eta / ETA_MAX))))


def simulate_lfa(alpha: float, rng, d: float = THICKNESS, noise: float = 0.01, n_terms: int = 60):
    """模拟激光闪射：前表面被激光脉冲加热，记录背面温升（归一化）。Parker 1961 的级数解。"""
    t_half = PARKER * d ** 2 / alpha
    t = np.linspace(0, 6 * t_half, 600)
    n = np.arange(1, n_terms + 1)[:, None]
    v = 1 + 2 * np.sum((-1.0) ** n * np.exp(-(n ** 2) * np.pi ** 2 * alpha * t / d ** 2), axis=0)
    v[0] = 0.0
    return t, v + noise * rng.standard_normal(t.size)


def extract_t_half(t, v) -> float:
    """先平滑，以末尾 100 点均值估计温升平台，再插值求第一次半升温时刻。"""
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    if (t.ndim != 1 or v.ndim != 1 or t.size != v.size or t.size < 9
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(v))
            or np.any(t < 0) or np.any(np.diff(t) <= 0)):
        raise ValueError("LFA 需要至少 9 个有限数据点，时间必须非负且递增")
    v_max = float(np.mean(v[-100:]))
    if v_max <= 0:
        raise ValueError("LFA 温升平台必须为正")
    v_s = np.convolve(v, np.ones(9) / 9, mode="same")
    crossings = np.flatnonzero(v_s >= v_max / 2)
    if crossings.size == 0 or crossings[0] == 0:
        raise ValueError("LFA 曲线未包含可插值的半升温交点")
    i = int(crossings[0])
    return float(t[i - 1] + (v_max / 2 - v_s[i - 1]) * (t[i] - t[i - 1]) / (v_s[i] - v_s[i - 1]))


class ThermalInterface(Profile):
    name = "tim"
    target = "score"
    target_unit = "W/(m·K)"
    params = [
        Param("phi", 0.30, 0.75, "体积分数", "氧化铝填料体积分数"),
        Param("r_small", 0.0, 1.0, "比例", "填料中细颗粒的比例"),
    ]
    infeasible_value = 0.0

    def __init__(self):
        self.capacities = {"dispenser": 1, "mixer": 1, "rheometer": 1, "oven": 8, "lfa": 1}
        self.workflow = [
            Step("称量投料", "dispenser", 90, "prepare", self.prepare),
            Step("行星搅拌", "mixer", 120, "prepare"),
            Step("测粘度", "rheometer", 0, "prepare", self.rheology),
            Step("固化", "oven", 1800, "prepare"),
            Step("激光闪射测导热", "lfa", 0, "measure", self.measure),
        ]

    def truth(self, actual: dict) -> float:
        phim = phi_max(actual["r_small"])
        if actual["phi"] >= 0.98 * phim:
            return self.infeasible_value
        return score(lewis_nielsen(actual["phi"], phim), krieger_dougherty(actual["phi"], phim))

    async def prepare(self, sample, lab) -> None:
        phi = sample.requested["phi"] * (1 + 0.005 * lab.rng.standard_normal())   # 粉体称量有 0.5% 误差
        sample.actual = {"phi": float(phi), "r_small": sample.requested["r_small"]}
        if phi >= 0.98 * phi_max(sample.actual["r_small"]):
            raise RecipeInfeasible("填料接近最大堆积分数，膏体无法混匀")

    async def rheology(self, sample, lab) -> None:
        phim = phi_max(sample.actual["r_small"])
        eta = krieger_dougherty(sample.actual["phi"], phim) * (1 + 0.03 * lab.rng.standard_normal())
        await lab.clock.sleep(120)
        sample.raw["viscosity_Pa_s"] = round(float(eta), 1)

    async def measure(self, sample, lab) -> None:
        phim = phi_max(sample.actual["r_small"])
        k_true = lewis_nielsen(sample.actual["phi"], phim)
        if lab.rng.random() < lab.p_timeout:            # 注入故障：样品台卡住
            await lab.clock.sleep(60)
            raise DeviceError("激光闪射仪样品台卡住")
        t, v = simulate_lfa(k_true / rho_cp(sample.actual["phi"]), lab.rng)
        t_half = extract_t_half(t, v)
        k = PARKER * THICKNESS ** 2 / t_half * rho_cp(sample.actual["phi"])
        await lab.clock.sleep(240)
        sample.raw.update(t_half_s=round(t_half, 4), k_W_mK=round(k, 3))
        sample.result = score(k, sample.raw["viscosity_Pa_s"])
