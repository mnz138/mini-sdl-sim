"""电解液配方：锂盐浓度 × 溶剂配比，目标是离子电导率最高。
注意：truth() 是“物理启发的合成函数”，形状参照文献里电导率随浓度先升后降的规律，
参数是展示用的，是仿真结果
"""
import numpy as np

from ..models import DeviceError
from .base import Param, Profile, Step

C_STOCK = 2.0     # 含盐储备液浓度，mol/L
TOTAL_UL = 200.0  # 每个样品的总体积，µL
MIN_UL = 5.0      # 移液器能准确移取的最小体积，µL
K_CELL = 1.0      # 电导池常数，cm^-1（由电极面积和间距决定，用标准 KCl 溶液标定）

# 4 个储液槽：两种含盐储备液 + 两种纯溶剂
STOCKS = ["salt_EC", "salt_DMC", "EC", "DMC"]


def recipe_to_volumes(c_salt: float, x_ec: float) -> tuple:
    """把配方（浓度、EC 比例）换算成 4 个储液槽各取多少 µL。

    简化假设：体积可加和、忽略盐本身的体积、EC 按液体处理（真实的 EC 熔点约 36 °C，需加热或预混）。
    返回 (各储液体积, 实际配出来的配方)。小于最小移液量的体积直接取 0，因此实际配方可能和要求略有不同。
    """
    if not (0 <= c_salt <= C_STOCK and 0 <= x_ec <= 1):
        raise ValueError(f"配方超出可配范围：c_salt={c_salt}, x_ec={x_ec}")
    s = c_salt / C_STOCK                       # 体积中来自含盐储备液的比例
    vols = {
        "salt_EC": x_ec * s * TOTAL_UL,
        "salt_DMC": (1 - x_ec) * s * TOTAL_UL,
        "EC": x_ec * (1 - s) * TOTAL_UL,
        "DMC": (1 - x_ec) * (1 - s) * TOTAL_UL,
    }
    vols = {k: (0.0 if v < MIN_UL else round(v, 1)) for k, v in vols.items()}
    total = sum(vols.values())
    actual = {
        "c_salt": C_STOCK * (vols["salt_EC"] + vols["salt_DMC"]) / total,
        "x_ec": (vols["salt_EC"] + vols["EC"]) / total,
    }
    return vols, actual


def casteel_amis(c: float, kappa_max: float, mu: float, a: float = 1.0, b: float = -0.5) -> float:
    """Casteel–Amis 经验式：电导率随浓度先升后降，在 c = mu 处达到最大值 kappa_max。"""
    return kappa_max * (c / mu) ** a * np.exp(b * (c - mu) ** 2 - a / mu * (c - mu))


def simulate_eis(r_bulk: float, rng, c_geo: float = 1e-10, q_dl: float = 1e-5, n_dl: float = 0.9,
                 noise: float = 0.003):
    """模拟电导池的电化学阻抗谱（1 Hz–1 MHz）。

    等效电路：体电阻 R_b 与几何电容并联，再串联电极界面的双电层（常相位元件 CPE）。
    和用网络分析仪测谐振电路一样，本质都是“扫频测复阻抗”。
    """
    f = np.logspace(0, 6, 61)
    w = 2 * np.pi * f
    z = r_bulk / (1 + 1j * w * r_bulk * c_geo) + 1 / (q_dl * (1j * w) ** n_dl)
    z = z.real * (1 + noise * rng.standard_normal(f.size)) + 1j * z.imag * (1 + noise * rng.standard_normal(f.size))
    return f, z


def extract_rb(f, z) -> float:
    """取 -Im(Z) 最小处（高频半圆与低频直线之间的“谷”）的实部，作为体电阻 R_b。"""
    f, z = np.asarray(f, dtype=float), np.asarray(z, dtype=complex)
    if (f.ndim != 1 or z.ndim != 1 or f.size != z.size or f.size < 2
            or not np.all(np.isfinite(f)) or not np.all(np.isfinite(z))
            or np.any(f <= 0) or np.any(np.diff(f) <= 0)):
        raise ValueError("EIS 需要长度一致、有限的一维数据，频率必须为正且递增")
    k = int(np.argmin(-z.imag))
    rb = float(z.real[k])
    if rb <= 0:
        raise ValueError("提取的体电阻必须为正")
    return rb


class Electrolyte(Profile):
    name = "electrolyte"
    target = "conductivity"
    target_unit = "mS/cm"
    params = [
        Param("c_salt", 0.2, 2.0, "mol/L", "锂盐浓度"),
        Param("x_ec", 0.1, 0.9, "体积分数", "溶剂中 EC 的比例（其余为 DMC）"),
    ]
    infeasible_value = 0.0

    def __init__(self, n_cells: int = 1, stock_volume_uL: float = 1500.0):
        self.capacities = {"liquid_handler": 1, "shaker": 8, "cond_cell": n_cells}
        self.stock_volume_uL = stock_volume_uL
        self.workflow = [
            Step("配液", "liquid_handler", 0, "prepare", self.prepare),
            Step("混匀", "shaker", 60, "prepare"),
            Step("测电导率", "cond_cell", 0, "measure", self.measure),
        ]

    def truth(self, actual: dict) -> float:
        x = actual["x_ec"]
        kappa_max = 5 + 6.5 * np.exp(-((x - 0.45) / 0.28) ** 2)   # EC 帮助解离、DMC 降低粘度，折中处最好
        mu = 0.9 + 0.4 * x                                        # 最佳浓度随 EC 比例略升
        return float(casteel_amis(actual["c_salt"], kappa_max, mu))

    def setup_lab(self, lab) -> None:
        from ..lab import LiquidStation
        lab.liquid = LiquidStation(lab, STOCKS, self.stock_volume_uL)

    async def prepare(self, sample, lab) -> None:
        vols, actual = recipe_to_volumes(**sample.requested)
        sample.actual = actual
        well = lab.liquid.new_well()
        sample.raw.update(well=well.name, volumes_uL=vols)
        for stock, v in vols.items():
            if v > 0:
                await lab.liquid.transfer(stock, well, v)

    async def measure(self, sample, lab) -> None:
        r_true = K_CELL / (self.truth(sample.actual) / 1000)     # κ(mS/cm) -> R_b(Ω)
        if lab.rng.random() < lab.p_timeout:                     # 注入故障：恒电位仪通讯超时
            await lab.clock.sleep(60)
            raise DeviceError("恒电位仪通讯超时")
        f, z = simulate_eis(r_true, lab.rng)
        r_b = extract_rb(f, z)
        await lab.clock.sleep(150)                               # 扫频约 60 s + 清洗吹干约 90 s
        sample.raw["R_b_ohm"] = round(r_b, 2)
        sample.raw["cell_temperature_C"] = 25.0                  # 仿真恒温；真机应读温度传感器
        sample.result = K_CELL / r_b * 1000
