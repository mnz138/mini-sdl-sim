"""按名字选择实验配置，供命令行和网页共同使用。"""

from .cavity import Cavity
from .electrolyte import Electrolyte
from .tim import ThermalInterface

PROFILES = {
    "electrolyte": Electrolyte,
    "tim": ThermalInterface,
    "cavity": Cavity,
}
