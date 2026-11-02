"""优化器：统一成 ask（要下一批配方）/ tell（交回结果）两个动作，方便随时替换算法。"""
import warnings

import numpy as np
import pandas as pd


class RandomOptimizer:
    """随机搜索：在参数范围内均匀抽样，作为对照组。"""

    def __init__(self, params, target: str, seed: int = 0):
        self.params, self.target = params, target
        self.rng = np.random.default_rng(seed)

    def ask(self, n: int) -> list:
        return [{p.name: float(self.rng.uniform(p.low, p.high)) for p in self.params} for _ in range(n)]

    def tell(self, rows: list) -> None:
        pass   # 随机搜索不从结果里学习


class BayesOptimizer:
    """BayBE 贝叶斯优化：先随机探一批，之后用高斯过程代理模型 + 采集函数推荐下一批。"""

    def __init__(self, params, target: str, seed: int = 0):
        from baybe import Campaign
        from baybe.parameters import NumericalContinuousParameter
        from baybe.searchspace import SearchSpace
        from baybe.settings import Settings
        from baybe.targets import NumericalTarget

        warnings.filterwarnings("ignore", module="baybe")
        Settings(random_seed=seed).activate()
        self.params, self.target = params, target
        space = SearchSpace.from_product([NumericalContinuousParameter(p.name, (p.low, p.high)) for p in params])
        self.campaign = Campaign(searchspace=space, objective=NumericalTarget(target))   # 默认求最大值

    def ask(self, n: int) -> list:
        return self.campaign.recommend(batch_size=n).to_dict("records")

    def tell(self, rows: list) -> None:
        if rows:
            columns = [p.name for p in self.params] + [self.target]
            df = pd.DataFrame([{name: float(row[name]) for name in columns} for row in rows])
            if not np.isfinite(df.to_numpy()).all():
                raise ValueError("回流的实际配方和测量结果必须为有限数值")
            # 称量和移液误差可能使实际值略超出推荐范围；保留实际值，不能截断后冒充实测值。
            # 此标志只放宽测量数据的范围检查，下一轮推荐仍受搜索空间边界约束。
            self.campaign.add_measurements(df, numerical_measurements_must_be_within_tolerance=False)


OPTIMIZERS = {"random": RandomOptimizer, "bayes": BayesOptimizer}
