import functools

import numpy as np
import pandas as pd
from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.shapely_value.shapley_value import ShapleyValue

from .shapley_value_algorithm import ShapleyValueAlgorithm


class IntervalShapleyValue(ShapleyValue):
    def __init__(
        self, players: list, last_round_metric: float = 0, LAMBDA: float = 0.9904
    ) -> None:
        super().__init__(players=players, last_round_metric=last_round_metric)
        self.shapley_values: dict = {}
        self.metrics: dict = {}  # 新增属性来保存metrics字典
        self.LAMBDA = LAMBDA
        self.last_round_number = 0

    def compute(self, round_number: int) -> None:
        # 初始化metrics字典，包含空集的度量值
        self.metrics[round_number] = {}
        self.last_round_number = round_number
        assert self.metric_fun is not None
        for subset in self.powerset(self.complete_player_indices):
            subset = tuple(sorted(subset))
            if not subset:
                metric = 0
            else:
                metric = self.metric_fun(subset)
            self.metrics[round_number][subset] = metric
            log_info("round %s subset %s metric %s", round_number, subset, metric)

    # 按照区间shapley值的计算公式要求生成所有子集的正确排序
    @classmethod
    def sorted_subsets(cls, nums) -> list[tuple]:
        if not nums:
            return [()]
        subsets_without_first = cls.sorted_subsets(nums[1:])
        subsets_with_first = [(nums[0],) + subset for subset in subsets_without_first]
        return subsets_with_first + subsets_without_first

    def exit(self):
        # 拿到效用以后的计算过程
        # 定义区间效用的字典
        interval_min = {}
        for subset, metric in self.metrics.values():
            if subset not in interval_min:
                interval_min[subset] = metric
            else:
                interval_min[subset] = min(metric, interval_min[subset])

        interval_max = {}
        for subset, metric in self.metrics.values():
            if subset not in interval_max:
                interval_max[subset] = metric
            else:
                interval_max[subset] = max(metric, interval_max[subset])

        # 定义区间上（下）限的列表，并存入相应的值 维度（1, 1024）

        M_MIN = []
        M_MAX = []
        sorted_subsets = self.sorted_subsets(self.complete_player_indices)
        # 根据正确的子集顺序提取并填入对应的效用值
        for subset in sorted_subsets:
            # 从interval_min提取值并添加到M_MIN
            M_MIN.append(interval_min[subset])
            # 从interval_max提取值并添加到M_MAX
            M_MAX.append(interval_max[subset])

        # 导入E和F 维度（1024，10）目前先这样,后边可以尝试连接matlab自动生成(参数是_lambda和players)
        E = pd.read_excel("E.xls")
        F = pd.read_excel("F.xls")
        E_mat = E.values
        F_mat = F.values

        # 定义存放参与者贡献的两个列表
        fai_min_list = []
        fai_max_list = []
        # 利用矩阵计算公式进行计算
        fai_min_list = np.dot(M_MIN, E_mat) - self.LAMBDA * np.dot(M_MAX, F_mat)
        fai_max_list = np.dot(M_MAX, E_mat) - self.LAMBDA * np.dot(M_MIN, F_mat)
        # 将列表合并为区间
        self.shapley_values[self.last_round_number] = dict(
            zip(sorted_subsets, zip(fai_min_list, fai_max_list))
        )


class IntervalShapleyValueAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(IntervalShapleyValue, *args, **kwargs)
        self.sv_algorithm_cls = functools.partial(
            IntervalShapleyValue, LAMBDA=self.config.algorithm_kwargs["lambda"]
        )