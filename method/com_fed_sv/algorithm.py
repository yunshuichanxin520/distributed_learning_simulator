import operator as op
from functools import cached_property, reduce

import numpy as np
from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.shapely_value.shapley_value import \
    RoundBasedShapleyValue
from distributed_learning_simulation import DistributedTrainingConfig
from distributed_learning_simulator.algorithm.shapley_value_algorithm import \
    ShapleyValueAlgorithm
from lripy import drcomplete


class ComFedShapleyValue(RoundBasedShapleyValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shapley_values: list = []
        self.config: None | DistributedTrainingConfig = None
        self.all_subsets = [
            tuple(sorted(s)) for s in self.powerset(self.complete_player_indices)
        ]

    @cached_property
    def utilities_matrix(self):
        assert self.config is not None
        return np.zeros(
            (self.config.round, len(list(self.powerset(self.complete_player_indices))))
        )

    # function for computing the combinatorial number
    def ncr(self, n, r):
        r = min(r, n - r)
        numer = reduce(op.mul, range(n, n - r, -1), 1)
        denom = reduce(op.mul, range(1, r + 1), 1)
        return numer // denom

    # 定义计算comfedsv的方法,一旦获得补全的效用矩阵，便可以通过下面分方法计算comfedsv
    def compute_shapley_value_from_matrix(self, utility_matrix, all_subsets):
        assert self.config is not None
        T = self.config.round
        N = self.config.worker_number
        sv_completed = np.zeros(N)
        for i in range(N):
            sub_list = list(range(N))
            sub_list.pop(i)
            sub_powerset = self.powerset(sub_list)
            for s in sub_powerset:
                id1 = all_subsets.index(s)
                id2 = all_subsets.index(tuple(sorted(list(s) + [i])))
                for t in range(T):
                    v1 = utility_matrix[t, id1]
                    v2 = utility_matrix[t, id2]
                    val = (v2 - v1) / self.ncr(N - 1, len(s))
                    sv_completed[i] += val
            sv_completed[i] /= N
        return sv_completed

    # 计算comfedsv需要的效用矩阵的产生过程
    # 注意：这里每轮的参与者集合是动态变化的，这篇论文用的是每轮随机选取一定比例的客户端，后边我的论文bifedsv中是根据客户端的效用选择的
    def _compute_impl(self, round_index: int) -> None:
        subsets = set()

        # 计算被选中参与者所有子集的效用，这里我们使用metric来代替utility
        for subset in self.powerset(self.complete_player_indices):
            if not subset:
                continue
            subset = tuple(sorted(subset))
            subsets.add(subset)
        assert self.batch_metric_fun is not None
        result_metrics: dict = {s: self.metric_fun(s) for s in subsets}
        # 将每个轮次中实际参与者的所有子集的效用对应到效用矩阵utilities_matrix中去
        self.utilities_matrix[round_index - 1][self.all_subsets.index(())] = 0

        for subset, metric in result_metrics.items():
            subset = self.get_players(subset)
            self.utilities_matrix[round_index - 1][
                self.all_subsets.index(subset)
            ] = metric
            log_info("round %s subset %s metric %s", round_index, subset, metric)

    def exit(self) -> None:
        assert self.config is not None
        # 利用lripy中的drcomplete方法补全效用矩阵，并调用compute_shapley_value_from_matrix方法计算
        mask = np.zeros(shape=(self.config.round, len(self.all_subsets)), dtype=int)
        # U = csr_matrix(np.multiply(self.utilities_matrix, mask))
        utility_matrix_completed = drcomplete(self.utilities_matrix, mask, 3, 2)[0]
        sv_completed = self.compute_shapley_value_from_matrix(
            utility_matrix_completed, self.all_subsets
        )
        log_info("comfedsv: %s", sv_completed)
        self.shapley_values = list(sv_completed)

    def get_result(self) -> list:
        return self.shapley_values


class ComFedShapleyValueAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(ComFedShapleyValue, *args, **kwargs)

    @property
    def sv_algorithm(self) -> ComFedShapleyValue:
        algorithm = super().sv_algorithm
        assert isinstance(algorithm, ComFedShapleyValue)
        algorithm.config = self.config
        return algorithm
