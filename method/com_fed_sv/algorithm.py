import os
from itertools import combinations

import numpy as np
import pandas as pd

from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.shapely_value.shapley_value import \
    RoundBasedShapleyValue
from distributed_learning_simulation import DistributedTrainingConfig
from distributed_learning_simulator.algorithm.shapley_value_algorithm import \
    ShapleyValueAlgorithm
from lripy import drcomplete
from tqdm import tqdm





class ComFedShapleyValue(RoundBasedShapleyValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.frac:float = 0.5
        self.shapley_values: list = []
        self.metrics: dict[int, dict] = {}  # 新增属性来保存metrics字典
        self.config: None | DistributedTrainingConfig = None
        self.utilities_matrix: np.ndarray #新增T*2^N维的效用矩阵保存utilities（metrics）
        self.all_subsets = self.powerset(self.complete_player_indices)
        self.get_utilities_matrix()


    def get_utilities_matrix(self):
        self.utilities_matrix = np.zeros((self.config.round,len(list(self.powerset(self.complete_player_indices)))))
    #

    # 定义计算comfedsv的方法,一旦获得补全的效用矩阵，便可以通过下面分方法计算comfedsv
    def compute_shapley_value_from_matrix(self, utility_matrix, all_subsets):
        T = self.config.round
        N = self.config.worker_number
        sv_completed = np.zeros(N)
        for i in range(N):
            sub_list = list(range(N))
            sub_list.pop(i)
            sub_powerset = self.powerset(sub_list)
            for s in sub_powerset.keys():
                # id1 = all_subsets.index(s)
                id1 = all_subsets[s]
                id2 = all_subsets[tuple(sorted(list(s) + [i]))]
                for t in range(T):
                    v1 = utility_matrix[t, id1]
                    v2 = utility_matrix[t, id2]
                    val = (v2 - v1) / combinations(N-1, len(s))
                    sv_completed[i] += val
            sv_completed[i] /= N
        return sv_completed

    # 计算bifedsv需要的效用矩阵的产生过程
    # 注意：这里每轮的参与者集合是动态变化的，这篇论文用的是每轮随机选取一定比例的客户端，后边我的论文bifedsv中是根据客户端的效用选择的
    def _compute_impl(self, round_number: int) -> None:
        self.metrics[round_number] = {}
        subsets = set()

        # 选择参与者，frac是可调的比例参数
        m = max(int(self.frac * self.config.worker_number), 1)
        # 强制规定第一轮N中的所有参与者必须参加，这是论文的假设条件
        if round_number == 1:
            index_players = list(range(self.config.worker_number))
        else:
            index_players = np.random.choice(range(self.config.worker_number), m, replace=False)

        #计算被选中参与者所有子集的效用，这里我们使用metric来代替utility
        for subset in self.powerset(index_players):
            subset = tuple(sorted(subset))
            if not subset:
                metric = 0
                self.metrics[round_number][subset] = metric
                log_info("round %s subset %s metric %s", round_number, subset, metric)
            else:
                subsets.add(subset)
        assert self.batch_metric_fun is not None
        result_metrics: dict = self.batch_metric_fun(subsets)
        # 将每个轮次中实际参与者的所有子集的效用对应到效用矩阵utilities_matrix中去
        for subset, metric in result_metrics.items():
            self.utilities_matrix[round_number][list(self.all_subsets).index(subset)] = metric
            log_info("round %s subset %s metric %s", round_number, subset, metric)

        self.metrics[round_number].update(result_metrics)



    def exit(self) -> None:
        # 利用lripy中的drcomplete方法补全效用矩阵，并调用compute_shapley_value_from_matrix方法计算
        mask = np.zeros((self.config.round, self.config.worker_number))
        mask = mask.astype(int)
        utility_matrix_completed = drcomplete(self.utilities_matrix, mask, 3, 2)[0]
        sv_completed = self.compute_shapley_value_from_matrix(utility_matrix_completed,self.all_subsets)
        log_info("comfedsv: %s", sv_completed)

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
