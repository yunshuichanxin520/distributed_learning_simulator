import os
from itertools import combinations

import numpy as np
import pandas as pd
import operator as op
from functools import reduce
from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.shapely_value.shapley_value import \
    RoundBasedShapleyValue
from distributed_learning_simulation import DistributedTrainingConfig
from distributed_learning_simulator.algorithm.shapley_value_algorithm import \
    ShapleyValueAlgorithm
from scipy.sparse import csr_matrix
from lripy import drcomplete
from tqdm import tqdm





class ComFedShapleyValue(RoundBasedShapleyValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.frac:float = 0.5
        self.shapley_values: list = []
        self.metrics: dict[int, dict] = {}  # 新增属性来保存metrics字典
        self.config: None | DistributedTrainingConfig = None
        self.utilities_matrix: np.ndarray
        self.all_subsets = self.powerset(self.complete_player_indices)
        self.get_utilities_matrix()


    def get_utilities_matrix(self):
        self.utilities_matrix = np.zeros((self.config.round,len(list(self.powerset(self.complete_player_indices)))))
    #
    # # mask in each round
    def roundly_mask(self, idxs_users, all_subsets):
        mask_vec = np.zeros(len(all_subsets))
        sub_powerset = self.powerset(idxs_users)
        for s in sub_powerset.keys():
            i = all_subsets[s]
            mask_vec[i] = 1
        return mask_vec
    #
    # # our approach
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

    def _compute_impl(self, round_number: int) -> None:
        self.metrics[round_number] = {}
        subsets = set()

        # select participants
        m = max(int(self.frac * self.config.worker_number), 1)
        if round_number == 1:
            index_players = list(range(self.config.worker_number))
        else:
            index_players = np.random.choice(range(self.config.worker_number), m, replace=False)

        #计算被选中参与者所有子集的效用
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
        #for metric in result_metrics
        for subset, metric in result_metrics.items():
            self.utilities_matrix[round_number][list(self.all_subsets).index(subset)] = metric
            log_info("round %s subset %s metric %s", round_number, subset, metric)

        self.metrics[round_number].update(result_metrics)



    def exit(self) -> None:
        # 补全效用矩阵
        mask = np.zeros((self.config.round, self.config.worker_number))
        mask = mask.astype(int)
        U = csr_matrix(np.multiply(self.utilities_matrix, mask))
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
