import operator as op
from functools import cached_property, reduce
import pandas as pd
import numpy as np
from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.shapely_value.shapley_value import \
    RoundBasedShapleyValue
from distributed_learning_simulation import DistributedTrainingConfig
from distributed_learning_simulator.algorithm.shapley_value_algorithm import \
    ShapleyValueAlgorithm
import itertools
import os

class BiFedShapleyValue(RoundBasedShapleyValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shapley_values: list = []
        self.config: None | DistributedTrainingConfig = None

    # 生成集合 N 的所有子集
    def generate_subsets(self, N):
        subsets = []
        for r in range(len(N) + 1):
            subsets.extend(itertools.combinations(N, r))
        return subsets

    # 生成集合 N 的所有不相交子集对
    def generate_subset_pairs(self, N):
        subsets_S = self.generate_subsets(N)
        subsets_T = self.generate_subsets(N)
        subset_pairs = [(S, T) for S in subsets_S for T in subsets_T if set(S).isdisjoint(T)]
        return subset_pairs

    # 生成 N \ (S ∪ T) 的所有子集
    def generate_remaining_subsets(self, N, S, T):
        remaining = N - set(S) - set(T)
        return self.generate_subsets(remaining)

    # 计算 v(S ∪ A) 的累计效用
    def calculate_cumulative_utility(self, S, remaining_subsets, v_S):
        cumulative_utility = 0
        for subset in remaining_subsets:
            union_set = set(S).union(subset)
            cumulative_utility += v_S[tuple(sorted(union_set))]
        return cumulative_utility

    # 计算排序键
    def sort_key(self, pair, N):
        S, T = pair
        deltas = []
        for i in N:
            if i in S:
                deltas.append([1, 0, 0])
            elif i in T:
                deltas.append([0, 0, 1])
            else:
                deltas.append([0, 1, 0])
        # 计算矩阵半张量积
        Z = deltas[0]
        for delta_i in deltas[1:]:
            Z = np.kron(Z, delta_i)
        # 找到矩阵 Z 中第一个非零元素的索引
        non_zero_index = np.argmax(Z != 0)
        return non_zero_index

    def read_matrix_from_csv(self, N):
        """从CSV文件中读取矩阵"""
        ROOT_DIR = "/home/cuitianxu/dls20240722/distributed_learning_simulator/tmp"
        data_dir = os.path.join(ROOT_DIR, 'theta_n')
        data_file = os.path.join(data_dir, 'theta_{}.csv'.format(len(N)))
        data = pd.read_csv(data_file, header=None, delim_whitespace=True)  # 根据你的CSV文件分隔符调整
        return data.to_numpy()

    def compute_bifedsv(self, M_v, theta_n):
        """计算M_v与从CSV文件读取的矩阵的乘积"""
        bifedsv = np.dot(M_v, theta_n)
        return bifedsv


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

    def exit(self, n) -> None:
        assert self.config is not None

        N = set(range(1, n + 1))
        # 生成 N 的所有子集及其效用 v(S)
        all_subsets = self.generate_subsets(N)
        v_S = {subset: 1 for subset in all_subsets}  # 这里假设 v(S) 都为 1，可以根据需要修改

        # 生成所有不相交子集对 (S, T)
        subset_pairs = self.generate_subset_pairs(N)

        # 计算 v(S, T)
        v_ST = {}
        for S, T in subset_pairs:
            remaining_subsets = self.generate_remaining_subsets(N, S, T)
            cumulative_utility = self.calculate_cumulative_utility(S, remaining_subsets, v_S)
            v_ST[(S, T)] = cumulative_utility / (2 ** len(N - set(S) - set(T)))

        # 对生成的子集对进行自定义顺序排序
        sorted_pairs = sorted(subset_pairs, key=lambda pair: self.sort_key(pair, N))
        M_v =
        theta_n = self.read_matrix_from_csv(N)
        sv_bifed = self.compute_bifedsv(self, theta_n, M_v)
        log_info("bifedsv: %s", sv_bifed)

    def get_result(self) -> list:
        return self.shapley_values

class BiFedShapleyValueAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(BiFedShapleyValue, *args, **kwargs)

    @property
    def sv_algorithm(self) -> BiFedShapleyValue:
        algorithm = super().sv_algorithm
        assert isinstance(algorithm, BiFedShapleyValue)
        algorithm.config = self.config
        return algorithm
