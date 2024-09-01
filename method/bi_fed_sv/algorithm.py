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
        # self.shapley_values: list = []
        self.shapley_values: dict[int, list] = {}
        self.config: None | DistributedTrainingConfig = None

    # 生成集合 N 的所有子集
    def generate_subsets(self, round_N):
        subsets = []
        for r in range(len(round_N) + 1):
            subsets.extend(itertools.combinations(round_N, r))
        return subsets

    # 生成集合 N 的所有不相交子集对
    def generate_subset_pairs(self, round_N):
        subsets_S = self.generate_subsets(round_N)
        subsets_T = self.generate_subsets(round_N)
        subset_pairs = [(S, T) for S in subsets_S for T in subsets_T if set(S).isdisjoint(T)]
        return subset_pairs

    # 生成 N \ (S ∪ T) 的所有子集
    def generate_remaining_subsets(self, round_N, S, T):
        remaining = round_N - set(S) - set(T)
        return self.generate_subsets(remaining)

    # 计算 v(S ∪ A) 的累计效用
    def calculate_cumulative_utility(self, S, remaining_subsets, v_S):
        cumulative_utility = 0
        for subset in remaining_subsets:
            union_set = set(S).union(subset)
            cumulative_utility += v_S[tuple(sorted(union_set))]
        return cumulative_utility

    # 计算排序键
    def sort_key(self, pair, round_N):
        S, T = pair
        deltas = []
        for i in round_N:
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

    def read_matrix_from_csv(self, round_N):
        """从CSV文件中读取矩阵"""
        ROOT_DIR = "/home/cuitianxu/dls20240722/distributed_learning_simulator/tmp"
        data_dir = os.path.join(ROOT_DIR, 'theta_n')
        data_file = os.path.join(data_dir, 'theta_{}.csv'.format(len(round_N)))
        data = pd.read_csv(data_file, header=None, delim_whitespace=True)  # 根据你的CSV文件分隔符调整
        return data.to_numpy()

    def calculate_bifedsv(self, participants_set, theta_matrix, feature_matrix):
        """
        Calculate BiFedSV based on Matrix Semi Tensor Product Method.
        Parameters:
        participants_set (list): List of participants in the current round N^(r).
        theta_matrix (np.ndarray): Shapley matrix Θ|N^(r)|.
        feature_matrix (np.ndarray): Feature matrix M_v.
        historical_bifedsv (list): Historical BiFedSV values [Φ^(0), ..., Φ^(r-1)].

        Returns:
        dict: The BiFedSV Φ^(r) for each participant i in N^(r).
        """
        # Step 1: Calculate the BiFedSV for participants (p)
        bifedsv_p = np.dot(feature_matrix, theta_matrix)

        # Create a dictionary to store the results
        bifedsv = {}

        # Store BiFedSV for participants
        for i, participant in enumerate(participants_set):
            bifedsv[participant] = bifedsv_p[i]

        # Step 2: Find the historical optimal BiFedSV for non-participants (n - p)
        non_participants = set(range(len(self.config.worker_number))) - set(participants_set)
        bifedsv_non_p = {}

        for i in non_participants:
            # Find the optimal historical BiFedSV value for the non-participants
            bifedsv_non_p[i] = self.find_optimal_bifedsv[i]

        # Step 3: Update BiFedSV for all participants and non-participants
        for i in non_participants:
            bifedsv[i] = bifedsv_non_p[i]

        # Step 4: Combine and return the final BiFedSV Φ^(r) for all participants
        return bifedsv

    def find_optimal_bifedsv(self):
        #{round:[S_V,S_V]}
        #{subset:[S_V,S_V,...]}
        subsets: dict = {int: list}
        for round, S_Vs in self.shapley_values.items():
            for i, S_V in enumerate(S_Vs):
                subsets[i].append(S_V);
        optimal_bifedsv = []
        for S_Vs in subsets.values():
            optimal_bifedsv.append(max(S_Vs))

        # For simplicity, let's assume the optimal value is the maximum
        return optimal_bifedsv

    # 计算comfedsv需要的效用矩阵的产生过程
    # 注意：这里每轮的参与者集合是动态变化的，这篇论文用的是每轮随机选取一定比例的客户端，后边我的论文bifedsv中是根据客户端的效用选择的
    def _compute_impl(self, round_index: int) -> None:
        subsets = set()

        # 计算被选中参与者所有子集的效用 round_N -> self.selection_result[round_index]
        for subset in self.generate_subsets(self.selection_result[round_index]):
            if not subset:
                continue
            subset = tuple(sorted(subset))
            subsets.add(subset)
        assert self.batch_metric_fun is not None
        result_metrics: dict = {s: self.metric_fun(s) for s in subsets}
        # 将每个轮次中实际参与者的所有子集的效用对应到效用矩阵utilities_matrix中去
        #self.utilities_matrix[round_index - 1][self.all_subsets.index(())] = 0

        for subset, metric in result_metrics.items():
            # subset = self.get_players(subset)
            # self.utilities_matrix[round_index - 1][
            #     self.all_subsets.index(subset)
            # ] = metric
            log_info("round %s subset %s metric %s", round_index, subset, metric)
        theta_matrix = self.read_matrix_from_csv(self.selection_result[round_index])
        subset_pairs = self.generate_subset_pairs(self.selection_result[round_index])
        sorted_pairs = sorted(subset_pairs, key=lambda pair: self.sort_key(pair, self.selection_result[round_index]))

        v_ST = {}
        feature_matrix = []

        for S, T in sorted_pairs:
            remaining_subsets = self.generate_remaining_subsets(self.selection_result[round_index], S, T)
            cumulative_utility = self.calculate_cumulative_utility(S, remaining_subsets, result_metrics)
            matrix = cumulative_utility / (2 ** len(self.selection_result[round_index] - set(S) - set(T)))
            v_ST[(S, T)] = matrix
            feature_matrix.append(matrix)

        bifedsv = self.calculate_bifedsv(self.selection_result[round_index], theta_matrix, feature_matrix, self.find_optimal_bifedsv)
        log_info("bifedsv: %s", bifedsv)


    # def get_result(self) -> list:
    #     return self.shapley_values

    def get_result(self) -> dict:
        return {
            "round_shapley_values": self.shapley_values,
        }
class BiFedShapleyValueAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(BiFedShapleyValue, *args, **kwargs)

    @property
    def sv_algorithm(self) -> BiFedShapleyValue:
        algorithm = super().sv_algorithm
        assert isinstance(algorithm, BiFedShapleyValue)
        algorithm.config = self.config
        return algorithm
