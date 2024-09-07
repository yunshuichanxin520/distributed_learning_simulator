import itertools
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.shapely_value.shapley_value import \
    RoundBasedShapleyValue
from distributed_learning_simulation import DistributedTrainingConfig
from distributed_learning_simulator.algorithm.shapley_value_algorithm import \
    ShapleyValueAlgorithm

# from .server import BiFedSVServer


class BiFedShapleyValue(RoundBasedShapleyValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bifed_sv: dict = {}  # 初始化 bifed_sv
        self.shapley_values: dict[int, list] = {}
        self.config: None | DistributedTrainingConfig = None

    # 生成集合 round_participants 的所有子集
    def generate_subsets(self, round_participants):
        subsets = []
        for i in range(len(round_participants) + 1):
            subsets.extend(itertools.combinations(round_participants, i))
        return subsets

    # 生成集合 round_participants 的所有不相交子集对
    def generate_subset_pairs(self, round_participants):
        subsets_S = self.generate_subsets(round_participants)
        subsets_T = self.generate_subsets(round_participants)
        subset_pairs = [
            (S, T) for S in subsets_S for T in subsets_T if set(S).isdisjoint(T)
        ]
        return subset_pairs

    # 生成 round_participants \ (S ∪ T) 的所有子集
    def generate_remaining_subsets(self, round_participants, S, T):
        remaining = round_participants - set(S) - set(T)
        return self.generate_subsets(remaining)

    # 计算 v(S ∪ A) 的累计效用
    def calculate_cumulative_utility(self, S, remaining_subsets, v_S):
        cumulative_utility = 0
        for subset in remaining_subsets:
            union_set = set(S).union(subset)
            cumulative_utility += v_S[tuple(sorted(union_set))]
        return cumulative_utility

    # 计算排序键，因为不相交子集对在特征矩阵feature_matrix中出现的顺序有要求
    def sort_key(self, pair, round_participants):
        S, T = pair
        deltas = []
        for i in round_participants:
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

    # 从文件夹tmp中自动的读取theta_matrix，theta_n，n为每次参与联邦学习的参与者个数
    def read_matrix_from_csv(self, round_participants):
        """从CSV文件中读取矩阵"""
        ROOT_DIR = "/home/cuitianxu/dls20240722/distributed_learning_simulator/tmp"
        data_dir = os.path.join(ROOT_DIR, "theta_n")
        data_file = os.path.join(
            data_dir, "theta_{}.csv".format(len(round_participants))
        )
        data = pd.read_csv(
            data_file, header=None, delim_whitespace=True
        )  # 根据你的CSV文件分隔符调整
        return data.to_numpy()

    # 计算bifed_sv, participants_set == round_N
    def calculate_bifed_sv(self, participants_set, theta_matrix, feature_matrix):
        assert self.config is not None
        # Step 1: Calculate the BiFedSV for participants (p)
        bifed_sv_p = np.dot(feature_matrix, theta_matrix)
        # Create a dictionary to store the results
        bifed_sv = {}
        # Store BiFedSV for participants
        for i, participant in enumerate(participants_set):
            bifed_sv[participant] = bifed_sv_p[i]
        # Step 2: Find the historical optimal BiFedSV for non-participants (n - p)
        non_participants = set(range(self.config.worker_number)) - set(participants_set)
        bifed_sv_non_p = {}
        for i in non_participants:
            # Find the optimal historical BiFedSV value for the non-participants
            bifed_sv_non_p[i] = self.find_optimal_bifed_sv()
        # Step 3: Update BiFedSV for all participants and non-participants
        for i in non_participants:
            bifed_sv[i] = bifed_sv_non_p[i]
        # Step 4: Combine and return the final BiFedSV Φ^(r) for all participants
        return bifed_sv

    def find_optimal_bifed_sv(self):
        subsets = defaultdict(list)
        for S_Vs in self.shapley_values.values():
            # 假设S_Vs的长度与worker_number相同，且按顺序排列
            for i, S_V in enumerate(S_Vs):
                subsets[i].append(S_V)

                # 创建一个字典来存储最优BiFedSV
        optimal_bifed_sv = {}
        for i, S_Vs in subsets.items():
            if S_Vs:  # 确保列表不为空
                optimal_bifed_sv[i] = max(S_Vs)  # 取最大值作为最优值

        return optimal_bifed_sv

    # 注意：这里每轮的参与者集合是动态变化的
    def _compute_impl(self, round_index: int) -> None:
        # 获取当前轮次的参与者
        round_participants = self.players
        print(f"Algorithm round participants: {round_participants}")
        subsets = set()

        # 计算每个子集的效用
        for subset in self.generate_subsets(round_participants):
            if not subset:
                continue
            subset = tuple(sorted(subset))
            subsets.add(subset)

        result_metrics = {s: self.metric_fun(s) for s in subsets}

        # 从 CSV 文件中读取 theta_matrix
        theta_matrix = self.read_matrix_from_csv(round_participants)

        # 计算 BiFed Shapley 值
        subset_pairs = self.generate_subset_pairs(round_participants)
        sorted_pairs = sorted(
            subset_pairs,
            key=lambda pair: self.sort_key(pair, round_participants),
        )

        v_ST = {}
        feature_matrix = []

        for S, T in sorted_pairs:
            remaining_subsets = self.generate_remaining_subsets(
                round_participants, S, T
            )
            cumulative_utility = self.calculate_cumulative_utility(
                S, remaining_subsets, result_metrics
            )
            matrix = cumulative_utility / (
                2 ** len(round_participants - set(S) - set(T))
            )
            v_ST[(S, T)] = matrix
            feature_matrix.append(matrix)

        # 计算 BiFed Shapley 值
        bifed_sv = self.calculate_bifed_sv(
            round_participants, theta_matrix, feature_matrix
        )
        self.bifed_sv = bifed_sv  # 更新 bifed_sv
        log_info("bifed_sv: %s", bifed_sv)

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
