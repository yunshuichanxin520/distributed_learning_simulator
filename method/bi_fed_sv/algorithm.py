import itertools
import os
import numpy as np
import pandas as pd
from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.shapely_value.shapley_value import \
    RoundBasedShapleyValue
from distributed_learning_simulation import DistributedTrainingConfig
from distributed_learning_simulator.algorithm.shapley_value_algorithm import \
    ShapleyValueAlgorithm


class BiFedShapleyValue(RoundBasedShapleyValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bifed_sv: dict = {}  # 初始化 bifed_sv
        self.history_bifed_sv: dict[int, dict[int, float]] = {}  # 用于存储每轮的 bifed_sv
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
        round_participants = set(round_participants)
        remaining = round_participants - set(S) - set(T)
        return self.generate_subsets(remaining)

    # 计算 v(S ∪ A) 的累计效用
    def calculate_cumulative_utility(self, S, remaining_subsets, v_S):
        cumulative_utility = 0
        for subset in remaining_subsets:
            union_set = set(S).union(subset)
            sorted_union_set = tuple(sorted(union_set))

            # 检查 union_set 是否为空
            if not sorted_union_set:
                continue  # 如果是空集，跳过

            # 确保 v_S 中有这个子集的效用值
            if sorted_union_set in v_S:
                cumulative_utility += v_S[sorted_union_set]
            else:
                # 如果为空，记录一个警告，或者赋予默认值
                print(f"Warning: No utility found for subset {sorted_union_set}")
                # 或者提供一个默认的效用值（例如0）
                cumulative_utility += 0

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
        data_dir = os.path.join(os.path.dirname(__file__), "tmp", "theta_n")
        print(data_dir)
        data_file = os.path.join(
            data_dir, "theta_{}.csv".format(len(round_participants))
        )

        # 使用逗号作为分隔符读取数据
        data = pd.read_csv(data_file, sep=",", header=None, dtype=float)

        # 检查是否存在非数值内容，并将其填充为 0 或其他适当值
        if data.isnull().values.any():
            log_info("Null values found in theta_matrix, filling with zeros")
            data = data.fillna(0)  # 可以根据需求选择适当的填充值

        return data.to_numpy()

# 计算并保存每轮的 bifed_sv
    def calculate_bifed_sv(
            self, participants_set, theta_matrix, feature_matrix, round_index
    ):
        assert self.config is not None
        # Step 1: 计算参与者的 BiFed Shapley 值
        bifed_sv_p = np.dot(feature_matrix, theta_matrix)
        bifed_sv = {}
        for i, participant in enumerate(participants_set):
            bifed_sv[participant] = bifed_sv_p[i]

        # Step 2: 计算非参与者的最优历史 BiFed Shapley 值
        non_participants = set(range(self.config.worker_number)) - set(participants_set)
        bifed_sv_non_p = self.find_optimal_bifed_sv(round_index - 1, non_participants)

        # Step 3: 合并所有参与者和非参与者的 Shapley 值
        bifed_sv.update(bifed_sv_non_p)

        # Step 4: 保存当前轮次的 bifed_sv
        self.history_bifed_sv[round_index] = bifed_sv
        return bifed_sv

    # 实现 find_optimal_bifed_sv，找到之前轮次中每个worker的最大 Shapley 值
    def find_optimal_bifed_sv(
            self, max_round_index: int, non_participants: set
    ) -> dict[int, float]:
        optimal_bifed_sv = {}

        # 遍历每一个非参与者，寻找历史最优值
        for worker in non_participants:
            max_value = float('-inf')  # 设置初始最大值为负无穷
            # 遍历所有之前的轮次
            for round_idx in range(max_round_index + 1):
                if worker in self.history_bifed_sv.get(round_idx, {}):
                    max_value = max(
                        max_value,
                        self.history_bifed_sv[round_idx][worker]
                    )

            if max_value == float('-inf'):
                # 如果没有找到任何历史值，赋予默认值 0
                max_value = 0.0

            optimal_bifed_sv[worker] = max_value
        return optimal_bifed_sv

    # 修改 _compute_impl 调用 calculate_bifed_sv，传入 round_index
    def _compute_impl(self, round_index: int) -> None:
        round_participants = set(self.players)
        print(f"Algorithm round participants: {round_participants}")

        # 计算子集和效用
        subsets = set()
        for subset in self.generate_subsets(round_participants):
            if not subset:
                continue
            subset = tuple(sorted(subset))
            subsets.add(subset)

        result_metrics = {
            s: self.metric_fun(tuple(self.players.index(worker_id) for worker_id in s))
            for s in subsets
        }

        # 读取 theta_matrix
        theta_matrix = self.read_matrix_from_csv(round_participants)

        # 生成特征矩阵
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

        # 计算并保存 BiFed Shapley 值，传入 round_index
        bifed_sv = self.calculate_bifed_sv(
            round_participants, theta_matrix, feature_matrix, round_index
        )
        self.bifed_sv = bifed_sv  # 更新当前轮次的 bifed_sv
        log_info("bifed_sv: %s", bifed_sv)

    def get_result(self) -> dict:
        return {"round_shapley_values": self.history_bifed_sv}


class BiFedShapleyValueAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(BiFedShapleyValue, *args, **kwargs)

    @property
    def sv_algorithm(self) -> BiFedShapleyValue:
        algorithm = super().sv_algorithm
        assert isinstance(algorithm, BiFedShapleyValue)
        algorithm.config = self.config
        return algorithm
