from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .algorithm import BiFedShapleyValueAlgorithm


class BiFedSVServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=BiFedShapleyValueAlgorithm(server=self))
        self.round_participants = {}

    # def server_client_bidirectional_selection(self, bifed_sv):
    #     # Initialize participant set N^(r+1) as an empty set
    #
    #     if self.round_number == 1:
    #         for key in bifed_sv:
    #             round_participants.add(key)
    #     else:
    #         # Iterate over each participant i in N (all participants in the bifed_sv dictionary)
    #
    #         for key in bifed_sv:
    #             # Check if φ^(r)_i >= 0 and φ^(r)_i >= φ^(r)'_i
    #             # expected_sv[i]: sum(bifed_sv)/len(bifed_sv)
    #             if bifed_sv[key] >= 0 and bifed_sv[key] >= sum(bifed_sv)/len(bifed_sv):
    #                 # Add participant i to N^(r+1)
    #                 round_participants.add(key)
    def select_workers(self) -> set[int]:
        # 初始化当前轮次的参与者集合
        bifed_sv = self.algorithm.bifed_sv
        round_participants = set()

        # 如果是第一轮，直接将所有参与者添加到当前轮次的集合中
        if self.round_index == 1:
            round_participants.update(bifed_sv.keys())
        else:
            # 否则，根据条件筛选参与者
            for key, value in bifed_sv.items():
                # 检查条件：φ^(r)_i >= 0 且 φ^(r)_i >= φ^(r)'_i，其中φ^(r)'_i为平均值
                if value >= 0 and value >= sum(bifed_sv.values()) / len(bifed_sv):
                    round_participants.add(key)

                    # 将当前轮次的参与者集合存入selection_result中
        self.selection_result[self.round_index] = round_participants

        # 直接返回当前轮次的选择结果（如果需要）
        return round_participants
