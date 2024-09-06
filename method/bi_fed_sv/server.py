from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .algorithm import BiFedShapleyValueAlgorithm


class BiFedSVServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=BiFedShapleyValueAlgorithm(server=self))
        # self.round_participants = {}
        self.selection_result = {}  # 保存每轮选择的参与者集合

    def select_workers(self) -> set[int]:
        # 通过 sv_algorithm 访问 BiFedShapleyValue 中的 bifed_sv
        bifed_sv = self.algorithm.sv_algorithm.bifed_sv
        round_participants = set()

        # 如果是第一轮，确保参与者集合非空
        if self.round_index == 1:
            if not bifed_sv:
                # 如果 bifed_sv 为空，手动添加所有参与者
                round_participants.update(range(self.config.worker_number))
            else:
                round_participants.update(bifed_sv.keys())
        else:
            # 否则根据 BiFed Shapley 值筛选符合条件的参与者
            for key, value in bifed_sv.items():
                if value >= 0 and value >= sum(bifed_sv.values()) / len(bifed_sv):
                    round_participants.add(key)

        # 保存当前轮次选择结果到 selection_result
        self.selection_result[self.round_index] = round_participants

        return round_participants


    # def select_workers(self) -> set[int]:
    #     if not self.selection_result:
    #         # 初始化当前轮次的参与者集合
    #         round_participants = set()
    #         # 如果是第一轮，直接将所有参与者添加到当前轮次的集合中
    #         if self.round_index == 1:
    #             round_participants.update(self.shapley_values.keys())
    #             log_info("round_participants %s ", round_participants)
    #         else:
    #             # 否则，根据条件筛选参与者
    #             for key, value in self.shapley_values.items():
    #                 # 检查条件：φ^(r)_i >= 0 且 φ^(r)_i >= φ^(r)'_i，其中φ^(r)'_i为平均值
    #                 if value >= 0 and value >= sum(self.shapley_values.values()) / len(self.shapley_values):
    #                     round_participants.add(key)
    #             log_info("round_participants %s ", round_participants)
    #             # 将当前轮次的参与者集合存入selection_result中
    #         # self.selection_result[self.round_number] = round_participants
    #         # 假设有某种方式需要更新轮次号，这里只是简单递增
    #         self.round_number += 1
    #         # 直接返回当前轮次的选择结果（如果需要）
    #         return round_participants
    #     # Return the selected participants set N^(r+1)
    #     return super().select_workers()