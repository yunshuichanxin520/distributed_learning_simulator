from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .algorithm import BiFedShapleyValueAlgorithm

class BiFedSVServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=BiFedShapleyValueAlgorithm(server=self))
        self.selection_result = {}  # 保存每轮选择的参与者集合

    def select_workers(self) -> set[int]:
        # 如果是刚开始分发模型，调用默认实现
        if not self.selection_result:
            return super().select_workers()
        # 初始化当前轮次的参与者集合
        assert isinstance(self.algorithm, BiFedShapleyValueAlgorithm)
        bifed_sv_algorithm = self.algorithm.sv_algorithm  # 获取BiFedShapleyValue实例
        round_participants = set()

        # 根据 BiFed Shapley 值筛选符合条件的参与者
        for key, value in bifed_sv_algorithm.bifed_sv.items():
            # 筛选满足条件的参与者
            if value >= 0 and value >= sum(bifed_sv_algorithm.bifed_sv.values()) / len(bifed_sv_algorithm.bifed_sv):
                round_participants.add(key)

        # 保存当前轮次选择结果到 selection_result
        self.selection_result[self.round_index] = round_participants

        return round_participants

# class BiFedSVServer(ShapleyValueServer):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs, algorithm=BiFedShapleyValueAlgorithm(server=self))
#         # self.round_participants = {}
#         self.selection_result = {}  # 保存每轮选择的参与者集合
#
#     def select_workers(self) -> set[int]:
#         # 初始化当前轮次的参与者集合
#         assert isinstance(self.algorithm, BiFedShapleyValueAlgorithm)
#         bifed_sv = self.algorithm.sv_algorithm
#         round_participants = set()
#
#         # 如果是第一轮，确保参与者集合非空
#         if self.round_index == 1:
#             if not bifed_sv:
#                 # 如果 bifed_sv 为空，手动添加所有参与者
#                 round_participants.update(range(self.config.worker_number))
#             else:
#                 round_participants.update(bifed_sv.keys())
#         else:
#             # 否则根据 BiFed Shapley 值筛选符合条件的参与者
#             for key, value in bifed_sv.items():
#                 if value >= 0 and value >= sum(bifed_sv.values()) / len(bifed_sv):
#                     round_participants.add(key)
#
#         # 保存当前轮次选择结果到 selection_result
#         self.selection_result[self.round_index] = round_participants
#
#         return round_participants

