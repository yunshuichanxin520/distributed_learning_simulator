from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer
from cyy_naive_lib.log import log_info
from .algorithm import BiFedShapleyValueAlgorithm


class BiFedSVServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=BiFedShapleyValueAlgorithm(server=self))
        # self.selection_result = {}
        # self.round_number = 1

    def select_workers(self) -> set[int]:
        if not self.selection_result:
            # 初始化当前轮次的参与者集合
            round_participants = set()
            # 如果是第一轮，直接将所有参与者添加到当前轮次的集合中
            if self.round_index == 1:
                round_participants.update(self.shapley_values.keys())
                log_info("round_participants %s ", round_participants)
            else:
                # 否则，根据条件筛选参与者
                for key, value in self.shapley_values.items():
                    # 检查条件：φ^(r)_i >= 0 且 φ^(r)_i >= φ^(r)'_i，其中φ^(r)'_i为平均值
                    if value >= 0 and value >= sum(self.shapley_values.values()) / len(self.shapley_values):
                        round_participants.add(key)
                log_info("round_participants %s ", round_participants)
                        # 将当前轮次的参与者集合存入selection_result中
            # self.selection_result[self.round_number] = round_participants
            # 假设有某种方式需要更新轮次号，这里只是简单递增
            self.round_number += 1
            # 直接返回当前轮次的选择结果（如果需要）
            return round_participants
        # Return the selected participants set N^(r+1)
        return super().select_workers()



