from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .algorithm import BiFedShapleyValueAlgorithm


class BiFedSVServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        # from .algorithm import BiFedShapleyValueAlgorithm  # 延迟导入，避免循环导入问题
        super().__init__(**kwargs, algorithm=BiFedShapleyValueAlgorithm(server=self))

    def select_workers(self) -> set[int]:
        # 如果这是第一次调用，使用父类的 select_workers 方法选择参与者
        if not self.selection_result:
            return super().select_workers()

        print(f"Selection result before update: {self.selection_result}")

        assert isinstance(self.algorithm, BiFedShapleyValueAlgorithm)
        bifed_sv_algorithm = self.algorithm.sv_algorithm

        round_participants = set()

        if bifed_sv_algorithm.bifed_sv:
            # 过滤掉非数值类型的 BiFed Shapley 值
            filtered_bifed_sv = {
                key: value for key, value in bifed_sv_algorithm.bifed_sv.items()
                if isinstance(value, (float, int))
            }

            # 检查过滤后的字典是否为空
            if not filtered_bifed_sv:
                print("Warning: No valid numeric Shapley values found.")
                return round_participants

            # 计算 BiFed Shapley 值的平均值
            avg_shapley_value = sum(filtered_bifed_sv.values()) / len(filtered_bifed_sv)
            print(f"Average Shapley value: {avg_shapley_value}")

            # 筛选 BiFed Shapley 值大于等于平均值的参与者
            for key, value in filtered_bifed_sv.items():
                if value >= avg_shapley_value:
                    round_participants.add(key)

            print(f"Round {self.round_index} participants: {round_participants}")
        else:
            print("Warning: bifed_sv is empty. No participants selected.")

        # 保存本轮选择结果
        self.selection_result[self.round_index] = round_participants
        print(f"Selection result for round {self.round_index}: {round_participants}")

        return round_participants
