from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer
from .algorithm import BiFedShapleyValueAlgorithm

class BiFedSVServer(ShapleyValueServer):

    # works=select_workers()
    def __init__(self, **kwargs) -> None:
        # from .algorithm import BiFedShapleyValueAlgorithm  # 延迟导入，避免循环导入问题
        super().__init__(**kwargs, algorithm=BiFedShapleyValueAlgorithm(server=self))
        self.selection_result = {}  # 用于保存每一轮的参与者选择结果
        # works # 用于保存每一轮的参与者选择结果

    def select_workers(self) -> set[int]:
        # 第一次调用时，直接使用父类的 select_workers
        if not self.selection_result:
            return super().select_workers()

        # 输出之前保存的选择结果
        print(f"Selection result before update: {self.selection_result}")

        # 导入 BiFedShapleyValueAlgorithm，确保实例是正确类型
        # from .algorithm import BiFedShapleyValueAlgorithm
        assert isinstance(self.algorithm, BiFedShapleyValueAlgorithm)
        bifed_sv_algorithm = self.algorithm.sv_algorithm  # 获取 Shapley 值算法实例

        # 初始化当前轮次的参与者集合
        round_participants = set()
        print(f"Round {self.round_index} participants: {round_participants}")

        # 根据 BiFed Shapley 值筛选参与者
        for key, value in bifed_sv_algorithm.bifed_sv.items():
            # 选择符合条件的参与者（满足某些 Shapley 值阈值）
            if value >= 0 and value >= sum(bifed_sv_algorithm.bifed_sv.values()) / len(bifed_sv_algorithm.bifed_sv):
                round_participants.add(key)

        # 保存选择结果
        self.selection_result[self.round_index] = round_participants
        print(f"Selection result for round {self.round_index}: {round_participants}")

        return round_participants



