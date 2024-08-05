from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .algorithm import ComFedShapleyValueAlgorithm


class ComFedShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=ComFedShapleyValueAlgorithm(server=self))

    def select_workers(self) -> set[int]:
        if self.round_index == 1:
            result = set(range(self.worker_number))
            self.selection_result[self.round_index] = result
            return result
        return super().select_workers()
