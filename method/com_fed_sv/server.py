from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .algorithm import ComFedShapleyValueAlgorithm


class ComFedShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=ComFedShapleyValueAlgorithm(server=self))

    def select_workers(self) -> set[int]:
        if not self.selection_result:
            result = set(range(self.worker_number))
            self.selection_result[-1] = result
            return result
        return super().select_workers()

