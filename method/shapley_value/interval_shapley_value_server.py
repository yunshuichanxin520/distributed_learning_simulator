from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .interval_shapley_value_algorithm import IntervalShapleyValueAlgorithm


class IntervalShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=IntervalShapleyValueAlgorithm(server=self))
