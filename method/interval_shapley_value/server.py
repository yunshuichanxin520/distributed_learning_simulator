from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .algorithm import IntervalShapleyValueAlgorithm


class IntervalShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=IntervalShapleyValueAlgorithm(server=self))
