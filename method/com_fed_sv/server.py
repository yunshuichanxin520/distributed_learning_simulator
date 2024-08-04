from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .algorithm import ComFedShapleyValueAlgorithm


class ComFedShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=ComFedShapleyValueAlgorithm(server=self))
