from .interval_shapley_value_algorithm import IntervalShapleyValueAlgorithm
from .shapley_value_server import ShapleyValueServer


class IntervalShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=IntervalShapleyValueAlgorithm(server=self))
