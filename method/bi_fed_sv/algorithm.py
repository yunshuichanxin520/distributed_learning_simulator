from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.shapely_value.shapley_value import \
    RoundBasedShapleyValue
from distributed_learning_simulation import DistributedTrainingConfig
from distributed_learning_simulator.algorithm.shapley_value_algorithm import \
    ShapleyValueAlgorithm


class BiFedShapleyValue(RoundBasedShapleyValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shapley_values: list = []
        self.metrics: dict[int, dict] = {}
        self.subset_metrics: dict[int, dict] = {}
        self.config: None | DistributedTrainingConfig = None

    def _compute_impl(self, round_number: int) -> None:
        self.metrics[round_number] = {}
        if 
        subsets = set()
        for subset in self.powerset(self.complete_player_indices):
            subset = tuple(sorted(subset))
            if not subset:
                metric = 0
                self.metrics[round_number][subset] = metric
                log_info("round %s subset %s metric %s", round_number, subset, metric)
            else:
                subsets.add(subset)
        assert self.batch_metric_fun is not None
        result_metrics: dict = self.batch_metric_fun(subsets)
        for subset, metric in result_metrics.items():
            log_info("round %s subset %s metric %s", round_number, subset, metric)

        self.metrics[round_number].update(result_metrics)

    def get_result(self) -> list:
        return self.shapley_values


class BiFedSVAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(BiFedShapleyValue, *args, **kwargs)

    @property
    def sv_algorithm(self) -> BiFedShapleyValue:
        algorithm = super().sv_algorithm
        assert isinstance(algorithm, BiFedShapleyValue)
        algorithm.config = self.config
        return algorithm
