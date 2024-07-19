
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
        self.metrics: dict[int, dict] = {}  # 新增属性来保存metrics字典
        self.config: None | DistributedTrainingConfig = None

    def _compute_impl(self, round_number: int) -> None:
        self.metrics[round_number] = {}
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

    # 按照区间shapley值的计算公式要求生成所有子集的正确排序
    @classmethod
    def sorted_subsets(cls, nums) -> list[tuple]:
        if not nums:
            return [()]
        subsets_without_first = cls.sorted_subsets(nums[1:])
        subsets_with_first = [(nums[0],) + subset for subset in subsets_without_first]
        return subsets_with_first + subsets_without_first

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
