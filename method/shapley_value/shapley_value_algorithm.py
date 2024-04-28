import copy
import json
import os
from typing import Any, Iterable, Type

from cyy_naive_lib.log import get_logger
from distributed_learning_simulation import (AggregationServer,
                                             FedAVGAlgorithm, ParameterMessage)


class ShapleyValueAlgorithm(FedAVGAlgorithm):
    def __init__(
        self, sv_algorithm_cls: Type, server: AggregationServer, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._server: AggregationServer = server
        self.accumulate = False
        self.metric_type: str = "accuracy"
        self.sv_algorithm = None
        self.sv_algorithm_cls = sv_algorithm_cls
        self.shapley_values: dict[int, dict] = {}
        self.shapley_values_S: dict[int, dict] = {}

    @property
    def choose_best_subset(self) -> bool:
        return self.config.algorithm_kwargs.get("choose_best_subset", False)

    def __get_players(self) -> Iterable[int]:
        return sorted(self._all_worker_data.keys())

    def aggregate_worker_data(self) -> ParameterMessage:
        if self.sv_algorithm is None:
            assert self._server.round_index == 1
            self.sv_algorithm = self.sv_algorithm_cls(
                players=self.__get_players(),
                last_round_metric=self._server.performance_stat[
                    self._server.round_index - 1
                ][f"test_{self.metric_type}"],
            )
        assert self.sv_algorithm is not None
        self.sv_algorithm.set_metric_function(self._get_subset_metric)
        self.sv_algorithm.compute(round_number=self._server.round_index)
        self.shapley_values[self._server.round_index] = copy.deepcopy(
            self.sv_algorithm.shapley_values
        )
        if self.choose_best_subset:
            self.shapley_values_S[self._server.round_index] = (
                self.sv_algorithm.shapley_values_S
            )
            best_subset: set = set(
                self.shapley_values_S[self._server.round_index].keys()
            )
            if best_subset:
                get_logger().warning("use subset %s", best_subset)
                self._all_worker_data = {
                    k: v for k, v in self._all_worker_data.items() if k in best_subset
                }
        return super().aggregate_worker_data()

    def _get_subset_metric(self, subset) -> dict:
        assert subset
        aggregated_parameter = FedAVGAlgorithm.aggregate_parameter(
            {k: v for k, v in self._all_worker_data.items() if k in subset}
        )

        assert aggregated_parameter
        return self._server.get_metric(
            aggregated_parameter, log_performance_metric=False
        )[self.metric_type]

    def exit(self) -> None:
        assert self.sv_algorithm is not None
        if hasattr(self.sv_algorithm, "exit"):
            self.sv_algorithm.exit(config=self.config)
            self.shapley_values = copy.deepcopy(self.sv_algorithm.shapley_values)
        with open(
            os.path.join(self.config.save_dir, "shapley_values.json"),
            "wt",
            encoding="utf8",
        ) as f:
            json.dump(self.shapley_values, f)
        if self.choose_best_subset:
            with open(
                os.path.join(self.config.save_dir, "shapley_values_S.json"),
                "wt",
                encoding="utf8",
            ) as f:
                json.dump(self.shapley_values_S, f)
