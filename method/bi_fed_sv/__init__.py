from distributed_learning_simulation import (AggregationWorker,
                                             CentralizedAlgorithmFactory)

from .server import BiFedSVServer

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="bi_fed_shapley_value",
    client_cls=AggregationWorker,
    server_cls=BiFedSVServer,
)
