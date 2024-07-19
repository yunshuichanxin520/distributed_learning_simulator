from distributed_learning_simulation import (AggregationWorker,
                                             CentralizedAlgorithmFactory)

from .server import IntervalShapleyValueServer

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="interval_shapley_value",
    client_cls=AggregationWorker,
    server_cls=IntervalShapleyValueServer,
)
