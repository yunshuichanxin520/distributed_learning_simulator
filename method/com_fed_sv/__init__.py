from distributed_learning_simulation import (AggregationWorker,
                                             CentralizedAlgorithmFactory)

from .server import ComFedShapleyValueServer

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="com_fed_shapley_value",
    client_cls=AggregationWorker,
    server_cls=ComFedShapleyValueServer,
)
