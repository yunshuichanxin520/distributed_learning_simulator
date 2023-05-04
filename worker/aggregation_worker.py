import os

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException
from cyy_torch_toolbox.model_util import ModelUtil
from util.model import load_parameters
from util.model_cache import ModelCache

from .client import Client


class AggregationWorker(Client):
    def __init__(self, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self._aggregation_time = ExecutorHookPoint.AFTER_EXECUTE
        self._reuse_learning_rate: bool = False
        self._choose_model_by_validation: bool = True
        self._send_parameter_diff: bool = True
        self._model_cache: ModelCache = ModelCache()

    def _register_aggregation(self):
        get_logger().debug("use aggregation_time %s", self._aggregation_time)
        self.trainer.remove_named_hook(name="aggregation")
        self.trainer.append_named_hook(
            self._aggregation_time,
            "aggregation",
            self.__aggregation_impl,
        )

    def _should_aggregate(self, **kwargs):
        return True

    def __aggregation_impl(self, **kwargs):
        if not self._should_aggregate(**kwargs):
            return
        sent_data = self._get_sent_data()
        self._aggregation(sent_data=sent_data)

    def _aggregation(self, sent_data):
        raise NotImplementedError()

    def _get_sent_data(self) -> dict:
        sent_data = {
            "dataset_size": self.trainer.dataset_size,
        }
        model_util = self.trainer.model_util
        if self._choose_model_by_validation:
            get_logger().debug("use best model")
            assert self.trainer.best_model is not None
            model_util = ModelUtil(self.trainer.best_model)
            sent_data["model_epoch"] = self.trainer.best_epoch

        parameter = model_util.get_parameter_dict()
        if self._send_parameter_diff:
            sent_data["parameter_diff"] = self._model_cache.get_parameter_diff(
                parameter
            )
        else:
            sent_data["parameter"] = parameter
        return sent_data

    def _load_result_from_server(self, result):
        if "end_training" in result:
            self._force_stop = True
            raise StopExecutingException()
        model_path = os.path.join(
            self.save_dir, "aggregated_model", f"round_{self._round_num}.pk"
        )
        if "parameter_path" in result:
            self._model_cache.load_file(result["parameter_path"])
        elif "parameter" in result:
            self._model_cache.cache_parameter_dict(result["parameter"], path=model_path)
        elif "partial_parameter" in result:
            self._model_cache.cache_parameter_dict(
                result["partial_parameter"], path=model_path
            )
        elif "parameter_diff" in result:
            self._model_cache.add_parameter_diff(
                result["parameter_diff"], path=model_path
            )
        else:
            raise NotImplementedError()
        load_parameters(
            trainer=self.trainer,
            parameter_dict=self._model_cache.get_parameter_dict,
            reuse_learning_rate=self._reuse_learning_rate,
        )
