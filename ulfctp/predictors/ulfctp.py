from overrides import overrides
import sys
from stog.data import Instance
from stog.utils.exception_hook import ExceptionHook
from stog.predictors.predictor import Predictor, DEFAULT_PREDICTORS
from ulfctp.utils.string import JsonDict

sys.excepthook = ExceptionHook()

DEFAULT_PREDICTORS['ULFCTP'] = 'ULFCTP'

@Predictor.register('ULFCTP')
class ULFCTPPredictor(Predictor):
    """
    Predictor for the :class:`~ulfctp.models.ulfctp. model.
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @overrides
    def predict_batch_instance(self, instances):
        return self._model.forward_on_instances(instances)

    @overrides
    def dump_line(self, output):
        return (output['output_ulfamr_string'] + '\n\n',
                output['output_ulf_string'] + '\n\n')

