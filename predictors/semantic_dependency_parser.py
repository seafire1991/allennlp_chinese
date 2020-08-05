from typing import List

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from bita.nlp import segment

@Predictor.register('c-sdp')
class BiaffineDependencyParserPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.

        Returns
        -------
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        JSON格式如下:``{"sentence": "..."}``.
        这里需要用到分词和词性标注的预训练模型，因此词性标注和分词模型是bita.nlp的基础功能
        """
        root = self._dataset_reader.ROOT
        tokens = [word_pos.split("/") for word_pos in segment(json_dict["sentence"], ispos=True).split(" ")]
        tokens.insert(0, [root, ""])
        sentence_text = [token[0] for token in tokens]
        pos_tags = [token[1] for token in tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        new_outputs = {}
        new_outputs["tokens"] = outputs["tokens"]
        new_outputs["arcs"] = outputs["arcs"]
        new_outputs["arc_tags"] = outputs["arc_tags"]
        return sanitize(new_outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            words = output["words"]
            pos = output["pos"]
            heads = output["predicted_heads"]
            tags = output["predicted_dependencies"]
            output["hierplane_tree"] = self._build_hierplane_tree(words, heads, tags, pos)
        return sanitize(outputs)