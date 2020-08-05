from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from .utils import bmes_to_words
from .utils import format_postag_result
from .utils import format_ner_result

@Predictor.register('c-sentence-tagger')
class SentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        #return format_ner_result(self.predict_instance(instance))
        return self.predict_instance(instance)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        #sentence = " ".join(json_dict["sentence"].strip().replace(" ", ""))
        #print(sentence)
        tokens = self._tokenizer.split_words(sentence)
        #tokens = sentence
        return self._dataset_reader.text_to_instance(tokens)
