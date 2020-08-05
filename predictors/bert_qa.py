from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('bert-for-qa')
class BertQAPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.reading_comprehension.BertForQuestionAnswering` model.
    """

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage" : passage, "question" : question})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        paragraph_json = json_dict["documents"][json_dict["answer_docs"][0]] if "answer_docs" in json_dict and len(
            json_dict["answer_docs"]) > 0 else 0
        if paragraph_json == 0: return 0;
        paragraph = " ".join(paragraph_json["segmented_paragraphs"][paragraph_json["most_related_para"]])
        question_text = " ".join(json_dict["segmented_question"])
        answer_texts = json_dict["segmented_answers"][-1] if len(json_dict["segmented_answers"]) > 0 else 0
        if answer_texts == 0: return 0;
        question_text = question_text
        passage_text = paragraph
        return self._dataset_reader.text_to_instance(question_text, passage_text)
