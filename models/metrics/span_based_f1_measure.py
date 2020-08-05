from typing import Dict, List, Optional, Set, Callable, Tuple
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedStringSpan
import sys


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return ' '.join(self.tag_sequence)

TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]

def bio_tags_to_spans(tag_sequence: List[str],
                      classes_to_ignore: List[str] = None) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans. This function works properly when
    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").

    Parameters
    ----------
    tag_sequence : List[str], required.
        The integer class labels for a sequence.
    classes_to_ignore : List[str], optional (default = None).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.

    Returns
    -------
    spans : List[TypedStringSpan]
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "E", "O", "S", "r"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "S" or bio_tag == "E":
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif (bio_tag == "I") and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

def bmeso_tags_to_spans(tag_sequence: List[str],
                      classes_to_ignore: List[str] = None) -> List[TypedStringSpan]:
    """
        Given a sequence corresponding to BMES tags, extracts spans.
        Spans are inclusive and can be of zero length, representing a single word span.
        Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
        as otherwise it is possible to get a perfect precision score whilst still predicting
        ill-formed spans in addition to the correct spans.
        This function works properly when the spans are unlabeled (i.e., your labels are
        simply "B", "M", "E" and "S").

        Parameters
        ----------
        tag_sequence : List[str], required.
            The integer class labels for a sequence.
        classes_to_ignore : List[str], optional (default = None).
            A list of string class labels `excluding` the bio tag
            which should be ignored when extracting spans.

        Returns
        -------
        spans : List[TypedStringSpan]
            The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
            Note that the label `does not` contain any BIO tag prefixes.
        """

    def extract_bies_tag_label(text):
        bmes_tag = "O" if text == "O" else text[2]
        label = text[0]
        return bmes_tag, label

    spans: List[Tuple[str, List[int]]] = []
    prev_bmes_tag: Optional[str] = None
    for index, tag in enumerate(tag_sequence):
        bmes_tag, label = extract_bies_tag_label(tag)
        if bmes_tag in ('B', 'S'):
            # Regardless of tag, we start a new span when reaching B & S.
            spans.append(
                (label, [index, index])
            )
        elif bmes_tag in ('M', 'E') and prev_bmes_tag in ('B', 'M') and spans[-1][0] == label:
            # Only expand the span if
            # 1. Valid transition: B/M -> M/E.
            # 2. Matched label.
            spans[-1][1][1] = index
        elif bmes_tag in ('o'):
            spans.append(
                (tag, [index, index])
            )
        else:
            # Best effort split for invalid span.
            spans.append(
                (label, [index, index])
            )
        # update previous BMES tag.
        prev_bmes_tag = bmes_tag

    classes_to_ignore = classes_to_ignore or []
    return [
        # to tuple.
        (span[0], (span[1][0], span[1][1]))
        for span in spans
        if span[0] not in classes_to_ignore
    ]

# tags = ['c-B', 'c-M', 'c-M','c-E', 'o', 'o', 'b-S', 'o', 'o', 'c-S', 'o', 'o', 'o', 'o', 'b-B', 'b-E', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
#
# print(bmeso_tags_to_spans(tags))
# sys.exit()

def bies_tags_to_spans(tag_sequence: List[str],
                       classes_to_ignore: List[str] = None) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BMES tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans.
    This function works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "M", "E" and "S").

    Parameters
    ----------
    tag_sequence : List[str], required.
        The integer class labels for a sequence.
    classes_to_ignore : List[str], optional (default = None).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.

    Returns
    -------
    spans : List[TypedStringSpan]
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """

    def extract_bies_tag_label(text):
        bmes_tag = text[0]
        label = text[2:]
        return bmes_tag, label

    spans: List[Tuple[str, List[int]]] = []
    prev_bmes_tag: Optional[str] = None
    for index, tag in enumerate(tag_sequence):
        bmes_tag, label = extract_bies_tag_label(tag)
        if bmes_tag in ('B', 'S'):
            # Regardless of tag, we start a new span when reaching B & S.
            spans.append(
                    (label, [index, index])
                    )
        elif bmes_tag in ('I', 'E') and prev_bmes_tag in ('B', 'I') and spans[-1][0] == label:
            # Only expand the span if
            # 1. Valid transition: B/M -> M/E.
            # 2. Matched label.
            spans[-1][1][1] = index
        elif bmes_tag in ('r', 'O'):
            spans.append(
                (tag, [index, index])
            )
        else:
            # Best effort split for invalid span.
            spans.append(
                    (label, [index, index])
                    )
        # update previous BMES tag.
        prev_bmes_tag = bmes_tag

    classes_to_ignore = classes_to_ignore or []
    return [
            # to tuple.
            (span[0], (span[1][0], span[1][1]))
            for span in spans
            if span[0] not in classes_to_ignore
            ]
# tags = ['B-ARGM-LOC', 'I-ARGM-LOC', 'I-ARGM-LOC', 'E-ARGM-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-ARGM-ADV', 'rel', 'B-ARG1', 'I-ARG1', 'E-ARG1', 'O']
#
# print(bies_tags_to_spans(tags))
# sys.exit()


class SpanBasedF1Measure(Metric):
    """
    The Conll SRL metrics are based on exact span matching. This metric
    implements span-based precision and recall metrics for a BIO tagging
    scheme. It will produce precision, recall and F1 measures per tag, as
    well as overall statistics. Note that the implementation of this metric
    is not exactly the same as the perl script used to evaluate the CONLL 2005
    data - particularly, it does not consider continuations or reference spans
    as constituents of the original span. However, it is a close proxy, which
    can be helpful for judging model performance during training. This metric
    works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "I", "O" if using the "BIO" label encoding).

    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 tag_namespace: str = "tags",
                 ignore_classes: List[str] = None,
                 label_encoding: Optional[str] = "BMESO",
                 tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None) -> None:
        """
        Parameters
        ----------
        vocabulary : ``Vocabulary``, required.
            A vocabulary containing the tag namespace.
        tag_namespace : str, required.
            This metric assumes that a BIO format is used in which the
            labels are of the format: ["B-LABEL", "I-LABEL"].
        ignore_classes : List[str], optional.
            Span labels which will be ignored when computing span metrics.
            A "span label" is the part that comes after the BIO label, so it
            would be "ARG1" for the tag "B-ARG1". For example by passing:

             ``ignore_classes=["V"]``
            the following sequence would not consider the "V" span at index (2, 3)
            when computing the precision, recall and F1 metrics.

            ["O", "O", "B-V", "I-V", "B-ARG1", "I-ARG1"]

            This is helpful for instance, to avoid computing metrics for "V"
            spans in a BIO tagging scheme which are typically not included.
        label_encoding : ``str``, optional (default = "BIO")
            The encoding used to specify label span endpoints in the sequence.
            Valid options are "BIO", "IOB1", "BIOUL" or "BMES".
        tags_to_spans_function: ``Callable``, optional (default = ``None``)
            If ``label_encoding`` is ``None``, ``tags_to_spans_function`` will be
            used to generate spans.
        """
        if label_encoding and tags_to_spans_function:
            raise ConfigurationError(
                    'Both label_encoding and tags_to_spans_function are provided. '
                    'Set "label_encoding=None" explicitly to enable tags_to_spans_function.'
                    )
        if label_encoding:
            if label_encoding not in ["BIO", "IOB1", "BIOUL", "BMES", "BMESO"]:
                raise ConfigurationError("Unknown label encoding - expected 'BIO', 'IOB1', 'BIOUL', 'BMES'.")
        elif tags_to_spans_function is None:
            raise ConfigurationError(
                    'At least one of the (label_encoding, tags_to_spans_function) should be provided.'
                    )

        self._label_encoding = label_encoding
        self._tags_to_spans_function = tags_to_spans_function
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._ignore_classes: List[str] = ignore_classes or []

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 prediction_map: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        prediction_map: ``torch.Tensor``, optional (default = None).
            A tensor of size (batch_size, num_classes) which provides a mapping from the index of predictions
            to the indices of the label vocabulary. If provided, the output label at each timestep will be
            ``vocabulary.get_index_to_token_vocabulary(prediction_map[batch, argmax(predictions[batch, t]))``,
            rather than simply ``vocabulary.get_index_to_token_vocabulary(argmax(predictions[batch, t]))``.
            This is useful in cases where each Instance in the dataset is associated with a different possible
            subset of labels from a large label-space (IE FrameNet, where each frame has a different set of
            possible roles associated with it).
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask, prediction_map = self.unwrap_to_tensors(predictions,
                                                                                gold_labels,
                                                                                mask, prediction_map)

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to SpanBasedF1Measure contains an "
                                     "id >= {}, the number of classes.".format(num_classes))

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.max(-1)[1]

        if prediction_map is not None:
            argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
            gold_labels = torch.gather(prediction_map, 1, gold_labels.long())

        argmax_predictions = argmax_predictions.float()

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]

            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue

            predicted_string_labels = [self._label_vocabulary[label_id]
                                       for label_id in sequence_prediction[:length].tolist()]
            gold_string_labels = [self._label_vocabulary[label_id]
                                  for label_id in sequence_gold_label[:length].tolist()]
            # print(predicted_string_labels)
            # print(gold_string_labels)

            tags_to_spans_function = None
            # `label_encoding` is empty and `tags_to_spans_function` is provided.
            if self._label_encoding is None and self._tags_to_spans_function:
                tags_to_spans_function = self._tags_to_spans_function
            # Search by `label_encoding`.
            elif self._label_encoding == "BIO":
                tags_to_spans_function = bies_tags_to_spans
            elif self._label_encoding == "BMESO":
                tags_to_spans_function = bmeso_tags_to_spans
            predicted_spans = tags_to_spans_function(predicted_string_labels, self._ignore_classes)
            gold_spans = tags_to_spans_function(gold_string_labels, self._ignore_classes)
            predicted_spans = self._handle_continued_spans(predicted_spans)
            gold_spans = self._handle_continued_spans(gold_spans)
            # print(predicted_string_labels)
            # print(predicted_spans)
            # print(gold_spans)
            #print(".....")
            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    @staticmethod
    def _handle_continued_spans(spans: List[TypedStringSpan]) -> List[TypedStringSpan]:
        """
        The official CONLL 2012 evaluation script for SRL treats continued spans (i.e spans which
        have a `C-` prepended to another valid tag) as part of the span that they are continuing.
        This is basically a massive hack to allow SRL models which produce a linear sequence of
        predictions to do something close to structured prediction. However, this means that to
        compute the metric, these continuation spans need to be merged into the span to which
        they refer. The way this is done is to simply consider the span for the continued argument
        to start at the start index of the first occurrence of the span and end at the end index
        of the last occurrence of the span. Handling this is important, because predicting continued
        spans is difficult and typically will effect overall average F1 score by ~ 2 points.

        Parameters
        ----------
        spans : ``List[TypedStringSpan]``, required.
            A list of (label, (start, end)) spans.

        Returns
        -------
        A ``List[TypedStringSpan]`` with continued arguments replaced with a single span.
        """
        span_set: Set[TypedStringSpan] = set(spans)
        continued_labels: List[str] = [label[2:] for (label, span) in span_set if label.startswith("C-")]
        for label in continued_labels:
            continued_spans = {span for span in span_set if label in span[0]}

            span_start = min(span[1][0] for span in continued_spans)
            span_end = max(span[1][1] for span in continued_spans)
            replacement_span: TypedStringSpan = (label, (span_start, span_end))

            span_set.difference_update(continued_spans)
            span_set.add(replacement_span)

        return list(span_set)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
