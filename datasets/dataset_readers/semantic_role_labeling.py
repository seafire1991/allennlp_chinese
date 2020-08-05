import logging
from typing import Dict, List

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
import sys

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("c_srl")
class SrlReader(DatasetReader):
    """
    This DatasetReader is designed to read in the Chinese semantic role labelling.
    It returns a dataset of instances with the
    following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                line_list = [per_word.split("/") for per_word in line.split(" ")]
                tokens = [Token(token[0]) for token in line_list]
                tags = [tag[2] for tag in line_list if len(tag) > 2]
                verb_indicator = [1 if v[1] == "VV" else 0 for v in line_list]
                yield self.text_to_instance(tokens, verb_indicator, tags)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)
        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)

        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokens[verb_label.index(1)].text
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens],
                                            "verb": verb})
        return Instance(fields)
