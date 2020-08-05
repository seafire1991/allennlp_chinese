from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.fields import LabelField, TextField, Field, SequenceLabelField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_WORD_TAG_DELIMITER = "\t"

@DatasetReader.register("classification")
class ClassificationDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    TAG[TAB]Sequence \n ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    word_tag_delimiter: ``str``, optional (default=``"###"``)
        The text that separates each WORD from its TAG.
    token_delimiter: ``str``, optional (default=``None``)
        The text that separates each WORD-TAG pair from the next pair. If ``None``
        then the line will just be split on whitespace.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._word_tag_delimiter = word_tag_delimiter
        self._tokenizer = tokenizer or WordTokenizer()

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r", encoding="utf-8") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")

                # skip blank lines
                if not line:
                    continue

                tokens_and_tags = [pair for pair in line.split(self._word_tag_delimiter)]
                tokens = tokens_and_tags[0]
                #print(len(tokens))
                tokens = tokens_and_tags[0][0:510]
                tags = tokens_and_tags[1]
                #print(tags)
                #tags = 1
                if len(tokens) == 0:
                    continue
                if len(tags) > 1:
                    print(tags)
                yield self.text_to_instance(tokens, tags)

    @overrides
    def text_to_instance(self, text: str, target: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        sequence = TextField(tokenized_text, self._token_indexers)
        fields["tokens"] = sequence
        if target is not None:
            fields['label'] = LabelField(target)
        return Instance(fields)
