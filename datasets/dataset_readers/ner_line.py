from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from ..common.utils import format_sentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_WORD_TAG_DELIMITER = " "

@DatasetReader.register("ner_line")
class SequenceTaggingDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD TAG \n
    WORD TAG \n
    WORD TAG \n

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
                 token_delimiter: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._word_tag_delimiter = word_tag_delimiter
        self._token_delimiter = token_delimiter

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r", encoding="utf-8") as data_file:

            logger.info("Reading instances from lines in file at: %s", file_path)
            tokens_and_tags = []
            for line in data_file:
                if line != "\n":
                    line = line.replace("\n", "")
                    item = line.split(self._token_delimiter)
                    # print(item)
                    item_data = (item[0], item[1])
                    tokens_and_tags.append(item_data)
                else:
                    # skip blank lines
                    if not line:
                        continue
                    tokens = [Token(token) for token, tag in tokens_and_tags]
                    tags = [tag for token, tag in tokens_and_tags]
                    tokens_and_tags = []
                    # print(len(tokens))
                    if len(tokens) >= 510:
                        tokens = tokens[0:510]
                        tags = tags[0:510]
                    yield self.text_to_instance(tokens, tags)

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)
        return Instance(fields)
