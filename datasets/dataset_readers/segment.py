from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from ..common.utils import segment_to_bmes

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_LINE_DELEMITER = "\n"

@DatasetReader.register("segment")
class SegmentDatasetReader(DatasetReader):
    """
    分词数据读取
    响  起  在  农村  大地  上  的  钟声  －  －  －  看  电视  纪录片  《  村民  的  选择  》

    Parameters
    ----------
    line_delimiter: ``str``, optional (default=``"\n"``)
        每一段数据的分割标记.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 line_delimiter: str = DEFAULT_LINE_DELEMITER,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.line_delimiter = line_delimiter

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r", encoding="utf-8") as input_data:
            for line in input_data:
                word_list = line.strip(self.line_delimiter).split()
                tokens, tags = segment_to_bmes(word_list)
                tokens = [Token(token) for token in tokens]
                if len(tokens) >= 510 or len(tokens) == 0:
                    continue
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
