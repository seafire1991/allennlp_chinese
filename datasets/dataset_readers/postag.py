from typing import Dict, List
import logging

from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_LINE_DELEMITER = "\n"

@DatasetReader.register("postag_rm")
class PostagDatasetReader(DatasetReader):
    """
    1998 人民日报词性数据读取
    19980101-01-001-001/m 迈向/v 充满/v 希望/n 的/u 新/a 世纪/n ——/w 一九九八年/t 新年/t 讲话/n （/w 附/v 图片/n １/m 张/q ）/w

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

        def replace_symbol(symbols: list, news_str):
            start = news_str.find(symbols[0])
            end = news_str.find(symbols[1])
            word = "".join([x.split("/")[0] for x in news_str[start + 1:end].split(" ")])
            tag = news_str[end + 1:end + 3]
            newsparagraph = news_str.replace(news_str[start:end + 3], word + "/" + tag)
            while newsparagraph.find("[") >= 0 and newsparagraph.find("]") > 0:
                newsparagraph = replace_symbol(["[", "]"], newsparagraph)
            return newsparagraph

        with open(file_path, 'r', encoding='utf-8') as f:
            all_tag = []
            for line in f:
                newsparagraph = line[line.find("/m") + 2:][1:].strip() if line.find("/m") >= 0 else line.strip()
                if newsparagraph.find("[") >= 0 and newsparagraph.find("]") > 0 and line.find("/m") >= 0:
                    newsparagraph = replace_symbol(["[", "]"], newsparagraph)
                newsparagraph_list = newsparagraph.split(" ")
                if len(newsparagraph_list) <= 1:
                    continue
                word = [x.split("/")[0] for x in newsparagraph_list if len(x) > 1 and len(x.split("/")[0]) > 0]
                tag = [x.split("/")[1] for x in newsparagraph_list if len(x) > 1 and len(x.split("/")[1]) > 0]
                all_tag += tag
                tokens = [Token(token) for token in word]
                yield self.text_to_instance(tokens, tag)

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
