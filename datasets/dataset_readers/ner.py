from typing import Dict, List
import logging

from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from ..common.utils import rm_replace_symbol

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_LINE_DELEMITER = "\n"

@DatasetReader.register("ner_rm")
class NerDatasetReader(DatasetReader):
    """
    1998 人民日报命名体识别数据读取
    19980101-01-001-005/m 同胞/n 们/k 、/w 朋友/n 们/k 、/w 女士/n 们/k 、/w 先生/n 们/k ：/w
19980101-01-001-006/m 在/p １９９８年/t 来临/v 之际/f ，/w 我/r 十分/m 高兴/a 地/u 通过/p [中央/n 人民/n 广播/vn 电台/n]nt 、/w [中国/ns 国际/n 广播/vn 电台/n]nt

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
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                newsparagraph = line[line.find("/m") + 2:][1:].strip() if line.find("/m") > 0 else line.strip()
                if newsparagraph.find("[") >= 0 and newsparagraph.find("]") > 0:
                    newsparagraph = rm_replace_symbol(["[", "]"], newsparagraph)
                newsparagraph_list = newsparagraph.split(" ")
                if len(newsparagraph_list) <= 1:
                    continue
                words = [x.split("/")[0] for x in newsparagraph_list if len(x) > 1]
                tags = [x.split("/")[1] for x in newsparagraph_list if len(x) > 1]
                ner_tags = ["nr", "ns", "nt"]
                new_words = []
                new_tags = []
                for word_index, word in enumerate(words):
                    for tag_index, tag in enumerate(tags):
                        if word_index == tag_index:
                            if tag in ner_tags:
                                if len(word) == 1:
                                    new_words.append(word)
                                    new_tags.append("B_" + tag)
                                elif len(word) > 1:
                                    new_words.append(word[0])
                                    new_tags.append("B_" + tag)
                                    for w in word[1:len(word) - 1]:
                                        new_words.append(w)
                                        new_tags.append("M_" + tag)
                                    new_words.append(word[len(word) - 1])
                                    new_tags.append("E_" + tag)
                                else:
                                    continue
                            else:
                                for x in word:
                                    new_words.append(x)
                                    new_tags.append("O")
                tokens = [Token(token) for token in new_words]
                yield self.text_to_instance(tokens, new_tags)

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
