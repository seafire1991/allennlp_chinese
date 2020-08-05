from typing import Dict, List, Tuple
import logging
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import AdjacencyField, MetadataField, SequenceLabelField
from allennlp.data.fields import Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__) # pylint: disable=invalid-name

@DatasetReader.register("c_sdp")
class SemevalDependenciesDatasetReader(DatasetReader):

    """
    Reads a file SemEval-2016 Task 9 (Chinese Semantic Dependency Parsing)
    Datasets.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.ROOT = 'Root'
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        with open(file_path, 'r', encoding="utf-8") as f:
            datas = "".join(f.readlines()).split("\n\n")
        for data in datas:
            lines = [x.split("\t") for x in data.split("\n")]
            label_seqs, pos_tags, arc_indices, tokens = [self.ROOT], [""], [(0, 0)], [self.ROOT]
            for i, line in enumerate(lines):
                if len(lines[i]) > 1:
                    cols = line
                    line_index = cols[0]
                    head_seq = int(cols[6])
                    label_seqs.append(cols[7])
                    pos_tags.append(cols[3])
                    tokens.append(cols[1])
                    arc_indices.append((int(line_index), int(head_seq)))
            if len(tokens) == 1:
                continue
            yield self.text_to_instance(tokens, pos_tags, arc_indices, label_seqs)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         pos_tags: List[str] = None,
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_tags: List[str] = None) -> Instance:
        import sys
        # pylint: disable=arguments-differ
        # print(tokens)
        # print(pos_tags)
        # print(arc_indices)
        # print(arc_tags)
        # sys.exit()

        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = token_field
        fields["metadata"] = MetadataField({"tokens": tokens})
        if pos_tags is not None:
            fields["pos_tags"] = SequenceLabelField(pos_tags, token_field, label_namespace="pos")
        if arc_indices is not None and arc_tags is not None:
            fields["arc_tags"] = AdjacencyField(arc_indices, token_field, arc_tags)
        return Instance(fields)
