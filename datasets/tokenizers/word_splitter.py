import jieba
from typing import List

from overrides import overrides

from allennlp.common import Params
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter

# 字段名	含义
# s	输入字符串，在xml选项x为n的时候，代表输入句子；为y时代表输入xml
# x	用以指明是否使用xml
# t	用以指明分析目标，t可以为分词（ws）,词性标注（pos），命名实体识别（ner），依存句法分析（dp），语义角色标注（srl）或者全部任务（all）

@WordSplitter.register('jieba')
class JieBaWordSplitter(WordSplitter):
    @overrides
    def split_words(self, doc: str) -> List[Token]:
        return [Token(t) for t in jieba.cut(doc, cut_all=False)]

    @classmethod
    def from_params(cls, params: Params) -> 'JieBaWordSplitter':
        params.assert_empty(cls.__name__)
        return cls()

