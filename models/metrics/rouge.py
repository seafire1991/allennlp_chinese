from typing import Tuple
from typing import Optional
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
import torch

@Metric.register("rouge")
class Rouge(Metric):
    """
    sentence level
    """
    def __init__(self, n: int = 1, beta: int = 1) -> None:
        self._n = n
        self._beta = beta
        self._total_p = 0.0
        self._total_r = 0.0
        self._total_f = 0.0
        self._count = 0


    def __call__(self,
                 predictions:torch.Tensor,
                 gold_labels:torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        for i, j in zip(predictions, gold_labels):
            # try:
            #     print(f'reference:{j}')
            #     print(f'predict:{i}')
            # except:
            #     pass
            p, r, f_score = rouge_n([i], [j], n=self._n, beta=self._beta)
            self._total_p += p
            self._total_r += r
            self._total_f += f_score
            self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        precision = self._total_p / self._count if self._count > 0 else 0
        recall = self._total_r / self._count if self._count > 0 else 0
        f_score = self._total_f / self._count if self._count > 0 else 0
        # print("selfcount:"+str(self._count))
        # print("selftotal_f:" + str(self._total_f))
        if reset:
            self.reset()
        #return precision, recall, f_score
        return f_score

    def reset(self):
        self._total_p = 0.0
        self._total_r = 0.0
        self._total_f = 0.0
        self._count = 0

def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_kskip_ngrams(k, n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:k + i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    assert (len(sentences) > 0)
    assert (n > 0)

    words = set()
    for sentence in sentences:
        words.update(_get_ngrams(n,sentence))

    return words


def rouge_n(evaluated_sentences, reference_sentences, n=2, beta=1):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        return 0,0,0
        raise (ValueError("Collections must contain at least 1 sentence."))

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    evaluated_count = len(evaluated_ngrams)
    reference_count = len(reference_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    precision = float(overlapping_count) / float(evaluated_count + 1e-13)
    recall = float(overlapping_count) / float(reference_count + 1e-13)
    f_measure = ((1 + (beta ** 2)) * (precision * recall)) / ((beta ** 2)*precision + recall + 1e-13)
    return precision, recall, f_measure

def rouge_su_n(evaluated_sentences, reference_sentences, n=2):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise (ValueError("Collections must contain at least 1 sentence."))

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    evaluated_count = len(evaluated_ngrams)
    reference_count = len(reference_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    precision = float(overlapping_count) / float(evaluated_count + 1e-13)
    recall = float(overlapping_count) / float(reference_count + 1e-13)
    f_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f_measure

def _len_lcs(x, y):
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y):
    table = _lcs(x, y)

    def _recon(i, j):
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    i, j = len(x), len(y)
    recon_tuple = tuple(map(lambda r: r[0], _recon(i, j)))
    return recon_tuple

def _f_lcs(llcs, m, n):
    """
    Computes the LCS-based F-measure score
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    :param llcs: Length of LCS
    :param m: number of words in reference summary
    :param n: number of words in candidate summary
    :returns float: LCS-based F-measure score
    """
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / r_lcs
    num = (1 + (beta ** 2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta ** 2) * p_lcs)
    return num / denom


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise (ValueError("Collections must contain at least 1 sentence."))
    reference_words = _split_into_words(reference_sentences)
    evaluated_words = _split_into_words(evaluated_sentences)
    m = len(reference_words)
    n = len(evaluated_words)
    lcs = _len_lcs(evaluated_words, reference_words)
    return _f_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
    """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common subsequence
    between reference sentence ri and candidate summary C. For example, if
    r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and
    c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is
    “w1 w2” and the longest common subsequence of r_i and c2 is “w1 w3 w5”. The
    union longest common subsequence of r_i, c1, and c2 is “w1 w2 w3 w5” and
    LCS_u(r_i, C) = 4/5.

    :param evaluated_sentences:
        The sentences that have been picked by the summarizer
    :param reference_sentence:
        One of the sentences in the reference summaries
    :returns float: LCS_u(r_i, C)
    :raises ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0:
        raise (ValueError("Collections must contain at least 1 sentence."))

    lcs_union = set()
    reference_words = _split_into_words([reference_sentence])
    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s])
        lcs = set(_recon_lcs(reference_words, evaluated_words))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    union_lcs_count = len(lcs_union)
    union_lcs_value = union_lcs_count / combined_lcs_length
    return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary

    :param evaluated_sentences:
        The sentences that have been picked by the summarizer
    :param reference_sentences:
        The sentences from the referene set
    :returns float: F_lcs
    :raises ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise (ValueError("Collections must contain at least 1 sentence."))

    # total number of words in reference sentences
    m = len(_split_into_words(reference_sentences))

    # total number of words in evaluated sentences
    n = len(_split_into_words(evaluated_sentences))

    union_lcs_sum_across_all_references = 0
    for ref_s in reference_sentences:
        union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences, ref_s)
    return _f_lcs(union_lcs_sum_across_all_references, m, n)