from .tool import *
from .constants import *

def postag(sentence="", pretrained=True, finetune=False, isbert=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.POSTAG
        new_params = cover_dict(param, params)
        new_params["input"] = {"sentence": sentence}
        res = predict(**new_params)
        return format_postag_result(res)
    else:
        if not finetune:
            params = DEFAULT_TRAINS.POSTAG_BERT if isbert else DEFAULT_TRAINS.POSTAG
            new_params = cover_dict(param, params)
            train(**new_params)
        else:
            params = DEFAULT_FINE_TUNE.POSTAG_BERT if isbert else DEFAULT_FINE_TUNE.POSTAG
            new_params = cover_dict(param, params)
            fine_tune(**new_params)

def segment(sentence="", ispos=False, pretrained=True, finetune=False, isbert=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.SEGMENT
        new_params = cover_dict(param, params)
        new_params["input"] = {"sentence": format_sentence(sentence)}
        if ispos:
            result = postag(bmes_to_words(predict(**new_params)))
        else:
            result = bmes_to_words(predict(**new_params))
        return result
    else:
        if not finetune:
            params = DEFAULT_TRAINS.SEGMENT_BERT if isbert else DEFAULT_TRAINS.SEGMENT
            new_params = cover_dict(param, params)
            train(**new_params)
        else:
            params = DEFAULT_FINE_TUNE.SEGMENT_BERT if isbert else DEFAULT_FINE_TUNE.SEGMENT
            new_params = cover_dict(param, params)
            fine_tune(**new_params)

def ner(sentence="", pretrained=True, finetune=False, isbert=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.NER
        new_params = cover_dict(param, params)
        new_params["input"] = {"sentence": format_sentence(sentence)}
        res = predict(**new_params)
        print(res)
        return format_ner_result(res)
    else:
        if not finetune:
            params = DEFAULT_TRAINS.NER_BERT if isbert else DEFAULT_TRAINS.NER
            new_params = cover_dict(param, params)
            train(**new_params)
        else:
            params = DEFAULT_FINE_TUNE.NER_BERT if isbert else DEFAULT_FINE_TUNE.NER
            new_params = cover_dict(param, params)
            fine_tune(**new_params)

def classify(sentence="", pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.CLASSIFY
        new_params = cover_dict(param, params)
        new_params["input"] = {"sentence": segment(sentence)}
        res = predict(**new_params)
        return format_classify_result(res)
    else:
        params = DEFAULT_TRAINS.CLASSIFY
        new_params = cover_dict(param, params)
        train(**new_params)

def sentiment(sentence="", pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.SENTIMENT
        new_params = cover_dict(param, params)
        new_params["input"] = {"source": format_sentence(sentence)}
        res = predict(**new_params)
        return format_classify_result(res)
    else:
        params = DEFAULT_TRAINS.SENTIMENT
        new_params = cover_dict(param, params)
        train(**new_params)

def similarity(sentence=[], pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.SIMILARITY
        new_params = cover_dict(param, params)
        new_sentence = {}
        new_sentence["premise"] = format_sentence(sentence[0])
        new_sentence["hypothesis"] = format_sentence(sentence[1])
        new_params["input"] = new_sentence
        res = format_classify_result(predict(**new_params))
        return res
    else:
        if not finetune:
            params = DEFAULT_TRAINS.SIMILARITY
            new_params = cover_dict(param, params)
            train(**new_params)
        else:
            params = DEFAULT_FINE_TUNE.SIMILARITY
            new_params = cover_dict(param, params)
            train(**new_params)

def srl(sentence="", pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.SRL
        new_params = cover_dict(param, params)
        new_params["input"] = {"sentence": segment(sentence,)}
        res = predict(**new_params)
        return res
    else:
        params = DEFAULT_TRAINS.SRL
        new_params = cover_dict(param, params)
        train(**new_params)

def sdp(sentence="", pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.SDP
        new_params = cover_dict(param, params)
        new_params["input"] = {"sentence": segment(sentence)}
        res = predict(**new_params)
        return res
    else:
        params = DEFAULT_TRAINS.SDP
        new_params = cover_dict(param, params)
        train(**new_params)

def uni_dp(sentence="", pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.UNI_DP
        new_params = cover_dict(param, params)
        new_params["input"] = {"sentence": segment(sentence)}
        res = predict(**new_params)
        return res
    else:
        params = DEFAULT_TRAINS.UNI_DP
        new_params = cover_dict(param, params)
        train(**new_params)

def open_ie(sentence="", **param):
    params = DEFAULT_PREDICTORS.OPENIE
    new_params = cover_dict(param, params)
    new_params["input"] = {"sentence": segment(sentence)}
    res = predict(**new_params)
    return res

def lm(sentence="", pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.LM
        new_params = cover_dict(param, params)
        new_params["input"] = {"sentence": segment(sentence)}
        res = predict(**new_params)
        return res
    else:
        params = DEFAULT_TRAINS.LM
        new_params = cover_dict(param, params)
        train(**new_params)

def qa(sentence="", pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.QA
        new_params = cover_dict(param, params)
        new_sentence = {}
        new_sentence["passage"] = segment(sentence[0])
        new_sentence["question"] = segment(sentence[1])
        new_params["input"] = new_sentence
        res = predict(**new_params)
        return res
    else:
        params = DEFAULT_TRAINS.QA
        new_params = cover_dict(param, params)
        train(**new_params)

def bert_qa(sentence="", pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.BERT_QA
        new_params = cover_dict(param, params)
        new_sentence = {}
        new_sentence["passage"] = segment(sentence[0])
        new_sentence["question"] = segment(sentence[1])
        new_params["input"] = new_sentence
        res = predict(**new_params)
        return res
    else:
        params = DEFAULT_TRAINS.BERT_QA
        new_params = cover_dict(param, params)
        train(**new_params)

def summarize(sentence="", pretrained=True, finetune=False, **param):
    if pretrained:
        params = DEFAULT_PREDICTORS.SUMMARIZE
        new_params = cover_dict(param, params)
        new_params["input"] = {"sentence": sentence}
        res = predict(**new_params)
        return res
    else:
        params = DEFAULT_TRAINS.SUMMARIZE
        new_params = cover_dict(param, params)
        train(**new_params)



