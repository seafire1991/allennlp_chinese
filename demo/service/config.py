class DEFAULT_PREDICTORS:
    SEGMENT = {"model_file": "pretrain_model/segment/model.tar.gz",
                "include_package": ["bita.nlp.datasets", "bita.nlp.models"], "predictor": "c-sentence-tagger"}

    NER = {"model_file": "pretrain_model/ner/model.tar.gz",
            "include_package": ["bita.nlp.datasets", "bita.nlp.models"], "predictor": "c-sentence-tagger"}

    POSTAG = {"model_file": "pretrain_model/postag/model.tar.gz",
               "include_package": ["bita.nlp.datasets", "bita.nlp.models"], "predictor": "c-sentence-tagger"}

    SENTIMENT = {"model_file": "pretrain_model/sentiment_classification/model.tar.gz",
                  "include_package": ["bita.nlp.datasets", "bita.nlp.models"], "predictor": "c-simple-seq2seq"}

    CLASSIFY = {"model_file": "pretrain_model/news_classification/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models"], "predictor": "c-simple-seq2seq"}

    SIMILARITY = {"model_file": "pretrain_model/similarity/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models","bita.nlp.predictors"], "predictor": "similarity"}

    SRL = {"model_file": "pretrain_model/srl/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"], "predictor": "c-srl"}

    OPENIE = {"model_file": "pretrain_model/semantic_role_labeler/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"],
"predictor": "c-open-information-extraction"}

    SDP = {"model_file": "pretrain_model/sdp/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"], "predictor": "c-sdp"}

    UNI_DP= {"model_file": "pretrain_model/uni_dp/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"], "predictor": "c-dp"}


class WEB_CONFIG:
    PORT = 8000

