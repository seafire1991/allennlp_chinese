class DEFAULT_PREDICTORS:
    SEGMENT = {"model_file": "pretrain_model/segment/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models","bita.nlp.predictors"], "predictor": "sentence-tagger"}

    NER = {"model_file": "pretrain_model/event_ner/model.tar.gz",
            "include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"], "predictor": "c-sentence-tagger"}

    POSTAG = {"model_file": "pretrain_model/postag/model.tar.gz",
               "include_package": ["bita.nlp.datasets", "bita.nlp.models"], "predictor": "sentence-tagger"}

    SENTIMENT = {"model_file": "pretrain_model/sentiment_classification/model.tar.gz",
                  "include_package": ["bita.nlp.datasets", "bita.nlp.models"], "predictor": "c-simple-seq2seq"}

    CLASSIFY = {"model_file": "pretrain_model/news_classification/model.tar.gz",
                 "include_package": ["bita.nlp.datasets", "bita.nlp.models"], "predictor": "c-simple-seq2seq"}

    SIMILARITY = {"model_file": "pretrain_model/similarity/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models","bita.nlp.predictors"], "predictor": "similarity"}

    SRL = {"model_file": "pretrain_model/srl/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"], "predictor": "c-srl"}

    OPENIE = {"model_file": "pretrain_model/semantic_role_labeler/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"],"predictor": "c-open-information-extraction"}

    SDP = {"model_file": "pretrain_model/sdp/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"], "predictor": "c-sdp"}

    UNI_DP= {"model_file": "pretrain_model/uni_dp/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"], "predictor": "c-dp"}

    QA = {"model_file": "pretrain_model/qa/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"], "predictor": "dureader-for-qa"}

    BERT_QA = {"model_file": "pretrain_model/qa/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"],"predictor": "dureader-for-qa"}

    SUMMARIZE = {"model_file": "pretrain_model/summarize/model.tar.gz",
          "include_package": ["bita.nlp.datasets", "bita.nlp.models", "bita.nlp.predictors"], "predictor": "c-dp"}

class DEFAULT_TRAINS:
    SEGMENT = {"config_file": "configs/segment.jsonnet", "serialization_dir": "log/segment",
                "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SEGMENT_BERT = {"config_file": "configs/segment_bert.jsonnet", "serialization_dir": "log/segment_bert",
               "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    NER = {"config_file": "configs/ner.jsonnet", "serialization_dir": "log/ner",
            "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    NER_BERT = {"config_file": "configs/ner_bert.json", "serialization_dir": "log/bdevent_bert1",
           "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    POSTAG = {"config_file": "configs/postag.jsonnet", "serialization_dir": "log/postag",
               "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    POSTAG_BERT = {"config_file": "configs/postag_bert.jsonnet", "serialization_dir": "log/postag_bert",
              "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    CLASSIFY = {"config_file": "configs/senti.json", "serialization_dir": "log/senti",
                 "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SENTIMENT = {"config_file": "configs/sentiment_classification.json", "serialization_dir": "log/sentiment_classification",
                  "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SIMILARITY = {"config_file": "configs/similarity.jsonnet", "serialization_dir": "log/similarity",
                   "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SRL = {"config_file": "configs/semantic_role_labeler.jsonnet", "serialization_dir": "log/srl",
            "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SDP = {"config_file": "configs/semantic_dependencies_parser.json", "serialization_dir": "log/sdp",
            "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    UNI_DP = {"config_file": "configs/universal_dependencies_parser.json", "serialization_dir": "log/uni_dp",
                "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    LM = {"config_file": "configs/lm.jsonnet", "serialization_dir": "log/lm", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    QA = {"config_file": "configs/qa.jsonnet", "serialization_dir": "log/qa", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    BERT_QA = {"config_file": "configs/bert_qa.json", "serialization_dir": "log/bert_qa",
          "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SUMMARIZE = {"config_file": "configs/summarize.jsonnet", "serialization_dir": "log/summarize",
          "include_package": ["bita.nlp", "bita.nlp.datasets"]}

class DEFAULT_FINE_TUNE:
    SEGMENT = {"config_file": "configs/segment.jsonnet", "serialization_dir": "log/segment_finetune","model_file":"pretrain_model/segment/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SEGMENT_BERT = {"config_file": "configs/segment_bert.jsonnet", "serialization_dir": "log/segment_finetune_bert",
 "model_file": "pretrain_model/segment/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    NER = {"config_file": "configs/ner.jsonnet", "serialization_dir": "log/ner_finetune", "model_file": "pretrain_model/ner/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    NER_BERT = {"config_file": "configs/ner_bert.json", "serialization_dir": "log/ner_finetune_bert",
 "model_file": "pretrain_model/ner_bert/model5.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    POSTAG = {"config_file": "configs/postag.jsonnet", "serialization_dir": "log/postag_finetune", "model_file":  "pretrain_model/postag/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    POSTAG_BERT = {"config_file": "configs/postag-bert.jsonnet", "serialization_dir": "log/postag_finetune",
    "model_file": "pretrain_model/postag/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    CLASSIFY = {"config_file": "configs/news_classification.json", "serialization_dir": "log/news_classification_finetune","model_file": "pretrain_model/news_classification/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SENTIMENT = {"config_file": "configs/sentiment_classification.json","model_file": "pretrain_model/sentiment_classification/model.tar.gz","serialization_dir": "log/sentiment_classification_finetune","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SIMILARITY = {"config_file": "configs/similarity.jsonnet", "serialization_dir": "log/similarity_finetune","model_file": "pretrain_model/similarity/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SRL = {"config_file": "configs/semantic_role_labeler.jsonnet", "serialization_dir": "log/srl_finetune","model_file": "pretrain_model/semantic_role_labeler/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SDP = {"config_file": "configs/semantic_dependencies_parser.json", "serialization_dir": "log/sdp_finetune","model_file": "pretrain_model/semantic-dependency-parser/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    UNI_DP = {"config_file": "configs/universal_dependencies_parser.json", "serialization_dir": "log/uni_dp_finetune","model_file": "pretrain_model/uni_dp/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    LM = {"config_file": "configs/lm.jsonnet", "serialization_dir": "log/lm_finetune",
               "model_file": "pretrain_model/lm/model.tar.gz","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    QA = {"config_file": "configs/qa.jsonnet", "serialization_dir": "log/qa_finetune",
        "model_file": "pretrain_model/qa/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    BERT_QA = {"config_file": "configs/bert_qa.json", "serialization_dir": "log/bert_qa_finetune",
          "model_file": "pretrain_model/bert_qa/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}

    SUMMARIZE = {"config_file": "configs/summarize.jsonnet", "serialization_dir": "log/summarize_finetune",
          "model_file": "pretrain_model/summarize/model.tar.gz", "include_package": ["bita.nlp.datasets", "bita.nlp.models"]}