import logging
from allennlp.common.util import import_submodules
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.common import Params
from allennlp.commands.train import train_model
from allennlp.commands.fine_tune import fine_tune_model
from allennlp.commands.evaluate import evaluate
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
import sys

logger = logging.getLogger(__name__)


"""
预测
"""
def predict(**params):
    param_is_exist(["include_package", "model_file", "input"], params)
    for package_name in params["include_package"]:
        import_submodules(package_name)
    cuda = params["cuda"] if "cuda" in params else -1
    overrides = params["overrides"] if "overrides" in params else ""
    check_for_gpu(cuda)
    archive = load_archive(params["model_file"], cuda_device=cuda, overrides=overrides)
    predictor = Predictor.from_archive(archive, params["predictor"])
    # try:
    #     archive = load_archive(params["model_file"], cuda_device=cuda, overrides=overrides)
    #     predictor = Predictor.from_archive(archive, params["predictor"])
    # except Exception:
    #     print("请先下载预训练模型或者自己训练模型放在pretrain_model目录下面")
    #     sys.exit()
    results = predictor.predict_json(params["input"])
    return results

"""
训练
"""
def train(**params):
    param_is_exist(["config_file", "serialization_dir", "include_package"], params)
    for package_name in params["include_package"]:
        import_submodules(package_name)
    overrides = params["overrides"] if "overrides" in params else ""
    recover = params["recover"] if "recover" in params else False
    force = params["force"] if "force" in params else False
    config_params = Params.from_file(params["config_file"], overrides)
    return train_model(config_params, params["serialization_dir"], recover=recover, force=force,file_friendly_logging=True)

"""
微调模型
"""
def fine_tune(**params):
    param_is_exist(["config_file", "serialization_dir", "include_package", "model_file"], params)
    for package_name in params["include_package"]:
        import_submodules(package_name)
    overrides = params["overrides"] if "overrides" in params else ""
    recover = params["recover"] if "recover" in params else ""
    force = params["force"] if "force" in params else ""
    config_params = Params.from_file(params["config_file"], overrides)
    archive = load_archive(params["model_file"])
    return fine_tune_model(archive.model, config_params, params["serialization_dir"], recover, force)

"""
评测模型
"""
def evaluating(**params):
    param_is_exist(["model_file", "input_file", "include_package"], params)
    for package_name in params["include_package"]:
        import_submodules(package_name)
    cuda_device = params["cuda_device"] if "cuda_device" in params else -1
    overrides = params["overrides"] if "overrides" in params else ""
    weights_file = params["weights_file"] if "weights_file" in params else ""
    archive = load_archive(params["model_file"], cuda_device, overrides, weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = params["input_file"]
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    iterator_params = config.pop("validation_iterator", None)
    if iterator_params is None:
        iterator_params = config.pop("iterator")
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)
    metrics = evaluate(model, instances, iterator, cuda_device, batch_weight_key="loss")
    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics

def cover_dict(a: dict, b: dict):
    return dict(a, **b)

def param_is_exist(required_param:list, param_list=dict):
    """
    需要的参数是否在字典里面
    """
    error_param = [r_param for r_param in required_param if r_param not in param_list]
    if len(error_param) > 0:
        error_param = ",".join(error_param)
        print("缺少如下参数"+error_param)
        sys.exit()

def format_sentence(sentence:str):
    return " ".join(sentence.strip().replace(" ", ""))

def format_ner_result(results:dict):
    res = []
    entity = ""
    words = results["words"]
    tags = results["tags"]
    for index, tag in enumerate(tags):  # for every word
        if tag[0] == 'B':
            entity += words[index]
        elif tag[0] == 'M':
            entity += words[index]
        elif tag[0] == 'E':
            entity += words[index]
            res.append([entity, tag[2:]])
            st = ""
            for s in entity:
                st += s + ' '
            entity = ""
        elif tag[0] == 'S':
            entity = words[index]
            res.append([entity, tag[2:]])
            entity = ""
        else:
            entity = ""
    return res

def format_postag_result(results:dict):
    words = results["words"]
    tags = results["tags"]
    res = [str(words[index])+"/"+str(tag) for index, tag in enumerate(tags)]
    return " ".join(res)

def format_classify_result(results:dict):
    return results["label"]

def bmes_to_words(results:dict,require_s=True):
    chars = results["words"]
    tags = results["tags"]
    result = []
    if len(chars) == 0:
        return result
    word = chars[0]

    for c, t in zip(chars[1:], tags[1:]):
        if t[0] == 'B' or t[0] == 'S':
            result.append(word)
            word = ''
        word += c
    if len(word) != 0:
        result.append(word)
    return " ".join(result)