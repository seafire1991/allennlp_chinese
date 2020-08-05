# encoding=utf-8
import sys
import os
sys.path.append("../../../")
import bita
import torch
# import tensorflow as tf
# x = tf.constant([[[[1,1,1],[2,2,2]]]])
# print(x.shape)
# with tf.Session() as sess:
#
#     print(sess.run(tf.reduce_sum(x, [0, 1, 2], keepdims=True)))  #行列求和
#
# sys.exit()
#import torch
# decoder_input = torch.tensor([1]*36).view(36, -1)
# print(torch.tensor([1]*36))
# print(decoder_input.size())

import sys
# print(torch.cuda.is_available())

#电力行业实体识别
# data_path = "data/电力知识/"
# text = ""
#
# all_text = os.listdir(data_path)
# for t in all_text:
#     f = open(data_path + t)
#     data = f.read()
#     #text = text + data
#     print(bita.nlp.ner(data))

sentence = "若在个别位置上直流电阻异常，且三相在这一位置都出现这种情况，有可能是该分接位置经 常不运行，触头表面形成银硫化物或铜硫化物造成，操作5个循环后再测试。"

# sys.exit()
# ws = torch.ones(7, 7) * (-1)
# print(ws[(7, 0)])

#bita.nlp.bert_qa(pretrained=False)
#bita.nlp.summarize(pretrained=False)

#bita.nlp.qa(pretrained=False)
#print(bita.nlp.qa(["在360影视里看试试看吧", "大唐荣耀哪里能看全集"]))
#print(bita.nlp.qa(["壁虎是益虫,吃蚊子苍蝇昆虫,模样难看,实则益虫,不咬人", "壁虎是益虫吗"]))
#print(bita.nlp.evaluate(**{"model_file": "pretrain_model/qa/model.tar.gz", "input_file": "/media/kxf/软件/input_dir/datasets/dureader/train.json","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}))

#bita.nlp.segment(pretrained=False, isbert=True)
#bita.nlp.segment(pretrained=False)
#print(bita.nlp.segment(sentence))
#bita.nlp.segment(pretrained=False, finetune=True)
# print(bita.nlp.evaluate(**{"model_file": "pretrain_model/segment/model.tar.gz", "input_file": "data/segment/msr_train.txt","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}))

#bita.nlp.postag(pretrained=False,isbert=True)
#print(bita.nlp.postag(sentence))
#bita.nlp.classify(pretrained=False)
#bita.nlp.ner(pretrained=False, isbert=True)
print(bita.nlp.ner(sentence))

#bita.nlp.similarity(pretrained=False)
#print(bita.nlp.evaluate(**{"model_file": "pretrain_model/similarity/model.tar.gz", "input_file": "data/text_similarity/train.tsv","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}))

#bita.nlp.classify(pretrained=False)
#print(bita.nlp.classify(sentence))

#bita.nlp.sentiment(pretrained=False)
#print(bita.nlp.sentiment(sentence))


#print(bita.nlp.uni_dp(sentence))
#bita.nlp.uni_dp(pretrained=False)
#print(bita.nlp.evaluate(**{"model_file": "pretrain_model/uni_dp/model.tar.gz", "input_file": "data/ctb8.0/dep/dev.conll","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}))


#bita.nlp.sdp(pretrained=False)
#print(bita.nlp.sdp(sentence))
#print(bita.nlp.evaluate(**{"model_file": "pretrain_model/sdp/model.tar.gz", "input_file": "data/SemEval-2016/train/news.train.conll","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}))

#print(bita.nlp.open_ie(sentence))
#bita.nlp.srl(pretrained=False)
#print(bita.nlp.srl(sentence))
#print(bita.nlp.evaluate(**{"model_file": "pretrain_model/semantic_role_labeler/model.tar.gz", "input_file": "data/chinese_semantic_role_labeling/cpbtrain.txt","include_package": ["bita.nlp.datasets", "bita.nlp.models"]}))



