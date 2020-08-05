// 中文语义角色标注模型基于以下论文:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader":{"type":"c_srl"},
  "train_data_path": "data/chinese_semantic_role_labeling/cpbtrain.txt",
  "validation_data_path": "data/chinese_semantic_role_labeling/cpbdev.txt",
  "model": {
    "type": "c_srl",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 300,
            "pretrained_file": "data/embed/word2vec/sgns.baidubaike.bigram-char",
            "trainable": true
        }
      }
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ]
    ],
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 400,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1,
      "use_highway": true
    },
    "binary_feature_dim": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 40
  },

  "trainer": {
    "num_epochs": 20,
    "grad_clipping": 1.0,
    "patience": 10,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 0,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
