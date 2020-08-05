// 中文命名体识别模型基于:
//   Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).
//Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).
{
  "dataset_reader": {
    "type": "ner_rm",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters"
      }
    }
  },
  "train_data_path": "data/ner/renm/renmin.txt",
  "validation_data_path": "data/ner/renm/testright1.txt",
  "test_data_path": "data/ner/renm/testright1.txt",
  "evaluate_on_test": true,
  "model": {
    "type": "c_crf_tagger",
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 300,
            "pretrained_file": "data/embed/word2vec/sgns.baidubaike.bigram-char",
            "trainable": true
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "embedding_dim": 25
            },
            "encoder": {
            "type": "gru",
            "input_size": 25,
            "hidden_size": 80,
            "num_layers": 2,
            "dropout": 0.25,
            "bidirectional": true
            }
        }
      }
    },
    "encoder": {
      "type": "gru",
      "input_size": 460,
      "hidden_size": 300,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": true
    },
    "regularizer": [
      [
        "transitions$",
        {
          "type": "l2",
          "alpha": 0.01
        }
      ]
    ]
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 75,
    "patience": 25,
    "cuda_device": 0
  }
}

