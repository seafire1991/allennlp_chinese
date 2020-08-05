{
  "dataset_reader": {
    "type": "pg",
    "lazy": true,
    "source_tokenizer": {
      "type": "word",
      "word_splitter": {
         "type": "jieba"
      }
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id"
      }
    },
    "max_encoding_steps": 400
  },
  "train_data_path": "data/summarization/test.jsonl",
  "validation_data_path": "data/summarization/test.jsonl",
  "test_data_path": "data/summarization/test.jsonl",
  "model": {
    "type": "pgp",
    "source_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 300,
        "projection_dim": 128,
        "pretrained_file": "data/embed/word2vec/sgns.baidubaike.bigram-char",
        "trainable": true
      }
    },
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 128,
      "hidden_size": 128,
      "num_layers": 1,
      "dropout": 0.5
    },
    "max_decoding_steps": 30,
    "attention_function": {
      "type": "bilinear",
      "vector_dim": 256,
      "matrix_dim": 256
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["source_tokens", "num_tokens"],["target_tokens","num_tokens"]],
    "biggest_batch_first": true,
    "batch_size": 4,
    "max_instances_in_memory": 72
  },
  "trainer": {
    "num_epochs": 30,
    "patience": 20,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "grad_norm": 2.0,
    "optimizer": {
      "type": "adam"
    },
    "num_serialized_models_to_keep": 5
  }
}
