if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/vocab.txt
export BERT_WEIGHTS=$BERT_BASE_FOLDER/weights.tar.gz

export CONFIG_FILE=scirex/training_config/scirex_full.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

export IS_LOWERCASE=true

if [ -z ${DATA_BASE_PATH+x} ]; then export DATA_BASE_PATH=scirex_dataset/release_data; fi

export TRAIN_PATH=$DATA_BASE_PATH/train.jsonl
export DEV_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$DATA_BASE_PATH/test.jsonl

if [ -z ${use_citation_graph_embeddings+x} ]; then export use_citation_graph_embeddings=false; fi
if [ -z ${random_seed+x} ]; then export random_seed=13370; fi
if [ -z ${numpy_seed+x} ]; then export numpy_seed=1337; fi
if [ -z ${pytorch_seed+x} ]; then export pytorch_seed=133; fi

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_scirex_full_graph_${pytorch_seed}/$1}

export bert_fine_tune=10,11,pooler
export finetune_embedding=false
export graph_embedding_dim=128

if [ -z ${citation_embedding_file+x} ]; then
    export citation_embedding_file=""
    export doc_to_idx_mapping_file=""
    export use_citation_graph_embeddings=false
    export graph_embedding_dim=0
fi

nw=1 lw=1 rw=1 em=false \
relation_cardinality=4 \
allennlp train -s $OUTPUT_BASE_PATH --include-package scirex $RECOVER $CONFIG_FILE
