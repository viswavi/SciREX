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

export DATA_BASE_PATH=scirex_dataset/data_with_citances_ctx_1_no_sort

export TRAIN_PATH=$DATA_BASE_PATH/train.jsonl
export DEV_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$DATA_BASE_PATH/test.jsonl

if [ -z "$random_seed" ]; then
  export random_seed=13370
fi

if [ -z "$numpy_seed" ]; then
  export numpy_seed=1337
fi

if [ -z "$pytorch_seed" ]; then
  export pytorch_seed=133
fi

default_output_dir=outputs/pwc_outputs/experiment_scirex_full_with_citances_${pytorch_seed}/$1
export OUTPUT_BASE_PATH=${OUTPUT_DIR:-$default_output_dir}
export bert_fine_tune=10,11,pooler

nw=1 lw=1 rw=1 em=false \
relation_cardinality=4 \
allennlp train -s $OUTPUT_BASE_PATH --include-package scirex $RECOVER $CONFIG_FILE
