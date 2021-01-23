if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/vocab.txt
export BERT_WEIGHTS=$BERT_BASE_FOLDER/weights.tar.gz

export CONFIG_FILE=scirex/training_config/relations_only_main.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

export IS_LOWERCASE=true

export DATA_BASE_PATH=scirex_dataset/data_with_citances_ctx_1_no_sort

export TRAIN_PATH=$DATA_BASE_PATH/train.jsonl
export DEV_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$DATA_BASE_PATH/test.jsonl

default_output_dir=outputs/pwc_outputs/experiment_relations_only_with_graph_early_and_citances_${pytorch_seed}/$1
export OUTPUT_BASE_PATH=${OUTPUT_DIR:-$default_output_dir}

export bert_fine_tune=10,11,pooler

export use_citation_graph_embeddings=true
export citation_embedding_file=/projects/metis0_ssd/users/vijayv/SciREX/graph_embeddings/embeddings.npy
export doc_to_idx_mapping_file=/projects/metis0_ssd/users/vijayv/SciREX/graph_embeddings/scirex_docids.json
nw=1 lw=1 rw=1 em=false \
relation_cardinality=4 \
allennlp train -s $OUTPUT_BASE_PATH --include-package scirex $RECOVER $CONFIG_FILE
