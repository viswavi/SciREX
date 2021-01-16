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

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_relations_only_with_graph_embeddings_early_fusion_and_citances/$1}

export bert_fine_tune=10,11,pooler

export use_citation_graph_embeddings=true
export citation_embedding_file=/projects/ogma1/vijayv/SciREX/graph_embeddings/embeddings.npy
export doc_to_idx_mapping_file=/projects/ogma1/vijayv/SciREX/graph_embeddings/scirex_docids.json
nw=1 lw=1 rw=1 em=false \
relation_cardinality=4 \
allennlp train -s $OUTPUT_BASE_PATH --include-package scirex $RECOVER $CONFIG_FILE
