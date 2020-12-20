if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/vocab.txt
export BERT_WEIGHTS=$BERT_BASE_FOLDER/weights.tar.gz

export CONFIG_FILE=scirex/training_config/salient_only_main.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

export IS_LOWERCASE=true

export DATA_BASE_PATH=scirex_dataset/release_data

export TRAIN_PATH=$DATA_BASE_PATH/train.jsonl
export DEV_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_salient_only_citation_tfidf/$1}

export bert_fine_tune=10,11,pooler

nw=1 lw=1 rw=1 em=false \
relation_cardinality=4 \
in_edges_tfidf_path=/projects/metis0_ssd/users/vijayv/SciREX/s2orc_caches/fulltexts/tf_idfs/in_edges.json \
out_edges_tfidf_path=/projects/metis0_ssd/users/vijayv/SciREX/s2orc_caches/fulltexts/tf_idfs/out_edges.json \
undirected_edges_tfidf_path=/projects/metis0_ssd/users/vijayv/SciREX/s2orc_caches/fulltexts/tf_idfs/undirected_edges.json \
allennlp train -s $OUTPUT_BASE_PATH --include-package scirex $RECOVER $CONFIG_FILE
