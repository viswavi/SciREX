export test_file=scirex_dataset/release_data/test.jsonl
export test_output_folder=test_outputs/

# These scripts require you've already generated files for NER and cluster predictions
if [ ! -f $test_output_folder/cluster_predictions.jsonl ]; then
    echo "Need to generate $test_output_folder/cluster_predictions.jsonl, from the general SciREX prediction script"
    exit 1
elif [ ! -f $test_output_folder/ner_predictions.jsonl ]; then
    echo "Need to generate $test_output_folder/ner_predictions.jsonl, from the general SciREX prediction script"
    exit 1
fi

echo "Predicting Salient Mentions"
python scirex/predictors/predict_salient_mentions.py \
$salient_only_archive \
$test_output_folder/ner_predictions.jsonl \
$test_output_folder/salient_mentions_predictions.jsonl \
$cuda_device

echo "Predicting Salient Clustering "
python scirex/predictors/predict_salient_clusters.py \
$test_output_folder/cluster_predictions.jsonl \
$test_output_folder/salient_mentions_predictions.jsonl \
$test_output_folder/salient_clusters_predictions.jsonl


echo "Evaluating on all Predicted steps "
python scirex/evaluation_scripts/salient_only_evaluate.py \
--gold-file $test_file \
--ner-file $test_output_folder/ner_predictions.jsonl \
--clusters-file $test_output_folder/salient_clusters_predictions.jsonl \
