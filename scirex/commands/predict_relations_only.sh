export test_file=scirex_dataset/release_data/test.jsonl
export dev_file=scirex_dataset/release_data/dev.jsonl
# export test_output_folder=test_outputs_relations/
# export dev_output_folder=test_outputs_relations_dev/

# echo "Predicting Relations End-to-End"
# python scirex/predictors/predict_n_ary_relations.py \
# $scirex_archive \
# $test_output_folder/ner_predictions.jsonl \
# $test_output_folder/salient_clusters_predictions.jsonl \
# $test_output_folder/relations_predictions.jsonl \
# $cuda_device

# echo "Predicting relations End-to-End with gold cluster filtering"
# python scirex/predictors/predict_n_ary_relations.py \
# $scirex_archive \
# $test_output_folder/ner_predictions.jsonl \
# $test_output_folder/salient_clusters_predictions_using_gold.jsonl \
# $test_output_folder/relations_predictions_gold_salient_clusters.jsonl \
# $cuda_device

echo "Predicting Test Relations on gold clusters"
python scirex/predictors/predict_n_ary_relations.py \
$relations_only_archive \
$test_file \
$test_file \
$test_output_folder/relations_predictions_gold_clusters.jsonl \
$cuda_device

'''
echo "Predicting Dev Relations on gold clusters"
python scirex/predictors/predict_n_ary_relations.py \
$relations_only_archive \
$dev_file \
$dev_file \
$test_output_folder/relations_predictions_gold_clusters.jsonl \
$cuda_device
'''

echo "Evaluating on all Predicted steps "
python scirex/evaluation_scripts/relations_only_evaluate.py \
--gold-file $test_file \
--ner-file $test_file \
--clusters-file $test_file \
--relations-file $test_output_folder/relations_predictions_gold_clusters.jsonl

'''
echo "Evaluating on all Predicted steps "
python scirex/evaluation_scripts/relations_only_evaluate.py \
--gold-file $test_file \
--ner-file $test_file \
--clusters-file $test_file \
--dev-gold-file $dev_file \
--dev-ner-file $dev_file \
--dev-clusters-file $dev_file \
--dev-relations-file $dev_output_folder/relations_predictions_gold_clusters.jsonl \
--relations-file $test_output_folder/relations_predictions_gold_clusters.jsonl \
--choose-dev-thresholds \
'''
