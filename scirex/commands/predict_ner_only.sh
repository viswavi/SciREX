export test_file=scirex_dataset/release_data/test.jsonl
export ner_output_folder=test_outputs_ner/

echo "Predicting NER"
python scirex/predictors/predict_ner.py \
$ner_only_archive \
$test_file \
$ner_output_folder/ner_predictions_with_graph_embeddings.jsonl \
$cuda_device

echo "Evaluating on NER only"
python scirex/evaluation_scripts/ner_evaluate.py \
--gold-file $test_file \
--ner-file $ner_output_folder/ner_predictions_with_graph_embeddings.jsonl