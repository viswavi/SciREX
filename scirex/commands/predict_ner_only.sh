export test_file=scirex_dataset/release_data/test.jsonl
export ner_output_folder=test_outputs_ner/

echo "Predicting NER"
python scirex/predictors/predict_ner.py \
$scirex_archive \
$test_file \
$ner_output_folder/ner_predictions.jsonl \
$cuda_device

echo "Evaluating on all Predicted steps "
python scirex/evaluation_scripts/salient_only_evaluate.py \
--gold-file $test_file \
--ner-file $ner_output_folder/ner_predictions.jsonl