import argparse
from typing import Dict, List, Tuple

from scirex.metrics.paired_bootstrap import eval_with_hierarchical_paired_bootstrap
from scirex.evaluation_scripts.relations_only_evaluate import prepare_data, compute_relations_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file")
    parser.add_argument("--ner-files-a", nargs='+', type=str)
    parser.add_argument("--ner-files-b", nargs='+', type=str)
    parser.add_argument("--clusters-files-a", nargs='+', type=str)
    parser.add_argument("--clusters-files-b", nargs='+', type=str)
    parser.add_argument("--relations-files-a", help="Relation prediction files from system A", nargs='+', type=str)
    parser.add_argument("--threshes-a", help="List of thresholds to use", nargs='*', type=float)
    parser.add_argument("--relations-files-b", help="Relation predictions files from system B", nargs='+', type=str)
    parser.add_argument("--threshes-b", help="List of thresholds to use", nargs='*', type=float)

    args = parser.parse_args()

    thresholds_a = args.threshes_a
    if thresholds_a is None or len(thresholds_a) == 0:
        thresholds_a = [None] * len(args.relations_files_a)
    else:
        assert len(thresholds_a) == len(args.relations_files_a)
    
    thresholds_b = args.threshes_b
    if thresholds_b is None or len(thresholds_b) == 0:
        thresholds_b = [None] * len(args.relations_files_b)
    else:
        assert len(thresholds_b) == len(args.relations_files_b)
    
    assert len(args.ner_files_a) == len(args.ner_files_b)
    assert len(args.clusters_files_a) == len(args.clusters_files_b)
    assert len(args.ner_files_a) == len(args.clusters_files_a)
    assert len(args.clusters_files_a) == len(args.relations_files_a)
    processed_datas_a = []
    for ner_file_a, clusters_file_a, rel_file_a in zip(args.ner_files_a, args.clusters_files_a, args.relations_files_a):
        processed_data_a = prepare_data(args.gold_file, ner_file_a, clusters_file_a, rel_file_a)
        processed_datas_a.append(processed_data_a)

    processed_datas_b = []
    for ner_file_b, clusters_file_b, rel_file_b in zip(args.ner_files_b, args.clusters_files_b, args.relations_files_b):
        processed_data_b = prepare_data(args.gold_file, ner_file_b, clusters_file_b, rel_file_b)
        processed_datas_b.append(processed_data_b)

    gold_data, _, _, _, _ = processed_data_b

    for n in [2, 4]:
        print(f"N={n}")
        retrieval_f1_a_list = []
        retrieval_length = None
        for thresh_a, processed_data_a in zip(thresholds_a, processed_datas_a):
    
            gold_data_a, predicted_ner_a, predicted_salient_clusters_a, predicted_relations_a, predicted_cluster_to_gold_cluster_map_a = processed_data_a
            assert gold_data_a == gold_data
            retrieval_metrics_df_a, _, _, _, _, _, _ = compute_relations_metrics(
                                                    gold_data,
                                                    predicted_ner_a,
                                                    predicted_salient_clusters_a,
                                                    predicted_relations_a,
                                                    predicted_cluster_to_gold_cluster_map_a,
                                                    n=n,
                                                    thresh=thresh_a)
            
            if retrieval_length is None:
                retrieval_length = len(retrieval_metrics_df_a["f1"])
            else:
                assert retrieval_length == len(retrieval_metrics_df_a["f1"])
            retrieval_f1_a_list.append(retrieval_metrics_df_a["f1"])

        retrieval_f1_b_list = []
        for thresh_b, processed_data_b in zip(thresholds_b, processed_datas_b):
            gold_data_b, predicted_ner_b, predicted_salient_clusters_b, predicted_relations_b, predicted_cluster_to_gold_cluster_map_b = processed_data_b
            assert gold_data_b == gold_data
            retrieval_metrics_df_b, _, _, _, _, _, _ = compute_relations_metrics(
                                                    gold_data,
                                                    predicted_ner_b,
                                                    predicted_salient_clusters_b,
                                                    predicted_relations_b,
                                                    predicted_cluster_to_gold_cluster_map_b,
                                                    n=n,
                                                    thresh=thresh_b)
            assert retrieval_length == len(retrieval_metrics_df_b["f1"])
            retrieval_f1_b_list.append(retrieval_metrics_df_b["f1"])

        print("\n")
        print(f"Paired Bootstrap Comparison of System A and System B on relation retrieval metric:")
        # The bootstrap script expects a list of gold values, but here the "system" values are already 
        # comparisons with gold, so just pass in a list of Nones to satisfy the input.
        sys1_retrieval_list = retrieval_f1_a_list
        sys2_retrieval_list = retrieval_f1_b_list
        assert len(sys1_retrieval_list) == len(sys2_retrieval_list)

        gold = [None for _ in sys1_retrieval_list[0]]
        # Each bootstrap sample draws 50 items.
        eval_with_hierarchical_paired_bootstrap(gold, sys1_retrieval_list, sys2_retrieval_list,
                                num_samples=10000, sample_ratio=0.76,
                                eval_type='avg')

if __name__ == "__main__":
    main()