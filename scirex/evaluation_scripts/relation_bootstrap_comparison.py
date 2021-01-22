import argparse
from typing import Dict, List, Tuple

from scirex.metrics.paired_bootstrap import eval_with_paired_bootstrap
from scirex.evaluation_scripts.relations_only_evaluate import prepare_data, compute_relations_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file")
    parser.add_argument("--ner-file-a")
    parser.add_argument("--ner-file-b")
    parser.add_argument("--clusters-file-a")
    parser.add_argument("--clusters-file-b")
    parser.add_argument("--relations-file-a", help="Relation predictions from system A")
    parser.add_argument("--thresh-a", default=None, type=float)
    parser.add_argument("--relations-file-b", help="Relation predictions from system A")
    parser.add_argument("--thresh-b", default=None, type=float)

    args = parser.parse_args()

    processed_data_a = prepare_data(args.gold_file, args.ner_file_a, args.clusters_file_a, args.relations_file_a)
    gold_data_a, predicted_ner_a, predicted_salient_clusters_a, predicted_relations_a, predicted_cluster_to_gold_cluster_map_a = processed_data_a

    processed_data_b = prepare_data(args.gold_file, args.ner_file_b, args.clusters_file_b, args.relations_file_b)
    gold_data_b, predicted_ner_b, predicted_salient_clusters_b, predicted_relations_b, predicted_cluster_to_gold_cluster_map_b = processed_data_b

    assert gold_data_a == gold_data_b
    gold_data = gold_data_a

    for n in [2, 4]:
        print("\n")
        print(f"n: {n}")
        retrieval_metrics_df_a, _, _, _, _, _, _ = compute_relations_metrics(
                                                gold_data,
                                                predicted_ner_a,
                                                predicted_salient_clusters_a,
                                                predicted_relations_a,
                                                predicted_cluster_to_gold_cluster_map_a,
                                                n=n,
                                                thresh=args.thresh_a)
        retrieval_metrics_df_b, _, _, _, _, _, _ = compute_relations_metrics(
                                                gold_data,
                                                predicted_ner_b,
                                                predicted_salient_clusters_b,
                                                predicted_relations_b,
                                                predicted_cluster_to_gold_cluster_map_b,
                                                n=n,
                                                thresh=args.thresh_b)

        print("\n")
        print(f"Paired Bootstrap Comparison of System A and System B on relation retrieval metric:")
        # The bootstrap script expects a list of gold values, but here the "system" values are already 
        # comparisons with gold, so just pass in a list of Nones to satisfy the input.
        sys1_retrieval = list(retrieval_metrics_df_a["f1"])
        sys2_retrieval = list(retrieval_metrics_df_b["f1"])
        assert len(sys1_retrieval) == len(sys2_retrieval)

        gold = [None for _ in sys1_retrieval]
        # Each bootstrap sample draws 50 items.
        eval_with_paired_bootstrap(gold, sys1_retrieval, sys2_retrieval,
                                num_samples=1000, sample_ratio=0.76,
                                eval_type='avg')

if __name__ == "__main__":
    main()