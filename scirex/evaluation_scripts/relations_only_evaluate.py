import argparse
from itertools import combinations
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from scirex.metrics.clustering_metrics import match_predicted_clusters_to_gold
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex_utilities.entity_utils import used_entities
from scirex_utilities.json_utilities import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--ner-file")
parser.add_argument("--clusters-file")
parser.add_argument("--relations-file")
parser.add_argument("--dev-gold-file")
parser.add_argument("--dev-ner-file")
parser.add_argument("--dev-clusters-file")
parser.add_argument("--dev-relations-file")
parser.add_argument("--choose-dev-thresholds", action='store_true')
parser.add_argument("--choose-with-retrieval-metrics", action='store_true', help="If unset, then we will choose the best threshold with retrieval metrics instead of classification metrics.")
parser.add_argument("--choose-with-2-ary", action='store_true', help="If unset, then we will choose the best threshold with 2-ary relation metrics instead of 4-ary.")

def construct_valid_thresholds():
    return np.arange(0, 1, 0.0001)

def has_all_mentions(doc, relation):
    has_mentions = all(len(doc["clusters"][x[1]]) > 0 for x in relation)
    return has_mentions


def convert_to_dict(data):
    return {x["doc_id"]: x for x in data}


def ner_metrics(gold_data, predicted_data):
    mapping = {}
    for doc in gold_data:
        predicted_doc = predicted_data[doc["doc_id"]]
        predicted_spans = predicted_doc["ner"]
        gold_spans = doc["ner"]

        mapping[doc["doc_id"]] = map_predicted_spans_to_gold(predicted_spans, gold_spans)

    return mapping


def match_predicted_clusters_with_gold(gold_data, predicted_clusters, span_map):
    mappings = {}
    for doc in gold_data:
        predicted_doc = predicted_clusters[doc["doc_id"]]
        _, mapping = match_predicted_clusters_to_gold(
            predicted_doc["clusters"], doc["coref"], span_map[doc["doc_id"]], doc['words']
        )
        mappings[doc["doc_id"]] = mapping
    return mappings


def get_types_of_clusters(predicted_ner, predicted_clusters):
    for doc_id in predicted_clusters:
        clusters = predicted_clusters[doc_id]["clusters"]
        ner = {(x[0], x[1]): x[2] for x in predicted_ner[doc_id]["ner"]}

        predicted_clusters[doc_id]["types"] = {}
        for c, spans in clusters.items():
            types = set([ner[tuple(span)] for span in spans])
            if len(types) == 0:
                predicted_clusters[doc_id]["types"][c] = "Empty"
                continue
            predicted_clusters[doc_id]["types"][c] = list(types)[0]


def compute_relations_metrics(gold_data, predicted_ner, predicted_salient_clusters, predicted_relations, predicted_cluster_to_gold_cluster_map, thresh=None, n=4):
    tps = 0
    fps = 0
    fns = 0
    retrieval_metrics = []
    max_score = 0.0
    for types in combinations(used_entities, n):
        for doc in gold_data:
            predicted_data = predicted_relations[doc["doc_id"]]
            mapping = predicted_cluster_to_gold_cluster_map[doc["doc_id"]]

            if thresh is None:
                relations = list(set([
                    tuple([mapping.get(v, v) for v in x[0]])
                    for x in predicted_data["predicted_relations"]
                    if x[2] == 1
                ]))
            else:
                relations = list(set([
                    tuple([mapping.get(v, v) for v in x[0]])
                    for x in predicted_data["predicted_relations"]
                    if x[1] >= thresh
                ]))

            if len(predicted_data["predicted_relations"]):
                max_score = max(max_score, max([x[1] for x in predicted_data["predicted_relations"]]))

            relations = [dict(zip(used_entities, x)) for x in relations]
            relations = set([tuple((t, x[t]) for t in types) for x in relations])

            gold_relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]
            gold_relations = set([x for x in gold_relations if has_all_mentions(doc, x)])

            try:
                matched = relations & gold_relations
            except:
                breakpoint()

            metrics = {
                "p": len(matched) / (len(relations) + 1e-7),
                "r": len(matched) / (len(gold_relations) + 1e-7),
            }
            metrics["f1"] = 2 * metrics["p"] * metrics["r"] / (metrics["p"] + metrics["r"] + 1e-7)

            if len(gold_relations) > 0:
                retrieval_metrics.append(metrics)
                tps += len(matched)
                fps += len(relations) - len(matched)
                fns += len(gold_relations) - len(matched)


    metric_summary = pd.DataFrame(retrieval_metrics).describe().loc['mean'][['p', 'r', 'f1']]

    try:
        classification_precision = float(tps) / (tps + fps + 1e-7)
    except:
        breakpoint()
    classification_recall = float(tps) / (tps + fns + 1e-7)
    if classification_precision == 0.0 and classification_recall == 0.0:
        # Threshold is too high.
        return None, None

    f1 = 2 * (classification_precision * classification_recall) / (classification_precision + classification_recall)

    classification_metrics = {
                                "f1": f1,
                                "p": classification_precision,
                                "r": classification_recall
                            }
    return metric_summary, classification_metrics


def prepare_data(gold_file, ner_file, clusters_file, relations_file):
    gold_data = load_jsonl(gold_file)
    for d in gold_data:
        merge_method_subrelations(d)
        d["clusters"] = d["coref"]

    predicted_ner = convert_to_dict(load_jsonl(ner_file))
    predicted_salient_clusters = convert_to_dict(load_jsonl(clusters_file))
    for d, doc in predicted_salient_clusters.items() :
        if 'clusters' not in doc :
            merge_method_subrelations(doc)
            doc['clusters'] = {x:v for x, v in doc['coref'].items() if len(v) > 0}

    predicted_relations = convert_to_dict(load_jsonl(relations_file))

    predicted_span_to_gold_span_map: Dict[str, Dict[tuple, tuple]] = ner_metrics(gold_data, predicted_ner)
    get_types_of_clusters(predicted_ner, predicted_salient_clusters)
    get_types_of_clusters(convert_to_dict(gold_data), convert_to_dict(gold_data))
    predicted_cluster_to_gold_cluster_map = match_predicted_clusters_with_gold(
        gold_data, predicted_salient_clusters, predicted_span_to_gold_span_map
    )
    return gold_data, predicted_ner, predicted_salient_clusters, predicted_relations, predicted_cluster_to_gold_cluster_map


def main(args):
    processed_data = prepare_data(args.gold_file, args.ner_file, args.clusters_file, args.relations_file)
    gold_data, predicted_ner, predicted_salient_clusters, predicted_relations, predicted_cluster_to_gold_cluster_map = processed_data

    if args.choose_dev_thresholds:
        dev_processed_data = prepare_data(args.dev_gold_file, args.dev_ner_file, args.dev_clusters_file, args.dev_relations_file)
        dev_gold_data, dev_predicted_ner, dev_predicted_salient_clusters, dev_predicted_relations, dev_predicted_cluster_to_gold_cluster_map = dev_processed_data

        best_threshold = -1
        best_f1 = -1
        n = 2 if args.choose_with_2_ary else 4
        threshold_values = []
        for candidate_thresh in tqdm(construct_valid_thresholds()):
            retrieval_metrics, classification_metrics = compute_relations_metrics(
                                                    dev_gold_data,
                                                    dev_predicted_ner,
                                                    dev_predicted_salient_clusters,
                                                    dev_predicted_relations,
                                                    dev_predicted_cluster_to_gold_cluster_map,
                                                    thresh=candidate_thresh,
                                                    n=n)
            if retrieval_metrics is None and classification_metrics is None:
                continue
            f1 = retrieval_metrics['f1'] if args.choose_with_retrieval_metrics else classification_metrics['f1']
            prf1 = dict(zip(["f1", "precision", "recall"], [retrieval_metrics['f1'], retrieval_metrics['p'], retrieval_metrics['r']]))
            threshold_values.append((candidate_thresh, prf1))
            if f1 > best_f1:
                 best_f1 = f1
                 best_threshold = candidate_thresh

        thresh = best_threshold

        metric_objective_type = "retrieval metric" if args.choose_with_retrieval_metrics else "classification metric"
        print(f"Best threshold is {round(thresh, 4)}, with dev-set {metric_objective_type} (n={n}) value of {round(best_f1, 4)}")
    else:
        # Use pre-thresholded predictions
        thresh = None

    for n in [2, 4] :
        thresh_string = str(thresh) if thresh is not None else "<fixed>"
        print(f"At threshold {thresh_string}:")
        retrieval_metrics, classification_metrics = compute_relations_metrics(gold_data,
                                                predicted_ner,
                                                predicted_salient_clusters,
                                                predicted_relations,
                                                predicted_cluster_to_gold_cluster_map,
                                                n=n,
                                                thresh=thresh)
        print(f"Relation Metrics n={n}")
        print(retrieval_metrics)

        classification_precision = classification_metrics['p']
        classification_recall = classification_metrics['r']
        f1 = classification_metrics['f1']
        print(f"Classification Precision: {classification_precision}")
        print(f"Classification Recall: {classification_recall}")
        print(f"Classification F1: {f1}\n")

        print(f"Average precision across all thresholds:")

        classification_precisions = []
        retrieval_precisions = []
        for candidate_thresh in tqdm(construct_valid_thresholds()):
            retrieval_metrics, classification_metrics = compute_relations_metrics(gold_data,
                                        predicted_ner,
                                        predicted_salient_clusters,
                                        predicted_relations,
                                        predicted_cluster_to_gold_cluster_map,
                                        n=n,
                                        thresh=candidate_thresh)
            if retrieval_metrics is None and classification_metrics is None:
                continue
            retrieval_precisions.append(retrieval_metrics['p'])
            classification_precisions.append(classification_metrics['p'])
        print(f"Average Retrieval Precision: {np.mean(retrieval_precisions)}")
        print(f"Average Classification Precision: {np.mean(classification_precisions)}")

        print("\n\n")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
