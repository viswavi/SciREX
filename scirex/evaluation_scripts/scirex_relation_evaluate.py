import argparse
from itertools import combinations
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
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

def construct_valid_thresholds():
    return np.arange(0, 1, 0.001)

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


def clustering_metrics(gold_data, predicted_clusters, span_map):
    all_metrics = []
    mappings = {}
    for doc in gold_data:
        predicted_doc = predicted_clusters[doc["doc_id"]]
        metrics, mapping = match_predicted_clusters_to_gold(
            predicted_doc["clusters"], doc["coref"], span_map[doc["doc_id"]], doc['words']
        )
        mappings[doc["doc_id"]] = mapping
        all_metrics.append(metrics)

    all_metrics = pd.DataFrame(all_metrics)
    print("Salient Clustering Metrics")
    print(all_metrics.describe().loc['mean'])

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


def main(args):
    gold_data = load_jsonl(args.gold_file)
    for d in gold_data:
        merge_method_subrelations(d)
        d["clusters"] = d["coref"]

    predicted_ner = convert_to_dict(load_jsonl(args.ner_file))
    predicted_salient_clusters = convert_to_dict(load_jsonl(args.clusters_file))
    for d, doc in predicted_salient_clusters.items() :
        if 'clusters' not in doc :
            merge_method_subrelations(doc)
            doc['clusters'] = {x:v for x, v in doc['coref'].items() if len(v) > 0}

    predicted_relations = convert_to_dict(load_jsonl(args.relations_file))

    predicted_span_to_gold_span_map: Dict[str, Dict[tuple, tuple]] = ner_metrics(gold_data, predicted_ner)
    get_types_of_clusters(predicted_ner, predicted_salient_clusters)
    get_types_of_clusters(convert_to_dict(gold_data), convert_to_dict(gold_data))
    predicted_cluster_to_gold_cluster_map = clustering_metrics(
        gold_data, predicted_salient_clusters, predicted_span_to_gold_span_map
    )

    for n in [2, 4] :
        all_metrics = []
        for types in combinations(used_entities, n):
            for doc in gold_data:
                if "doc_id" not in doc or doc["doc_id"] not in predicted_relations:
                    # No predicted relations for this document.
                    relations = []
                else:
                    predicted_data = predicted_relations[doc["doc_id"]]
                    mapping = predicted_cluster_to_gold_cluster_map[doc["doc_id"]]

                    relations = list(set([
                        tuple([mapping.get(v, v) for v in x[0]])
                        for x in predicted_data["predicted_relations"]
                        if x[2] == 1
                    ]))

                    relations = [dict(zip(used_entities, x)) for x in relations]
                    relations = set([tuple((t, x[t]) for t in types) for x in relations])

                gold_relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]
                gold_relations = set([x for x in gold_relations if has_all_mentions(doc, x)])

                matched = relations & gold_relations

                metrics = {
                    "p": len(matched) / (len(relations) + 1e-7),
                    "r": len(matched) / (len(gold_relations) + 1e-7),
                }
                metrics["f1"] = 2 * metrics["p"] * metrics["r"] / (metrics["p"] + metrics["r"] + 1e-7)

                if len(gold_relations) > 0:
                    all_metrics.append(metrics)

        all_metrics = pd.DataFrame(all_metrics)
        print(f"Relation Metrics n={n}")
        print(all_metrics.describe().loc['mean'][['p', 'r', 'f1']])


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



def compute_relations_metrics(gold_data, predicted_ner, predicted_salient_clusters, predicted_relations, predicted_cluster_to_gold_cluster_map, thresh=None, n=4, average='binary', compute_mean_average_precision=False):
    retrieval_metrics = []
    sum_average_precision = 0.0
    number_of_documents = 0

    y_labels = []
    y_preds = []

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

            relations = [dict(zip(used_entities, x)) for x in relations]
            relations = set([tuple((t, x[t]) for t in types) for x in relations])

            gold_relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]
            gold_relations = set([x for x in gold_relations if has_all_mentions(doc, x)])

            relations_seen = set()
            relations_with_scores = []
            for relation_tuple in predicted_data["predicted_relations"]:
                relation_remapped = tuple([mapping.get(v, v) for v in relation_tuple[0]])
                relation_remapped = dict(zip(used_entities, relation_remapped))
                relation_remapped = tuple((t, relation_remapped[t]) for t in types)

                relation_score = relation_tuple[1]
                relation_pred = relation_tuple[2]
                if relation_remapped in relations_seen:
                    continue
                relations_seen.add(relation_remapped)
                relations_with_scores.append((relation_remapped, relation_score, relation_pred))

            relations_with_scores = sorted(relations_with_scores, key=lambda x: x[1], reverse=True)
            relations_sorted = [x[0] for x in relations_with_scores]

            y_preds_doc  = []
            y_labels_doc = []
            for relation_tuple in relations_with_scores:
                pred = relation_tuple[1] >= thresh if thresh is not None else relation_tuple[2] == 1
                label = relation_tuple[0] in gold_relations
                y_preds_doc.append(pred)
                y_labels_doc.append(label)


            if compute_mean_average_precision and len(gold_relations) > 0:
                average_precision = 0.0
                prev_recall = 0.0
                for k in range(1, len(relations_sorted) + 1):
                    relations_up_to_k = set(relations_sorted[:k])
                    matched_up_to_k = relations_up_to_k & gold_relations
                    precision = len(matched_up_to_k) / len(relations_up_to_k)
                    recall = len(matched_up_to_k) / len(gold_relations)
                    assert recall >= prev_recall

                    average_precision += precision * (recall - prev_recall)
                    prev_recall = recall

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
                if compute_mean_average_precision:
                    sum_average_precision += average_precision
                    number_of_documents += 1

                # The above predictions and labels were only covering relations that we did predict. We also need to count
                # relations that were labeled true, but we did not predict (equivalently, we predicted with score 0.0).
                for relation_tuple in gold_relations.difference(relations_seen):
                    y_labels_doc.append(True)
                    y_preds_doc.append(False)

                y_labels.extend(y_labels_doc)
                y_preds.extend(y_preds_doc)


    metric_summary = pd.DataFrame(retrieval_metrics).describe().loc['mean'][['p', 'r', 'f1']]

    f1 = f1_score(y_labels, y_preds, average=average)
    classification_precision = precision_score(y_labels, y_preds, average=average)
    classification_recall = recall_score(y_labels, y_preds, average=average)
    classification_metrics = {
                                "f1": f1,
                                "p": classification_precision,
                                "r": classification_recall
                            }
    if compute_mean_average_precision:
        mean_average_precision = sum_average_precision / number_of_documents
    else:
        mean_average_precision = None
    return metric_summary, classification_metrics, mean_average_precision


def main(args):
    processed_data = prepare_data(args.gold_file, args.ner_file, args.clusters_file, args.relations_file)
    gold_data, predicted_ner, predicted_salient_clusters, predicted_relations, predicted_cluster_to_gold_cluster_map = processed_data


    print("At fixed threshold")
    for n in [2, 4]:
        retrieval_metrics, classification_metrics, mean_average_precision, = compute_relations_metrics(gold_data,
                                                predicted_ner,
                                                predicted_salient_clusters,
                                                predicted_relations,
                                                predicted_cluster_to_gold_cluster_map,
                                                n=n,
                                                thresh=None,
                                                average='macro',
                                                compute_mean_average_precision=True)
        print(f"Relation Metrics n={n}")
        print(retrieval_metrics)
        print(f"Retrieval MAP (mean average precision): {mean_average_precision}")
    print("\n")

    if args.choose_dev_thresholds:
        dev_processed_data = prepare_data(args.dev_gold_file, args.dev_ner_file, args.dev_clusters_file, args.dev_relations_file)
        dev_gold_data, dev_predicted_ner, dev_predicted_salient_clusters, dev_predicted_relations, dev_predicted_cluster_to_gold_cluster_map = dev_processed_data

        best_threshold = -1
        best_f1 = -1
        for candidate_thresh in tqdm(construct_valid_thresholds()):
            retrieval_metrics, classification_metrics, _, = compute_relations_metrics(
                                                    dev_gold_data,
                                                    dev_predicted_ner,
                                                    dev_predicted_salient_clusters,
                                                    dev_predicted_relations,
                                                    dev_predicted_cluster_to_gold_cluster_map,
                                                    thresh=candidate_thresh,
                                                    n=4,
                                                    average='binary',
                                                    compute_mean_average_precision=True)
            if retrieval_metrics is None and classification_metrics is None:
                continue
            f1 = classification_metrics['f1']
            prf1 = dict(zip(["f1", "precision", "recall"], [classification_metrics['f1'], classification_metrics['p'], classification_metrics['r']]))
            if f1 > best_f1:
                 best_f1 = f1
                 best_threshold = candidate_thresh

        print(f"Best threshold is {round(thresh, 4)}, with dev-set classification metric (n=4) value of {round(best_f1, 4)}")

        for n in [2, 4]:
            print(f"At threshold {str(thresh)}:")
            retrieval_metrics, classification_metrics, _ = compute_relations_metrics(gold_data,
                                                    predicted_ner,
                                                    predicted_salient_clusters,
                                                    predicted_relations,
                                                    predicted_cluster_to_gold_cluster_map,
                                                    n=n,
                                                    thresh=best_threshold,
                                                    average='macro',
                                                    compute_mean_average_precision=False)
            print(f"Relation Metrics n={n}")
            print(retrieval_metrics)
            print(f"Retrieval MAP (mean average precision): {mean_average_precision}")

            classification_precision = classification_metrics['p']
            classification_recall = classification_metrics['r']
            f1 = classification_metrics['f1']
            print(f"Classification Precision: {classification_precision}")
            print(f"Classification Recall: {classification_recall}")
            print(f"Classification F1: {f1}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
