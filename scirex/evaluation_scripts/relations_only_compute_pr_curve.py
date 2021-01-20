import argparse
from itertools import combinations
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from scirex.metrics.clustering_metrics import match_predicted_clusters_to_gold
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex_utilities.entity_utils import used_entities
from scirex_utilities.json_utilities import load_jsonl


parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--ner-file")
parser.add_argument("--clusters-file")
parser.add_argument("--relations-file")

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
    predicted_cluster_to_gold_cluster_map = match_predicted_clusters_with_gold(
        gold_data, predicted_salient_clusters, predicted_span_to_gold_span_map
    )

    [types] = list(combinations(used_entities, 4))

    y_test = []
    y_score = []
    y_pred = []

    tps = 0
    fps = 0
    fns = 0

    for doc in gold_data:
        predicted_data = predicted_relations[doc["doc_id"]]

        relations = list(set([
            (tuple(x[0]), x[1], x[2])
            for x in predicted_data["predicted_relations"]
        ]))

        relations = [(dict(zip(used_entities, x)), score, pred) for x, score, pred in relations]
        relations = set([(tuple(x[t] for t in types), score, pred) for x, score, pred in relations])

        gold_relations = doc['n_ary_relations']
        gold_relations = set([tuple(x[t] for t in types) for x in gold_relations])

        relation_tuples = set([x for x, score, pred in relations])
        matched = gold_relations & relation_tuples

        tps += len(matched)
        fps += len(relation_tuples) - len(matched)
        fns += len(gold_relations) - len(matched)

        seen_relations = set()
        for pred_relation in relations:
            relation = tuple(pred_relation[0])
            if relation in seen_relations:
                continue
            else:
                seen_relations.add(relation)
            prob = pred_relation[1]
            fixed_pred = pred_relation[2]
            y_test.append(int(relation in gold_relations))
            y_score.append(prob)
            y_pred.append(fixed_pred)

    average_precision = average_precision_score(y_test, y_score)
    print(f"Average Precision: {average_precision}")

    f1 = f1_score(y_test, y_pred)
    print(f"F1 score (at \"best\" threshold): {f1}")

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.plot(recall, precision, lw=2, color='navy')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.grid()
    plt.title('Precision-Recall (baseline) - AUC={0:0.2f}'.format(average_precision))
    plt.savefig("/tmp/pr_curve.png")
    print("Wrote PR curve to /tmp/pr_curve.png")


    manual_prec = float(tps) / (tps + fps)
    manual_rec = float(tps) / (tps + fns)
    f1 = 2 * (manual_prec * manual_rec) / (manual_prec + manual_rec)
    print(f"\n\nManual statistics:")
    print(f"Manual Precision: {manual_prec}")
    print(f"Manual Recall: {manual_rec}")
    print(f"f1: {f1}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)