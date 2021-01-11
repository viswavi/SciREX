import argparse
from itertools import combinations
from typing import Dict

import pandas as pd

from scirex.metrics.clustering_metrics import match_predicted_clusters_to_gold
from scirex.metrics.f1 import compute_f1
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex_utilities.entity_utils import used_entities
from scirex_utilities.json_utilities import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--ner-file")
parser.add_argument("--clusters-file")
parser.add_argument("--salient-mentions-file")

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


def salent_mentions_metrics(gold_data, predicted_salient_mentions):
    all_metrics = []
    predicted = 0
    gold = 0
    matched = 0
    for doc in gold_data:
        gold_salient_spans = [span for coref_cluster in doc['coref'].values() for span in coref_cluster]

        predicted_doc = predicted_salient_mentions[doc["doc_id"]]
        saliency_spans = []
        for [start_span, end_span, saliency, _] in predicted_doc["saliency"]:
            if saliency:
                saliency_spans.append((start_span, end_span))

        matching_spans = set(gold_salient_spans).intersection(saliency_spans)
        matched += len(matching_spans)
        predicted += len(saliency_spans)
        gold += len(gold_salient_spans)

    precision, recall, f1 = compute_f1(predicted, gold, matched, m=1)
    all_metrics = pd.DataFrame({"f1": [f1], "p": [precision], "r": [recall]})
    print("Salient Mention Classification Metrics")
    print(all_metrics.describe().loc['mean'])


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

    predicted_salient_mentions = convert_to_dict(load_jsonl(args.salient_mentions_file))
    salent_mentions_metrics(gold_data, predicted_salient_mentions)

    predicted_ner = convert_to_dict(load_jsonl(args.ner_file))
    predicted_salient_clusters = convert_to_dict(load_jsonl(args.clusters_file))
    for d, doc in predicted_salient_clusters.items() :
        if 'clusters' not in doc :
            merge_method_subrelations(doc)
            doc['clusters'] = {x:v for x, v in doc['coref'].items() if len(v) > 0}

    predicted_span_to_gold_span_map: Dict[str, Dict[tuple, tuple]] = ner_metrics(gold_data, predicted_ner)
    get_types_of_clusters(predicted_ner, predicted_salient_clusters)
    get_types_of_clusters(convert_to_dict(gold_data), convert_to_dict(gold_data))
    predicted_cluster_to_gold_cluster_map = clustering_metrics(
        gold_data, predicted_salient_clusters, predicted_span_to_gold_span_map
    )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
