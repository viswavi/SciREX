import argparse
from itertools import combinations
from typing import Dict

import pandas as pd

from scirex.metrics.clustering_metrics import match_predicted_clusters_to_gold
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex_utilities.json_utilities import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--ner-file")


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


def main(args):
    gold_data = load_jsonl(args.gold_file)
    for d in gold_data:
        merge_method_subrelations(d)
        d["clusters"] = d["coref"]

    predicted_ner = convert_to_dict(load_jsonl(args.ner_file))
    predicted_span_to_gold_span_map: Dict[str, Dict[tuple, tuple]] = ner_metrics(gold_data, predicted_ner)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)