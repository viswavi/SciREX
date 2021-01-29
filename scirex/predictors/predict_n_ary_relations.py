#! /usr/bin/env python

import json
import os
from sys import argv
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch

from allennlp.common.util import import_submodules
from allennlp.data import DataIterator, DatasetReader
from allennlp.data.dataset import Batch
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util
from scirex.predictors.utils import merge_method_subrelations

from scirex_utilities.json_utilities import load_jsonl, annotations_to_jsonl

import logging
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)

def combine_span_and_cluster_file(span_file, cluster_file, delete_relations = True) :
    spans = load_jsonl(span_file)
    clusters = {item['doc_id'] :  item for item in load_jsonl(cluster_file)}

    for doc in spans :
        if "doc_id" not in doc:
            continue
        if 'clusters' in clusters[doc['doc_id']] :
            doc['coref'] = clusters[doc['doc_id']]['clusters']
        else :
            merge_method_subrelations(clusters[doc['doc_id']])
            doc['coref'] = {x: v for x, v in clusters[doc['doc_id']]['coref'].items() if len(v) > 0}

        if delete_relations:
            if 'n_ary_relations' in doc:
                del doc['n_ary_relations']

            if 'method_subrelations' in doc :
                del doc['method_subrelations']

       
    annotations_to_jsonl(spans, 'tmp_relation_42424242.jsonl')


def predict(archive_folder, span_file, cluster_file, output_file, cuda_device):
    combine_span_and_cluster_file(span_file, cluster_file, delete_relations=False)

    test_file = 'tmp_relation_42424242.jsonl'
    model_metrics = json.load(open(archive_folder + '/metrics.json'))
    if "best_validation__n_ary_rel_global_threshold" in model_metrics:
        # Using SciREX relation model
        relation_threshold = model_metrics['best_validation__n_ary_rel_global_threshold']
    elif "best_validation_threshold" in model_metrics:
        relation_threshold = model_metrics['best_validation_threshold']
    else:
        raise ValueError("Model metrics does not specify which threshold to use for classification")
    print(relation_threshold)
    
    import_submodules("scirex")
    logging.info("Loading Model from %s", archive_folder)
    archive_file = os.path.join(archive_folder, "model.tar.gz")
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()

    model.prediction_mode = True
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    dataset_reader.prediction_mode = True
    instances = dataset_reader.read(test_file)

    for instance in instances :
        batch = Batch([instance])
        batch.index_instances(model.vocab)

    data_iterator = DataIterator.from_params(config["validation_iterator"])
    iterator = data_iterator(instances, num_epochs=1, shuffle=False)

    with open(output_file, "w") as f:
        documents = {}
        for batch in tqdm(iterator):
            with torch.no_grad() :
                batch = nn_util.move_to_device(batch, cuda_device)
                pred_batch = model(batch["tokens"], batch["label"], batch["metadata"])
                output_res = model.decode(pred_batch)

            metadata = batch["metadata"]
            scores = output_res["label_probs"]

            keys = ["Material", "Metric", "Task", "Method"]

            for i in range(len(scores)):
                doc_id = metadata[i]["doc_id"]
                if doc_id not in documents:
                    documents[doc_id] = {'predicted_relations' : [], 'doc_id' : doc_id}

                relation = metadata[i]["relation"]
                relation_tuple = tuple([relation[k] for k in keys])
                score = scores[i]
                label = 1 if score > relation_threshold else 0
                documents[doc_id]['predicted_relations'].append((relation_tuple, round(float(score), 4), label))

        for d in documents.values() :
            predicted_relations = {}
            for r, s, l in d['predicted_relations'] :
                r = tuple(r)
                if r not in predicted_relations or predicted_relations[r][0] < s:
                    predicted_relations[r] = (s, l)

            d['predicted_relations'] = [(r, s, l) for r, (s, l) in predicted_relations.items()]

        f.write("\n".join([json.dumps(x) for x in documents.values()]))


if __name__ == '__main__' :
    predict(argv[1], argv[2], argv[3], argv[4], int(argv[5]))