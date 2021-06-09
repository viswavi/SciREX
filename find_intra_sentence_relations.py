'''
Usage:
python find_intra_sentence_relations.py \
    --scirex-dir /Users/vijay/Documents/code/SciREX/scirex_dataset/release_data \
    --n 4 \
    --print-relations \
    --max-relations-to-print 10
'''
import argparse
from collections import defaultdict
import itertools
import jsonlines
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--scirex-dir", type=str, required=True,
                    help="path to scirex directory")
parser.add_argument("--n", type=int, default=4,
                    help="number of entities required in intra-sentence relations")
parser.add_argument("--print-relations", action="store_true", help="whether to print all in-sentence relations after computing")
parser.add_argument("--max-relations-to-print", type=int, default=10)

# Find all combinations of subrelations (e.g. 3-ary relations) from 4-ary relation labels.
def compute_subrelations(four_ary_relations, n):
    n_ary_relations = []
    n_ary_relation_strings = []
    for relation in four_ary_relations:
        relation_wo_score = {}
        for k, v in relation.items():
            if k != "score":
                relation_wo_score[k] = v

        n_ary_rels = itertools.combinations(relation_wo_score.items(), r=n)
        for rel in n_ary_rels:
            str_rel = str(rel)
            if str_rel not in n_ary_relation_strings:
                n_ary_relation_strings.append(str_rel)
                n_ary_relations.append(dict(rel))
    return n_ary_relations


def find_all_intra_sentence_relations(single_doc, n=4, max_tuple_width = 200, deduplicate_intrasentence_relations=True):
    four_ary_relations = single_doc['n_ary_relations']
    n_ary_relations = compute_subrelations(four_ary_relations, n=n)

    corefs = single_doc['coref']
    sentences = single_doc['sentences']
    words = single_doc['words']

    matching_sentences = []

    # Keep track of all mention-level relations labeled in each sentence, to avoid duplicates.
    all_sentence_relations = defaultdict(list)

    for relation in n_ary_relations:
        matching_corefs = {c:idxs for c, idxs in corefs.items() if c in relation.values()}
        coref_clusters = list(matching_corefs.values())
        entity_tuples = list(itertools.product(*coref_clusters))
        entity_tuples = [mention_tuple for mention_tuple in entity_tuples if max(max(mention_tuple)) - min(min(mention_tuple)) <= max_tuple_width ]
        marker=-1
        for i, sentence in enumerate(sentences):
            for mention_tuple in entity_tuples:
                matched = True
                matching_entities = []
                for entity in mention_tuple:
                    marker=0
                    if entity[0] < sentence[0] or entity[1] >= sentence[1]:
                        matched = False
                        break
                    else:
                        matching_entities.append(" ".join(words[entity[0]:entity[1]]))

                if matched:
                    if deduplicate_intrasentence_relations:
                        if matching_entities not in all_sentence_relations[i]:
                            all_sentence_relations[i].append(matching_entities)

                            if matching_entities == ['BLEU scores', 'COCO dataset', 'LeakGAN']:
                                breakpoint()

                            matching_sentences.append({
                                                        "doc_id": single_doc['doc_id'],
                                                        "sentence_boundaries": sentence,
                                                        "relation_entities": list(relation.keys()),
                                                        "relation_mentions": matching_entities,
                                                        "sentence_words": " ".join(words[sentence[0]:sentence[1]])
                            })
                    else:
                        matching_sentences.append({
                                                    "doc_id": single_doc['doc_id'],
                                                    "sentence_boundaries": sentence,
                                                    "relation_entities": list(relation.keys()),
                                                    "relation_mentions": matching_entities,
                                                    "sentence_words": " ".join(words[sentence[0]:sentence[1]])
                        })


    return matching_sentences

    

if __name__ == "__main__":
    args = parser.parse_args()

    scirex_path = args.scirex_dir
    train = os.path.join(scirex_path, "train.jsonl")
    dev = os.path.join(scirex_path, "dev.jsonl")
    test = os.path.join(scirex_path, "test.jsonl")

    train_data = list(jsonlines.open(train))
    dev_data = list(jsonlines.open(dev))
    test_data = list(jsonlines.open(test))

    all_data = train_data + dev_data + test_data

    all_intra_sentence_relations = []

    for doc in tqdm(all_data):
        intra_sentence_relations = find_all_intra_sentence_relations(doc, n=args.n)
        all_intra_sentence_relations.extend(intra_sentence_relations)

    print(f"Number of intra-sentence relations: {len(all_intra_sentence_relations)}")
    num_docs_with_relations = len(set(x["doc_id"] for x in all_intra_sentence_relations))
    print(f"Number of documents with intra-sentence relations: {num_docs_with_relations}\n")

    if args.print_relations:
        for relation in all_intra_sentence_relations[:args.max_relations_to_print]:
            relation_sentence = relation["sentence_words"]
            relation_mentions = relation["relation_mentions"]
            print(f"sentence: {relation_sentence}")
            print(f"relation_mentions: {relation_mentions}.")
            print("\n===========\n")
