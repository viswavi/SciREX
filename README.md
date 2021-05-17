# SciREX : A Challenge Dataset for Document-Level Information Extraction

Our data can be found here : https://github.com/allenai/SciREX/blob/master/scirex_dataset/release_data.tar.gz
You can also browse the dataset at - https://allenai.github.io/SciREX/

It contains 3 files - {train, dev, test}.jsonl

Each file contains one document per line in format  - 

```python
{
    "doc_id" : str = Document Id as used by Semantic Scholar,
    "words" : List[str] = List of words in the document,
    "sentences" : List[Span] = Spans indexing into words array that indicate sentences,
    "sections" : List[Span] = Spans indexing into words array that indicate sections,
    "ner" : List[TypedMention] = Typed Spans indexing into words indicating mentions ,
    "coref" : Dict[EntityName, List[Span]] = Salient Entities in the document and mentions belonging to it,
    "n_ary_relations" : List[Dict[EntityType, EntityName]] = List of Relations where each Relation is a dictionary with 5 keys (Method, Metric, Task, Material, Score),
    "method_subrelations" : Dict[EntityName, List[Tuple[Span, SubEntityName]]] = Each Methods may be subdivided into simpler submethods and Submenthods in coref array. For example, DLDL+VGG-Face is broken into two methods DLDL , VGG-Face.
}

Span = Tuple[int, int] # Inclusive start and Exclusive end index
TypedMention = Tuple[int, int, EntityType]
EntityType = Union["Method", "Metric", "Task", "Material"]
EntityName = str
```

A note of concern: Further analysis of our dataset revealed that ~50% of relations contain atleast one entity with no mentions in the paper (they occur in tables which we have discarded from our dataset). This makes evaluation of end to end task difficult (no predicted cluster can match that gold cluster). Currently, we remove these relations during evaluation for the end to end task (https://github.com/allenai/SciREX/blob/master/scirex/evaluation_scripts/scirex_relation_evaluate.py#L110). Note that this artifically reduces the precision of our model.

<hr>

Installation
============

1. `conda create -n scirex python=3.7`
2. `pip install -r requirements.txt`
3. `python -m spacy download en`
4. Please set the `PYTHONPATH` env variable to root of this repository.

Training SciREX baseline Model
=================

1. Extract the dataset files in folder `tar -xvzf scirex_dataset/release_data.tar.gz --directory scirex_dataset`
2. Export path to scibert `export BERT_BASE_FOLDER=<path-to-scibert>` . This path should contain two files atleast - vocab.txt and weights.tar.gz. Download the file here https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar and untar it.
2. Run `CUDA_DEVICE=<cuda-device-num> bash scirex/commands/train_scirex_model.sh main` to train main scirex model
3. Run `CUDA_DEVICE=<cuda-device-num> bash scirex/commands/train_pairwise_coreference.sh main` to train secondary coreference model.

Generating Predictions
======================


```bash
scirex_archive=outputs/pwc_outputs/experiment_scirex_full/main \
scirex_coreference_archive=outputs/pwc_outputs/experiment_coreference/main \
cuda_device=<cuda-device-num> \
bash scirex/commands/predict_scirex_model.sh
```

Citation
========

```bibtex
@inproceedings{jain-etal-2020-scirex,
  title={SciREX: A Challenge Dataset for Document-Level Information Extraction},
  author={Sarthak Jain and Madeleine van Zuylen and Hannaneh Hajishirzi and Iz Beltagy},
  booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
  month={jul},
  year={2020},
  eprint={2005.00512},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
