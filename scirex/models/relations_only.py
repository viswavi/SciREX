import copy
import json
import logging
import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import Average
from overrides import overrides

# Import submodules.
from scirex.models.relations.entity_relation import RelationExtractor as NAryRelationExtractor
from scirex.models.ner.ner_crf_tagger import NERTagger
from scirex.models.span_classifiers.span_classifier import SpanClassifier

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("ner_only")
class NEROnlyModel(Model):
    def __init__(
        self,
        use_citation_graph_embeddings: str,
        citation_embedding_file: str,
        doc_to_idx_mapping_file: str,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        modules: Params,
        loss_weights: Dict[str, int],
        lexical_dropout: float = 0.2,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        display_metrics: List[str] = None,
    ) -> None:
        super(NEROnlyModel, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)

        if use_citation_graph_embeddings:
            if citation_embedding_file == "" or doc_to_idx_mapping_file == "":
                raise ValueError("Must supply citation embedding files to use graph embedding features")
            self._document_embedding = initialize_graph_embeddings(citation_embedding_file, finetune_embedding=False)
            self._doc_to_idx_mapping = json.load(open(doc_to_idx_mapping_file))
        else:
            self._document_embedding = None
            self._doc_to_idx_mapping = None

        modules = Params(modules)

        self._ner = NERTagger.from_params(vocab=vocab,
                                          document_embedding=self._document_embedding,
                                          doc_to_idx_mapping=self._doc_to_idx_mapping,
                                          params=modules.pop("ner"))

        for k in loss_weights:
            loss_weights[k] = float(loss_weights[k])
        self._loss_weights = loss_weights
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        self._display_metrics = display_metrics
        self._multi_task_loss_metrics = {k: Average() for k in ["ner"]}

        self.training_mode = True
        self.prediction_mode = False

        initializer(self)

    @overrides
    def forward(
        self,
        text,
        ner_type_labels,
        spans=None,
        span_cluster_labels=None,
        span_saliency_labels=None,
        span_type_labels=None,
        span_features=None,
        relation_to_cluster_ids=None,
        metadata=None,
    ):
        output_dict = {}
        loss = 0.0

        output_embedding = self.embedding_forward(text)

        if self._loss_weights["ner"] > 0.0:
            output_dict["ner"] = self.ner_forward(output_embedding=output_embedding, ner_type_labels=ner_type_labels, metadata=metadata)
            loss += self._loss_weights["ner"] * output_dict["ner"]["loss"]

        output_dict["loss"] = loss
        for k in self._multi_task_loss_metrics:
            if k in output_dict:
                l = output_dict[k]["loss"]
                self._multi_task_loss_metrics[k](l)

        return output_dict

    def embedding_forward(self, text):
        # Shape: (batch_size, max_sentence_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))
        text_mask = util.get_text_field_mask(text)
        sentence_lengths = text_mask.sum(-1)

        # Shape: (total_sentence_length, encoding_dim)
        flat_text_embeddings = text_embeddings.view(-1, text_embeddings.size(-1))
        flat_text_mask = text_mask.view(-1).byte()

        filtered_text_embeddings = flat_text_embeddings[flat_text_mask.bool()]
        filtered_contextualized_embeddings = self._context_layer(
            filtered_text_embeddings.unsqueeze(0),
            torch.ones((1, filtered_text_embeddings.size(0)), device=filtered_text_embeddings.device).byte(),
        ).squeeze(0)

        flat_contextualized_embeddings = torch.zeros(
            (flat_text_embeddings.size(0), filtered_contextualized_embeddings.size(1)),
            device=filtered_text_embeddings.device,
        )
        flat_contextualized_embeddings.masked_scatter_(
            flat_text_mask.unsqueeze(-1).bool(), filtered_contextualized_embeddings
        )

        # Shape: (batch_size, max_sentence_length, embedding_size)
        contextualized_embeddings = flat_contextualized_embeddings.reshape(
            (text_embeddings.size(0), text_embeddings.size(1), flat_contextualized_embeddings.size(-1))
        )

        output_embedding = {
            "contextualised": contextualized_embeddings,
            "text": text_embeddings,
            "mask": text_mask,
            "lengths": sentence_lengths,
        }
        return output_embedding

    def ner_forward(self, output_embedding, ner_type_labels, metadata):
        output_ner = {"loss": 0.0}

        output_ner = self._ner(
            output_embedding["contextualised"], output_embedding["mask"], ner_type_labels, metadata
        )

        if self.prediction_mode:
            output_ner = self._ner.decode(output_ner)
            output_ner["spans"] = output_ner["spans"].to(output_embedding["text"].device).long()
            output_ner["span_labels"] = output_ner["span_labels"].to(output_embedding["text"].device).long()

        return output_ner

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        res = {}
        res["ner"] = self._ner.decode(output_dict["ner"])
        return res

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_loss = {"loss_" + k: v.get_metric(reset) for k, v in self._multi_task_loss_metrics.items()}
        metrics_loss = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics_loss.items()}

        # Make sure that there aren't any conflicting names.
        metric_names = (
            list(metrics_ner.keys())
            + list(metrics_loss.keys())
        )
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(
            list(metrics_ner.items())
            + list(metrics_loss.items())
        )

        all_metrics["validation_metric"] = (
            self._loss_weights["ner"] * nan_to_zero(metrics_ner.get("ner_f1-measure", 0))
        )

        self._display_metrics.append("validation_metric")
        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res


def nan_to_zero(n):
    if n != n:
        return 0

    return n

def initialize_graph_embeddings(graph_embedding_file, finetune_embedding=False):
    """
    Initialize graph embedding lookup table, loaded from file.
    """
    graph_embedding_array = np.load(graph_embedding_file)
    num_embeddings, embedding_dim = graph_embedding_array.shape
    embedding = torch.nn.Embedding(num_embeddings = num_embeddings, embedding_dim = embedding_dim)
    embedding.weight.data.copy_(torch.from_numpy(graph_embedding_array))
    embedding.weight.requires_grad = finetune_embedding
    return embedding 