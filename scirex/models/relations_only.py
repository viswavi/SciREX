import copy
import logging
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
from scirex.models.span_classifiers.span_classifier import SpanClassifier

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("relations_only")
class RelationsOnlyModel(Model):
    def __init__(
        self,
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
        super(RelationsOnlyModel, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)

        modules = Params(modules)

        self._cluster_n_ary_relation = NAryRelationExtractor.from_params(
            vocab=vocab, params=modules.pop("n_ary_relation")
        )

        self._endpoint_span_extractor = EndpointSpanExtractor(
            context_layer.get_output_dim(), combination="x,y"
        )
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=context_layer.get_output_dim())

        for k in loss_weights:
            loss_weights[k] = float(loss_weights[k])
        self._loss_weights = loss_weights
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        self._display_metrics = display_metrics
        self._multi_task_loss_metrics = {k: Average() for k in ["n_ary_relation"]}

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
        output_span_embedding = self.span_embeddings_forward(
            output_embedding, spans, span_type_labels, span_features, metadata
        )
        if self._loss_weights["n_ary_relation"] > 0.0:
            output_dict["n_ary_relation"] = self.relation_forward(
                output_span_embedding,
                metadata,
                relation_to_cluster_ids,
                span_cluster_labels,
            )
            loss += self._loss_weights["n_ary_relation"] * output_dict["n_ary_relation"]["loss"]

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
        

    def span_embeddings_forward(self, output_embedding, spans, span_type_labels, span_features, metadata):
        output_span_embeddings = {"valid": False}

        if spans.nelement() != 0:
            span_mask, spans, span_embeddings = self.extract_span_embeddings(
                output_embedding["contextualised"], spans
            )

            if span_mask.sum() != 0:
                span_offset = self.offset_span_by_para_start(metadata, spans, span_mask)
                span_position = self.get_span_position(metadata, span_offset)
                span_type_labels_one_hot = self.get_span_one_hot_labels(
                    "span_type_labels", span_type_labels, spans
                )

                span_features = torch.cat(
                    [span_position, span_type_labels_one_hot, span_features.float()], dim=-1
                )
                featured_span_embeddings = torch.cat([span_embeddings, span_features], dim=-1)
                span_ix = span_mask.view(-1).nonzero().squeeze(1).long()

                output_span_embeddings = {
                    "span_mask": span_mask,
                    "span_ix": span_ix,
                    "spans": span_offset,
                    "span_embeddings": span_embeddings,
                    "featured_span_embeddings": featured_span_embeddings,
                    "span_type_labels": span_type_labels_one_hot,
                    "span_features": span_features.float(),
                    "valid": True,
                }

        return output_span_embeddings

    def relation_forward(
        self, output_span_embedding, metadata, relation_to_cluster_ids, span_cluster_labels):
        output_n_ary_relation = {"loss": 0.0}

        skipped = True
        if output_span_embedding["valid"]:
            spans, featured_span_embeddings, span_ix, span_mask = (
                output_span_embedding["spans"],
                output_span_embedding["featured_span_embeddings"],
                output_span_embedding["span_ix"],
                output_span_embedding["span_mask"],
            )

            if relation_to_cluster_ids is not None or self.prediction_mode:
                n_salient_clusters = len(metadata[0]["document_metadata"]["cluster_name_to_id"])
                type_to_cluster_ids = metadata[0]["document_metadata"]["type_to_cluster_ids"]
                relation_to_cluster_ids = metadata[0]["document_metadata"]["relation_to_cluster_ids"]
                span_cluster_labels = span_cluster_labels[:, :, :n_salient_clusters]

                output_n_ary_relation = self._cluster_n_ary_relation.compute_representations(
                    span_embeddings=featured_span_embeddings,
                    coref_labels=span_cluster_labels,
                    type_to_cluster_ids=type_to_cluster_ids,
                    relation_to_cluster_ids=relation_to_cluster_ids,
                    metadata=metadata,
                )
                skipped = False

        if skipped:
            output_n_ary_relation["loss"] = torch.tensor(0, device=output_span_embedding['spans'].device)

        return output_n_ary_relation

    def get_span_one_hot_labels(self, label_namespace, span_labels, spans):
        n_labels = self.vocab.get_vocab_size(label_namespace)
        span_labels_one_hot = torch.zeros((span_labels.size(0), span_labels.size(1), n_labels)).to(
            spans.device
        )
        span_labels_one_hot.scatter_(-1, span_labels.unsqueeze(-1), 1)
        return span_labels_one_hot

    @staticmethod
    def get_span_position(metadata, span_offset):
        doc_length = metadata[0]["document_metadata"]["doc_length"]
        span_position = span_offset.float().mean(-1, keepdim=True) / doc_length
        return span_position

    @staticmethod
    def offset_span_by_para_start(metadata, spans, span_mask):
        start_pos_in_doc = torch.LongTensor([x["start_pos_in_doc"] for x in metadata]).to(
            spans.device
        )  # (B,)
        para_offset = start_pos_in_doc.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        span_offset = spans + (para_offset * span_mask.unsqueeze(-1).long())
        return span_offset

    def extract_span_embeddings(self, contextualized_embeddings, spans):
        attended_span_embeddings = self._attentive_span_extractor(contextualized_embeddings, spans)
        span_mask = (spans[:, :, 0] >= 0).long()
        spans = F.relu(spans.float()).long()
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
        return span_mask, spans, span_embeddings

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        res = {}
        res["n_ary_relation"] = self._n_ary_relation.decode(output_dict["n_ary_relation"])

        return res

    def decode_relations(self, batch):
        output_embedding = self.embedding_forward(text=batch["text"])
        output_span_embedding = self.span_embeddings_forward(
            output_embedding=output_embedding,
            spans=batch["spans"],
            span_type_labels=batch["span_type_labels"],
            span_features=batch["span_features"],
            metadata=batch["metadata"],
        )

        output_n_ary_relation = self.relation_forward(
            output_span_embedding=output_span_embedding,
            metadata=batch["metadata"],
            relation_to_cluster_ids=batch.get("relation_to_cluster_ids", None),
            span_cluster_labels=batch["span_cluster_labels"],
        )

        res = {}
        res["n_ary_relation"] = self._cluster_n_ary_relation.decode(output_n_ary_relation)

        return res

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_n_ary = self._cluster_n_ary_relation.get_metrics(reset=reset)
        metrics_loss = {"loss_" + k: v.get_metric(reset) for k, v in self._multi_task_loss_metrics.items()}
        metrics_loss = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics_loss.items()}

        # Make sure that there aren't any conflicting names.
        metric_names = (
            list(metrics_n_ary.keys())
            + list(metrics_loss.keys())
        )
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(
            list(metrics_n_ary.items())
            + list(metrics_loss.items())
        )

        all_metrics["validation_metric"] = (
            self._loss_weights["n_ary_relation"]
            * nan_to_zero(metrics_n_ary.get("n_ary_rel_global_1.f1-score", 0))
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
