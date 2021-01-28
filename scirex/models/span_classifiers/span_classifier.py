import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from overrides import overrides

from nltk.corpus import stopwords
import numpy as np
import string

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from scirex.metrics.thresholding_f1_metric import BinaryThresholdF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SpanClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        mention_feedforward: FeedForward,
        label_namespace: str,
        n_features: int = 0,
        n_tfidf_features: int = 0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        in_edges_tfidf_map: Dict[str, Any] = None,
        out_edges_tfidf_map: Dict[str, Any] = None,
        undirected_edges_tfidf_map: Dict[str, Any] = None,
    ) -> None:
        super(SpanClassifier, self).__init__(vocab, regularizer)
        self._label_namespace = label_namespace

        self._mention_feedforward = TimeDistributed(mention_feedforward)
        if in_edges_tfidf_map is None or out_edges_tfidf_map is None or undirected_edges_tfidf_map is None:
            n_tfidf_features = 0
        self._ner_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim() + n_features + n_tfidf_features, 1))
        self._ner_metrics = BinaryThresholdF1()
        self._in_edges_tfidf_map = in_edges_tfidf_map
        self._out_edges_tfidf_map = out_edges_tfidf_map
        self._undirected_edges_tfidf_map = undirected_edges_tfidf_map
        self._stopwords = set(stopwords.words('english'))

        initializer(self)

    @staticmethod
    def process_word(word):
        # Make lowercase and strip punctuation
        word = word.lower().strip(string.punctuation)
        return word


    def compute_tf_idfs(self, span_words, doc_id, tf_idf_list, DEFAULT_MISSING_VALUE=0.0005):
        average_tf_idf = 0.0
        if doc_id not in tf_idf_list:
            return DEFAULT_MISSING_VALUE, True
        doc_tf_idf_list = tf_idf_list[doc_id]

        num_valid_words = 0
        for w in span_words:
            word_processed = self.process_word(w)
            if len(word_processed) == 0 or word_processed in self._stopwords:
                continue
            tf_idf = doc_tf_idf_list.get(word_processed, 0.0)
            average_tf_idf =+ tf_idf
            num_valid_words += 1
        if num_valid_words != 0:
            average_tf_idf /= num_valid_words
        # Return tf-idf value, and whether this value is missing or not (here, False)
        return average_tf_idf, False


    @overrides
    def forward(
        self,  # type: ignore
        spans: torch.IntTensor,  # (Batch Size, Number of Spans, 2)
        span_embeddings: torch.IntTensor,  # (Batch Size, Number of Spans, Span Embedding SIze)
        span_features: torch.FloatTensor = None,
        span_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:


        if self._in_edges_tfidf_map is not None and self._out_edges_tfidf_map is not None and self._undirected_edges_tfidf_map is not None:
            text_concatenated = []
            for m in metadata:
                text_concatenated.extend(m["paragraph"])

            tf_idf_features = []
            spans_list = spans.tolist()
            for i, batch in enumerate(spans_list):
                doc_id = metadata[i]["doc_id"]
                batch_features = []
                for [span_start, span_end] in batch:
                    span_words = text_concatenated[span_start:span_end+1]
                    in_graph_tfidf, in_edges_missing = self.compute_tf_idfs(span_words, doc_id, self._in_edges_tfidf_map)
                    in_graph_tfidf_missing = float(in_edges_missing)

                    out_graph_tfidf, out_edges_missing= self.compute_tf_idfs(span_words, doc_id, self._out_edges_tfidf_map)
                    out_graph_tfidf_missing = float(out_edges_missing)

                    undirected_graph_tfidf, any_edges_missing = self.compute_tf_idfs(span_words, doc_id, self._undirected_edges_tfidf_map)
                    undirected_graph_tfidf_missing = float(any_edges_missing)

                    # TODO(Vijay): try to include missing-tf-idf features
                    batch_features.append([#in_graph_tfidf,
                                           #in_graph_tfidf_missing,
                                           #out_graph_tfidf,
                                           #out_graph_tfidf_missing,
                                           undirected_graph_tfidf,
                                           #undirected_graph_tfidf_missing,
                                           ])

                tf_idf_features.append(batch_features)

            tf_idf_features = torch.tensor(tf_idf_features, device=spans.device)
        else:
            tf_idf_features = None

        # Shape: (Batch_size, Number of spans, H)
        span_feedforward = self._mention_feedforward(span_embeddings)
        if span_features is not None :
            span_feedforward = torch.cat([span_feedforward, span_features], dim=-1)

        if tf_idf_features is not None:
            span_feedforward = torch.cat([span_feedforward, tf_idf_features], dim=-1)

        ner_scores = self._ner_scorer(span_feedforward).squeeze(-1) #(B, NS)
        ner_probs = torch.sigmoid(ner_scores)

        output_dict = {
            "spans" : spans,
            "ner_probs": ner_probs,
            "loss" : 0.0
        }

        if span_labels is not None:
            assert ner_probs.shape == span_labels.shape, breakpoint()
            assert len(ner_probs.shape) == 2, breakpoint()
            self._ner_metrics(ner_probs, span_labels)
            loss = self._compute_loss_for_scores(ner_probs, span_labels, metadata)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["metadata"] = metadata

        return output_dict

    def _compute_loss_for_scores(self, ner_probs, ner_labels, metadata):
        ner_probs_flat = ner_probs.view(-1)
        ner_labels_flat = ner_labels.view(-1)

        loss = torch.nn.BCELoss(reduction="mean")(ner_probs_flat, ner_labels_flat.float())
        return loss

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        output_dict['decoded_spans'] = []
        if 'spans' in output_dict :
            for spans, spans_prob in zip(output_dict['spans'], output_dict['ner_probs']) :
                decoded = {(span[0].item(), span[1].item() + 1): label.item() for span, label in zip(spans, spans_prob)}
                output_dict['decoded_spans'].append(decoded)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._ner_metrics.get_metric(reset)
        metrics = {"span_" + k: v for k, v in metrics.items()}

        return metrics

