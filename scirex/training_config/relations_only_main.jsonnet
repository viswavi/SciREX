// Import template file.

local template = import "ner_only_template.jsonnet";

// Set options.

local params = {
  use_lstm: true,
  bert_fine_tune: std.extVar("bert_fine_tune"),
  loss_weights: {          // Loss weights for the modules.
    ner: std.extVar('nw'),
  },
  relation_cardinality: std.parseInt(std.extVar('relation_cardinality')),
  exact_match: std.extVar('em'),
  use_citation_graph_embeddings: std.extVar("use_citation_graph_embeddings"),
  citation_embedding_file: std.extVar("citation_embedding_file"),
  doc_to_idx_mapping_file: std.extVar("doc_to_idx_mapping_file")
};

template(params)