// Import template file.

local template = import "salient_only_template.jsonnet";

// Set options.

local params = {
  use_lstm: true,
  bert_fine_tune: std.extVar("bert_fine_tune"),
  loss_weights: {          // Loss weights for the modules.
    saliency: std.extVar('lw'),
  },
  relation_cardinality: std.parseInt(std.extVar('relation_cardinality')),
  exact_match: std.extVar('em'),
  in_edges_tfidf_path: std.extVar("in_edges_tfidf_path"),
  out_edges_tfidf_path: std.extVar("out_edges_tfidf_path"),
  undirected_edges_tfidf_path: std.extVar("undirected_edges_tfidf_path"),
};

template(params)