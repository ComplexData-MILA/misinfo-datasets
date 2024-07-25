"""
Web interface for labeling rollouts from a select forest.

Input args:
- forest_name
- random_seed
- num_examples

Data source:
- reasoning_trees collection.
- manual_labels collection for labels that have already been created.

Output:
- labels are written to manual_labels collection.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import pymongo
import streamlit as st
from pymongo.database import Database

from prm_pipeline.utils import PathElement
from prm_pipeline.utils.labeling_utils import ManualLabel
from prm_pipeline.utils.tree_utils import NodeId, ReasoningTree, ForestName
from prm_pipeline.webserver.db_utils import MongoReasoningTreeDB
from prm_pipeline.metrics.utils.metrics_utils import (
    get_tree_accuracy,
    get_prediction_folded,
    get_f1_safe,
    get_confidence_interval,
    get_tree_token_usage,
    get_majority_vote_accuracy_f1,
    get_tree_correctness_stats,
)

from utils import add_index_to_nested_dicts, sort_table


@st.cache_resource
def get_reasoning_tree_db() -> MongoReasoningTreeDB:
    client = pymongo.MongoClient(os.environ["MONGO_DB_SERVER"])
    db = client[os.environ["MONGO_DB_NAME"]]
    reasoning_tree_db = MongoReasoningTreeDB(db, -1)
    return reasoning_tree_db


tree_db = get_reasoning_tree_db()
forest_names: List[str] = []
st.session_state["forest_names"] = st.text_area("forest_names")
num_folds = st.number_input("num_folds", min_value=1, value=5)
source_features = st.text_input("source_features", "annotations").split(",")
assert num_folds is not None
num_folds = int(num_folds)

token_stats: Dict[ForestName, Counter] = defaultdict(Counter)
token_stats_estimates: Dict[ForestName, Dict[str, int]] = {}
action_stats: Dict[ForestName, Dict[str, int]] = defaultdict()
cost_stats: Dict[ForestName, Dict[str, float]] = defaultdict(Counter)
export_format = st.selectbox("Export Format", ["tsv", "csv"])
show_accuracy = st.checkbox("Show Accuracy")
show_f1 = st.checkbox("Show f1", value=True)

cost_reference: Dict[str, float] = {
    "gpt-3.5-turbo/context": 0.5 / 1e6,
    "gpt-3.5-turbo/output": 1.5 / 1e6,
    "gpt-3.5-turbo/total": 0,
    "gpt-3.5-turbo-0125/context": 0.5 / 1e6,
    "gpt-3.5-turbo-0125/output": 1.5 / 1e6,
    "gpt-3.5-turbo-0125/total": 0,
    "gpt-4-0125-preview/context": 5 / 1e6,
    "gpt-4-0125-preview/output": 15 / 1e6,
    "gpt-4-0125-preview/total": 0,
    "gpt-4-turbo-2024-04-09/context": 10 / 1e6,
    "gpt-4-turbo-2024-04-09/output": 30 / 1e6,
    "gpt-4-turbo-2024-04-09/total": 0,
    "gpt-4o/context": 5 / 1e6,
    "gpt-4o/output": 15 / 1e6,
    "gpt-4o/total": 0,
    "command/input_tokens": 0.50 / 1e6,
    "command/output_tokens": 1.50 / 1e6,
    "command-light/input_tokens": 0.50 / 1e6,
    "command-light/output_tokens": 1.50 / 1e6,
    "command-r/input_tokens": 0.50 / 1e6,
    "command-r/output_tokens": 1.50 / 1e6,
    "bedrock/meta.llama3-70b-instruct-v1:0/context": 0.00265 / 1e3,
    "bedrock/meta.llama3-70b-instruct-v1:0/output": 0.0035 / 1e3,
    "bedrock/meta.llama3-70b-instruct-v1:0/total": 0,
}


trees_to_process: Dict[str, List[ReasoningTree]] = defaultdict(list)

if st.button("Submit"):
    forest_names = st.session_state["forest_names"].splitlines()
    forest_names = [forest_name.strip(' ",') for forest_name in forest_names]

    # each counter maps token type (context, output, etc.) to tally.
    # retrieve all trees in each forest.
    for forest_name in forest_names:
        trees = tree_db.get_forest(forest_name)
        for tree in trees:
            source_dataset = tree.attributes.get("source", {}).get("dataset")
            source_split = tree.attributes.get("source", {}).get("split")

            source_attributes_minified = {
                "label": tree.attributes.get("source", {}).get(
                    "label", tree.attributes.get("ground_truth")
                ),
            }
            for feature in source_features:
                source_attributes_minified[feature] = tree.attributes.get(
                    "source", {}
                ).get(feature, {})

            tree.attributes["source"] = source_attributes_minified

            source_name = forest_name
            if source_dataset is not None:
                source_name = "{}/{}: {}".format(
                    source_dataset, source_split, forest_name
                )
                trees_to_process[source_name].append(tree)

            trees_to_process["_: {}".format(forest_name)].append(tree)

    for forest_name, trees in trees_to_process.items():
        # Track token count per tree for mean and stdev estimates.
        token_counts: Dict[str, List[int]] = defaultdict(list)
        action_counts: Dict[str, int] = Counter()

        # Keys are True/False strings.
        prediction_correctness_counts = Counter()

        for tree in trees:
            tree_token_usage = get_tree_token_usage(tree)
            for name, usage in tree_token_usage.items():
                token_counts[name].append(usage)

            token_stats[forest_name] += tree_token_usage

            action_counts += get_tree_token_usage(
                tree,
                lambda node: {node.attributes.get("action", {}).get("name", "none"): 1},
                True,
            )
            prediction_correctness_counts += get_tree_correctness_stats(tree)

        action_counts_per_query = {}
        for action_name, count in action_counts.items():
            is_correct, action_type = action_name.split("/")
            action_counts_per_query[action_name] = (
                count / prediction_correctness_counts[is_correct]
            )
        action_stats[forest_name] = action_counts_per_query

        token_stats_estimate = {
            name: int(np.asarray(counts).mean().item())
            for name, counts in token_counts.items()
        }
        token_stats_estimates[forest_name] = token_stats_estimate

        cost_stats[forest_name]["_sum"] = 0.0
        for token_type, token_count in token_stats_estimate.items():
            cost = cost_reference.get(token_type)
            if cost == 0.0:
                continue

            if cost is not None:
                cost_stats[forest_name][token_type] = token_count * cost
                cost_stats[forest_name]["_sum"] += token_count * cost
            else:
                cost_stats[forest_name][token_type] = -1
                cost_stats[forest_name]["_sum"] = float("inf")

        print(action_counts, prediction_correctness_counts)


with st.expander("Token usage"):
    st.table(pd.DataFrame(token_stats).fillna(0).convert_dtypes(convert_integer=True).T)

with st.expander("Token usage (average per tree)"):
    st.table(
        pd.DataFrame(token_stats_estimates)
        .fillna(0)
        .convert_dtypes(convert_integer=True)
        .T
    )

with st.expander("Cost estimate (dollars, average per 1,000 trees)"):
    st.table(pd.DataFrame(cost_stats).T * 1000)

with st.expander("Action usage (per query)"):
    st.table(
        sort_table(action_stats, "source_name", False)
        .fillna(0)
        .convert_dtypes(convert_integer=True)
    )

# Map forest name to stats table row.
accuracy_stats: Dict[str, Dict[ForestName, str]] = defaultdict(dict)
label_stats: Dict[str, str] = {}
sklearn_reports: Dict[ForestName, str] = {}
predictions_tables: Dict[ForestName, List[Dict[str, str]]] = defaultdict(list)

for forest_name, trees in trees_to_process.items():
    sample_means = np.zeros((num_folds,))
    f1_scores: List[Optional[float]] = []
    num_examples = 0
    num_examples_parsed = 0
    model_counter = Counter()

    # labels and predictions across all folds
    # (num_folds, num_trees)
    all_labels: List[List[Any]] = []
    all_predictions: List[List[Any]] = []

    if len(trees) == 0:
        st.text(f"No trees found in forest {forest_name}.")
        continue

    leaf_counts = Counter()
    for fold_index in range(num_folds):
        # (num_trees,)
        predictions = []
        labels = []
        raw_labels = []
        sample_means_fold: List[float] = []

        for reasoning_tree in trees:
            leaf_counts[len(reasoning_tree.get_leaf_node_ids())] += 1
            tree_accuracy: Optional[float] = get_tree_accuracy(
                reasoning_tree, fold_index, num_folds
            )
            prediction, model_name = get_prediction_folded(reasoning_tree, fold_index)
            label: Optional[Any] = reasoning_tree.attributes.get("ground_truth")
            model_counter += get_tree_token_usage(
                reasoning_tree,
                lambda node: {
                    node.attributes.get(
                        "model_name", node.data.get("model_name", "None")
                    ): 1
                },
            )

            num_examples += 1
            predictions.append(prediction)
            labels.append(label)
            raw_labels.append(reasoning_tree.attributes["source"]["label"])

            if tree_accuracy is not None:
                sample_means_fold.append(tree_accuracy)
                num_examples_parsed += 1
            else:
                sample_means_fold.append(0.0)

        sample_means[fold_index] = np.asarray(sample_means_fold).mean().item()
        f1_score = get_f1_safe(labels, predictions)
        f1_scores.append(f1_score)

        all_labels.append(labels)
        all_predictions.append(predictions)

    if num_examples == 0:
        num_examples = -1

    # (num_trees, num_folds)
    all_labels_array = np.asarray(all_labels).T
    all_predictions_array = np.asarray(all_predictions).T
    assert all_labels_array.shape == all_predictions_array.shape
    assert all_labels_array.shape[0] == len(trees)
    assert all_labels_array.shape[1] == num_folds

    majority_vote_accuracy, majority_vote_f1, report_dict = (
        get_majority_vote_accuracy_f1(all_labels_array, all_predictions_array, 0)
    )

    model_stats = []
    for model_name, count in model_counter.most_common():
        model_stats.append("{}: {}".format(model_name, count))

    raw_labels_stats = []
    for label_name, count in Counter(raw_labels).most_common():
        raw_labels_stats.append("{}: {}".format(label_name, count))

    label_stats[forest_name] = "; ".join(raw_labels_stats)

    accuracy_stats["models"][forest_name] = "; ".join(model_stats)
    accuracy_stats["parsed"][forest_name] = "{}/{} ({:.1f}% parsed)".format(
        num_examples_parsed, num_examples, 100 * num_examples_parsed / num_examples
    )
    if show_accuracy:
        accuracy_stats["accuracy"][forest_name] = "{} (maj@{}: {:.1f}%)".format(
            get_confidence_interval(sample_means), num_folds, majority_vote_accuracy
        )

    if all([f1_score is not None for f1_score in f1_scores]) and show_f1:
        f1_scores_array = np.asarray(f1_scores)
        accuracy_stats["f1"][forest_name] = "{} (maj@{}: {:.1f}%)".format(
            get_confidence_interval(f1_scores_array), num_folds, majority_vote_f1
        )

    sklearn_reports[forest_name] = json.dumps(report_dict, indent=2)

    # (num_trees,)
    predictions_table: List[Dict[str, Any]] = []
    for tree, predictions in zip(trees, all_predictions_array):
        prediction_row: Dict[str, Any] = {}
        for source_feature in source_features:
            prediction_row = {
                **prediction_row,
                **(tree.attributes.get("source", {}).get(source_feature, {})),
            }

        for fold_index, prediction in enumerate(predictions):
            prediction_row[str(fold_index)] = prediction

        predictions_table.append(prediction_row)

    predictions_tables[forest_name] = predictions_table

with st.expander("Label Statistics"):
    st.table(label_stats)

with st.expander("Prediction Export"):
    st.subheader("Prediction Export")
    for forest_name, predictions_table in predictions_tables.items():
        delimiter_lookup = {"tsv": "\t", "csv": ","}
        st.text(forest_name)
        st.download_button(
            "Download",
            pd.DataFrame(predictions_table)
            .to_csv(index=False, sep=delimiter_lookup.get(str(export_format), ","))
            .encode("utf-8"),
            "predictions-{}.{}".format(forest_name, export_format),
        )

with st.expander("Scikit-Learn Metrics Report"):
    for forest_name, sklearn_report_str in sklearn_reports.items():
        st.code("{}\n---\n{}".format(forest_name, sklearn_report_str))

with st.expander("TeX Tabular Stats"):
    output = sort_table(accuracy_stats, "source_name").to_latex(
        columns=["source_name", "accuracy", "f1"], escape=True
    )
    st.code(output)


st.table(sort_table(accuracy_stats, "source_name"))
