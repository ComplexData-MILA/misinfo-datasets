"""
Web interface for presenting summary 
of a given labeling configuration.
"""

import argparse
from collections import Counter, defaultdict
import os
from typing import Any, Dict, List

from tqdm.auto import tqdm
import pandas
import streamlit as st

from prm_pipeline.utils.labeling_utils import ManualLabel
from prm_pipeline.webserver.db_utils import MongoReasoningTreeDB

from utils import (
    get_included_kv_pairs,
    get_mongo_db,
    get_labeling_configs,
    aggregate_by_key,
    render_aggregations,
    get_viewer_url,
    filter_newlines,
)


parser = argparse.ArgumentParser()
parser.add_argument("config_file_name")
args, _ = parser.parse_known_args()


if "config_name_override" in st.query_params.keys():
    config_name = os.path.split(st.query_params["config_name_override"])[-1]
else:
    config_name = args.config_file_name


@st.cache_data
def get_manual_label_table(labels: List[ManualLabel]) -> List[Dict[str, Any]]:
    """
    Given a list of manual label items,
    return a tabular representation of the content.

    Params:
        labels: list of manual label items.

    Returns:
        list of dict, one for each table rows.
        Labels are grouped into columns, as specified in group_by.
    """
    db = get_mongo_db()
    reasoning_tree_db = MongoReasoningTreeDB(db, -1)
    output_table: List[Dict[str, Any]] = []
    labeling_configs = get_labeling_configs(config_name)
    source_attributes = labeling_configs["source_attributes"]
    leaf_attributes = labeling_configs.get("leaf_attributes", ["leaf_value_sum"])

    for label_instance in tqdm(labels):
        tree = reasoning_tree_db.cache.get(
            (label_instance.forest_name, label_instance.tree_id)
        )
        if tree is None:
            continue

        path = tree.get_walk(label_instance.node_id)
        tree_attributes: Dict[str, Any] = {
            **get_included_kv_pairs(tree.attributes["source"], source_attributes),
            **get_included_kv_pairs(path[-1].node.attributes, leaf_attributes),
            "is_correct": tree.attributes.get("ground_truth")
            == path[-1].node.attributes.get("prediction"),
        }
        table_row: Dict[str, Any] = {**tree_attributes, **label_instance._asdict()}
        output_table.append(table_row)

    return output_table


label_collection = get_mongo_db()["manual_labels"]

if "config_git_hash" not in st.query_params:
    st.query_params["config_git_hash"] = ""
config_git_hash = st.text_input(
    "Config git hash", value=st.query_params["config_git_hash"]
)
assert config_git_hash is not None
st.query_params["config_git_hash"] = config_git_hash

config_git_hash_values = config_git_hash.split("-")
label_records = list(
    label_collection.find({"config_commit_hash": {"$in": config_git_hash_values}})
)


# Group by Forest Name
all_labels: Dict[str, List[ManualLabel]] = defaultdict(list)
for label_record in label_records:
    label_record.pop("_id")
    label_item = ManualLabel(**label_record)
    all_labels[label_item.forest_name].append(label_item)

for forest_name, forest_labels in all_labels.items():
    st.subheader(forest_name)
    table = get_manual_label_table(forest_labels)
    aggregations = aggregate_by_key(
        table, ("tree_id", "node_id"), "annotator", ("attribute_value", "comment")
    )
    name_counters = defaultdict(Counter)
    for aggregation in aggregations.values():
        for name, values_by_key in aggregation.values.items():
            for key, value in values_by_key.items():
                if value is not None:
                    name_counters[name][key] += 1

    counter_info = pandas.DataFrame(name_counters)
    counter_info = counter_info.fillna(0).T.convert_dtypes(convert_integer=True)
    st.table(counter_info)
    output_table = []
    for row in render_aggregations(aggregations):
        viewer_url = get_viewer_url(row["forest_name"], row["tree_id"], row["node_id"])
        row = {"url": viewer_url, **row}
        output_table.append(filter_newlines(row))

    output_table_pd = pandas.DataFrame(output_table)
    st.download_button(
        "Download",
        output_table_pd.to_csv(index=False, sep="\t").encode("utf-8"),
        "{}-{}.tsv".format(forest_name, config_git_hash),
    )
