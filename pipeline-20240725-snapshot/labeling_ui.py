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
import os
import random
from typing import List, Optional, Tuple

import streamlit as st

from prm_pipeline.utils import PathElement
from prm_pipeline.utils.labeling_utils import ManualLabel
from prm_pipeline.utils.tree_utils import NodeId, ReasoningTree
from prm_pipeline.webserver.db_utils import MongoReasoningTreeDB

from utils import (
    submodule_path,
    get_unique_paths,
    get_mongo_db,
    get_included_kv_pairs,
    get_experiment_commit_hash,
    get_labeling_configs,
)

parser = argparse.ArgumentParser()
parser.add_argument("config_file_name")
args, _ = parser.parse_known_args()


if "config_name_override" in st.query_params.keys():
    config_name = os.path.split(st.query_params["config_name_override"])[-1]
else:
    config_name = args.config_file_name


experiment_commit_hash = get_experiment_commit_hash(submodule_path)
labeling_configs = get_labeling_configs(config_name, submodule_path)

forest_names = labeling_configs["forest_names"]
prng_seed = labeling_configs.get("prng_seed", 1)
questions = labeling_configs["questions"]
num_examples = labeling_configs["num_examples"]
source_attributes = labeling_configs["source_attributes"]
config_version = labeling_configs["config_version"]
leaf_attributes = labeling_configs.get("leaf_attributes", ["leaf_value_sum"])
leaf_value_options = labeling_configs.get(
    "leaf_value_options", {"Correct": 1, "Incorrect": 0, "None": None}
)


@st.cache_resource
def get_paths(
    forest_name: str, data_source_name: Optional[str], value_selected: int
) -> List[Tuple[ReasoningTree, NodeId, List[PathElement]]]:
    """
    Params:
        value_selected: node is included only if node_attributes
            leaf_value_sum is specified and matches this value.
            e.g., 0 for incorrect, 1 for correct.
        data_source_name: Optionally, select only trees from the
            given data source.
    """
    db = get_mongo_db()
    reasoning_tree_db = MongoReasoningTreeDB(db, -1)
    trees = reasoning_tree_db.get_forest(forest_name, data_source_name)
    paths: List[Tuple[ReasoningTree, NodeId, List[PathElement]]] = []
    for tree in sorted(trees, key=lambda tree: tree.attributes["tree_id"]):
        for leaf_node_id in sorted(tree.get_leaf_node_ids()):
            path = tree.get_walk(leaf_node_id)
            leaf_value = path[-1].node.attributes.get("leaf_value_sum")
            if (leaf_value == value_selected) and (leaf_node_id != "0"):
                paths.append((tree, leaf_node_id, path))

    return paths


label_collection = get_mongo_db()["manual_labels"]

if "annotator_name" not in st.query_params:
    st.query_params["annotator_name"] = ""
annotator_name = st.sidebar.text_input(
    "Annotator", value=st.query_params["annotator_name"], placeholder="(optional)"
)
if annotator_name is not None:
    annotator_name = annotator_name.lower()
    st.query_params["annotator_name"] = annotator_name


question_map = {question["description"]: question for question in questions}
question_key = st.sidebar.selectbox("Question", options=question_map.keys())
question = question_map[question_key]


if "leaf_value_selected" not in st.query_params:
    st.query_params["leaf_value_selected"] = str(list(leaf_value_options.values())[0])

# Whether parsed prediction of rollout is equal to ground truth.
leaf_value_text = st.sidebar.selectbox(
    "Filter by Accuracy of Prediction",
    options=leaf_value_options.keys(),
    index=list(leaf_value_options.values()).index(
        int(st.query_params["leaf_value_selected"])
    ),
)
if leaf_value_text is None:
    raise ValueError()

leaf_value_selected = leaf_value_options.get(leaf_value_text, 0)


if "forest_name" not in st.query_params:
    st.query_params["forest_name"] = forest_names[0]

if st.query_params["forest_name"] in forest_names:
    forest_name = st.sidebar.radio(
        "Forest Name",
        options=forest_names,
        index=forest_names.index(st.query_params["forest_name"]),
    )

    assert forest_name is not None
    st.query_params["forest_name"] = forest_name
else:
    forest_name = st.query_params["forest_name"]

data_source_name: Optional[str] = None
if "/" in forest_name:
    forest_name, data_source_name = forest_name.split("/", maxsplit=1)

paths = get_paths(forest_name, data_source_name, leaf_value_selected)
generator = random.Random(prng_seed)
paths_to_label_randomized = generator.sample(paths, min(2 * num_examples, len(paths)))
paths_to_label = get_unique_paths(paths_to_label_randomized)[:num_examples]


def handle_answer_select():
    """
    Identify key that changed value.
    """

    updated_documents: List[ManualLabel] = []
    for question in questions:
        for tree, leaf_node_id, _ in paths_to_label:
            tree_id = tree.attributes["tree_id"]
            button_key = "{}/{}/{}".format(tree_id, leaf_node_id, question["key"])
            button_key_prev = button_key + "/_prev"
            value = st.session_state.get(button_key)
            label_document: ManualLabel = st.session_state.label_documents[button_key]

            if value != label_document.attribute_value:
                label_document = label_document._replace(attribute_value=value)
                updated_documents.append(label_document)
                label_document = st.session_state.label_documents[button_key] = (
                    label_document
                )

            st.session_state[button_key_prev] = value

    for label_document in updated_documents:
        label_document.upsert(label_collection)


st.sidebar.text(
    "labeling config: {}"
    "\n"
    "version {} ({})".format(config_name, config_version, experiment_commit_hash)
)

if "label_documents" not in st.session_state:
    # map button_key to label documents
    st.session_state.label_documents = {}

for index, (tree, leaf_node_id, path_selected) in enumerate(paths_to_label):
    prompt_messages = [path_element.node.data for path_element in path_selected]
    rollout_info = {
        **get_included_kv_pairs(tree.attributes["source"], source_attributes),
        **get_included_kv_pairs(path_selected[-1].node.attributes, leaf_attributes),
        "prediction": path_selected[-1].node.attributes.get("prediction"),
    }

    st.header("{}/{}".format(index + 1, len(paths_to_label)))
    st.link_button(
        "Viewer",
        "/".join(
            [
                os.environ["WEBSERVER_API_BASE"].rstrip("/"),
                str(forest_name),
                str(tree.attributes["tree_id"]),
                str(leaf_node_id),
            ]
        ),
    )
    st.table({str(k): str(v) for k, v in rollout_info.items()})

    # For each question for the annotator,
    # try retrieving previous selection for this example.
    description = question["description"]
    options: List = question["options"]
    tree_id = tree.attributes["tree_id"]
    question_key = question["key"]
    button_key = "{}/{}/{}".format(tree_id, leaf_node_id, question_key)
    button_key_prev = button_key + "/_prev"

    # use placeholder if new and not found in DB
    label_document = ManualLabel(
        forest_name,
        tree_id,
        leaf_node_id,
        config_version,
        experiment_commit_hash,
        annotator_name,
        question_key,
        None,
        data_source_name=data_source_name,
    )
    previous_label_instance = label_document.get_current_value(label_collection)
    if previous_label_instance is not None:
        label_document = previous_label_instance

    st.session_state.label_documents[button_key] = label_document

    placeholder_choice_index = None
    comment = None

    if label_document.attribute_value is not None:
        # Set label value in label_document.
        placeholder_choice_index = options.index(label_document.attribute_value)
        description = "{} (previously selected: {})".format(
            description, label_document.attribute_value
        )

    if label_document.comment is not None:
        previous_value = label_document.attribute_value
        comment = label_document.comment

    st.divider()
    label_value = st.radio(
        description,
        options,
        key=button_key,
        index=placeholder_choice_index,
        on_change=handle_answer_select,
    )

    comment = st.text_area(
        "Comments", value=comment, key="{}/comments".format(button_key)
    )
    if st.button("Save", key="{}/save_comment_button".format(button_key)):
        # If user cleared comment, set the value to None.
        if (comment is not None) and len(comment) == 0:
            comment = None

        label_document = label_document._replace(comment=comment)
        st.session_state.label_documents[button_key] = label_document
        label_document.upsert(label_collection)

    st.divider()
