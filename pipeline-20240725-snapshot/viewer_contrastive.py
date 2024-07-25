"""
Web interface for comparing rollouts from two different forests.

Data source:
- reasoning_trees collection.
"""

import os
from typing import List, Tuple

import streamlit as st

from prm_pipeline.utils import PathElement
from prm_pipeline.utils.text_generation import get_conversation_prompt
from prm_pipeline.utils.tree_utils import NodeId, ReasoningTree
from prm_pipeline.webserver.db_utils import MongoReasoningTreeDB

from utils import align_paths, get_mongo_db, get_included_kv_pairs

SOURCE_ATTRIBUTES = [
    "statement",
    "article_summary",
    "label",
    "claim",
    "evidence_wiki_url",
]
LEAF_ATTRIBUTES = []
VALUE_NAMES = ["Correct", "Incorrect"]


@st.cache_resource
def get_paths(
    forest_name: str, value_selected: int
) -> List[Tuple[ReasoningTree, NodeId, List[PathElement]]]:
    """
    Params:
        value_selected: node is included only if node_attributes
            leaf_value_sum is specified and matches this value.
            e.g., 0 for incorrect, 1 for correct.
    """
    db = get_mongo_db()
    reasoning_tree_db = MongoReasoningTreeDB(db, -1)
    trees = reasoning_tree_db.get_forest(forest_name)
    paths: List[Tuple[ReasoningTree, NodeId, List[PathElement]]] = []
    for tree in sorted(trees, key=lambda tree: tree.attributes["tree_id"]):
        tree.attributes["_forest_name"] = forest_name
        for leaf_node_id in sorted(tree.get_leaf_node_ids()):
            path = tree.get_walk(leaf_node_id)
            leaf_value = path[-1].node.attributes.get("leaf_value_sum")
            if (leaf_value == value_selected) and (leaf_node_id != "0"):
                paths.append((tree, leaf_node_id, path))

    return paths


label_collection = get_mongo_db()["manual_labels"]

forest_name_correct = st.sidebar.text_input(
    "Forest (Correct Predictions)", value=st.query_params.get("correct")
)
if forest_name_correct is not None:
    st.query_params["correct"] = forest_name_correct

forest_name_incorrect = st.sidebar.text_input(
    "Forest (Incorrect Predictions)", value=st.query_params.get("incorrect")
)
if forest_name_incorrect is not None:
    st.query_params["incorrect"] = forest_name_incorrect

if st.sidebar.button("\u21cc"):
    if (forest_name_correct is not None) and (forest_name_incorrect is not None):
        st.query_params["correct"] = forest_name_incorrect
        st.query_params["incorrect"] = forest_name_correct


paths = []
if (forest_name_correct is not None) and (forest_name_incorrect is not None):
    paths = [*get_paths(forest_name_correct, 1), *get_paths(forest_name_incorrect, 0)]


def _get_tree_id(path_tuple: Tuple[ReasoningTree, NodeId, List[PathElement]]) -> str:
    return path_tuple[0].attributes["tree_id"]


def _get_forest_name(
    path_tuple: Tuple[ReasoningTree, NodeId, List[PathElement]]
) -> str:
    return path_tuple[0].attributes["_forest_name"]


paths_aligned = align_paths(paths, _get_tree_id, _get_forest_name)

# Keep only ones where prediction differ.
path_pairs_selected = [
    path_dict for path_dict in paths_aligned.values() if len(path_dict.keys()) > 1
][1:100]


for index, paths_aligned in enumerate(path_pairs_selected):
    st.header("{}/{}".format(index + 1, len(path_pairs_selected)))

    for index, (forest_name, (tree, leaf_node_id, path_selected)) in enumerate(
        paths_aligned.items()
    ):

        prompt_messages = [path_element.node.data for path_element in path_selected]
        conversation = get_conversation_prompt(prompt_messages)
        rollout_info = {
            **get_included_kv_pairs(tree.attributes["source"], SOURCE_ATTRIBUTES),
            **get_included_kv_pairs(path_selected[-1].node.attributes, LEAF_ATTRIBUTES),
            "prediction": path_selected[-1].node.attributes.get("prediction"),
        }

        if index == 0:
            st.table(rollout_info)

        viewer_url = "/".join(
            [
                os.environ["WEBSERVER_API_BASE"].rstrip("/"),
                str(tree.attributes["_forest_name"]),
                str(tree.attributes["tree_id"]),
                str(leaf_node_id),
            ]
        )

        st.subheader("{}".format(VALUE_NAMES[index]))
        st.markdown("```{}```".format(viewer_url))
        st.link_button("Viewer", viewer_url)

    st.divider()
