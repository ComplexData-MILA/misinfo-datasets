import os
import time
from typing import Optional

import streamlit as st

from prm_pipeline._experiment_configs.rollout.liar_new_eval import Config
from prm_pipeline.rollout_generator.functions.reasoning import generate_reasoning
from prm_pipeline.rollout_generator.functions.search import generate_search
from prm_pipeline.rollout_generator.seed_utils import initialize_tree_from_dataset_entry
from prm_pipeline.utils.action_utils import ActionsConfig, process_actions
from prm_pipeline.utils.submodule_utils import get_commit_hash
from prm_pipeline.utils.tree_utils import PathElement
from prm_pipeline.webserver.db_utils import MongoReasoningTreeDB
from prm_pipeline.webserver.serialization_utils import RolloutSubmission, RolloutTask
from utils import get_mongo_db

rollout_config = Config()
query_text = st.text_input(
    "Query",
    value="Comparing the price of oil and gas in June 2008 to"
    " March 2022 shows that oil companies are price gouging.",
)

tree_db = MongoReasoningTreeDB(get_mongo_db(), 0)
sharable_forest_name = "interactive_{}".format(
    get_commit_hash("prm_pipeline/_experiment_configs"),
)
is_shared = st.checkbox("Make my query publicly visible", value=True)

if st.button("Submit"):
    dataset_entry = {"statement": query_text, "label": None}
    tree = initialize_tree_from_dataset_entry(dataset_entry, rollout_config)

    rollout_task = RolloutTask(
        -1,
        tree.attributes,
        PathElement.serialize_path(tree.get_walk("0")),
    )

    num_additional_children: int = 1
    for _ in range(7):
        action_name = (
            rollout_task.rollout_context[-1]["attributes"].get("action", {}).get("name")
        )

        rollout_submit_payload: Optional[RolloutSubmission] = None
        while rollout_submit_payload is None:
            if action_name == "search":
                with st.spinner(f"Running {action_name}"):
                    rollout_submit_payload = generate_search(rollout_task)
            else:
                with st.spinner(f"Analyzing"):
                    rollout_submit_payload = generate_reasoning(
                        rollout_task, os.environ.get("ENDPOINT_NAME", "Completion")
                    )

        actions_config_data = rollout_task.tree_attributes.get("actions_config")
        if actions_config_data is not None:
            actions_config = ActionsConfig.from_dict(actions_config_data)
            new_nodes_updated = process_actions(
                rollout_submit_payload.nodes, actions_config
            )
            rollout_submit_payload = rollout_submit_payload._replace(
                nodes=new_nodes_updated
            )

        parent_node_id = rollout_submit_payload.ancestor_node_id
        for node in rollout_submit_payload.nodes:
            # Add node to tree and obtain node_id.
            new_node_id = tree.add_node(node, parent_node_id)
            parent_node_id = new_node_id
            print(node)
            st.table(node.data)

        num_additional_children = rollout_submit_payload.nodes[-1].attributes.get(
            "num_children", 0
        )
        if num_additional_children <= 0:
            break

        rollout_task = RolloutTask(
            -1,
            tree.attributes,
            PathElement.serialize_path(tree.get_walk(new_node_id)),
        )

    if "prediction" in node.attributes:
        st.text("Prediction: {} statement".format(bool(node.attributes["prediction"])))

    if is_shared:
        tree_id = int(time.time())
        tree_db.upload_tree((sharable_forest_name, tree_id), tree)
        url = "{}{}/{}".format(
            os.environ.get("WEBSERVER_API_BASE"), sharable_forest_name, tree_id
        )
        print(url)
        st.code(url, language=None)
