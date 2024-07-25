"""
Functions for adding "search" nodes.
"""
import os
from typing import Dict, Optional, Any, Tuple
import logging

import cohere

from ...webserver.serialization_utils import RolloutSubmission, RolloutTask
from ..._experiment_configs.interfaces import RolloutWorkerConfig
from ...utils import ActionsConfig, ReasoningTreeNode


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_search(rollout_task: RolloutTask) -> Optional[RolloutSubmission]:
    """
    Conduct research on the given input and
    return the research response as a new node.

    Params:
        rollout_task: RolloutTask with the most recent node in context
            containing research query.

    Returns:
        RolloutSubmission with research response,
        or None if error is encountered.

    Extract research query from rollout context.
    Invoke research API.
    Instantiate rollout submission.
    """
    rollout_config_dict = rollout_task.tree_attributes["rollout_worker_config"]
    rollout_config = RolloutWorkerConfig(**rollout_config_dict)
    actions_config_data = rollout_task.tree_attributes["actions_config"]
    actions_config = ActionsConfig.from_dict(actions_config_data)
    action_settings = actions_config.get_action_by_name("search")
    assert action_settings is not None

    parent_node = rollout_task.rollout_context[-1]
    query_payload: Tuple[Any] = parent_node["attributes"]["action"]["payload"]
    assert len(query_payload) == 1

    query_text = query_payload[0]
    assert isinstance(query_text, str)

    api_key = os.environ["COHERE_KEY"]
    model_name = os.environ["COHERE_SEARCH_MODEL_NAME"]

    try:
        client = cohere.Client(api_key)
        response = client.chat(
            message=query_text,
            connectors=[{"id": "web-search"}],
            model=model_name,
        )
        query_response: str = response.text
        node_data = {
            "content": action_settings.response_template.format(query_response),
            "role": action_settings.response_role,
            "source": "search",
            "model_name": model_name,
        }
    except Exception as e:
        print(e)
        return

    parent_altitude = parent_node["attributes"]["altitude"]
    response_serializable = response.__dict__
    response_serializable.pop("client")
    node_attributes: Dict[str, Any] = {
        "num_children": rollout_config.max_num_children + 2,
        "altitude": parent_altitude + 1,
        "action_response": {
            "name": "search",
            "query": query_text,
            "response": response_serializable,
        },
        "token_count": response.token_count,
    }

    response_node = ReasoningTreeNode(node_attributes, node_data)
    rollout_submission = RolloutSubmission(
        tree_id=rollout_task.tree_id,
        ancestor_node_id=rollout_task.parent_node_id,
        nodes=[response_node],
    )
    return rollout_submission
