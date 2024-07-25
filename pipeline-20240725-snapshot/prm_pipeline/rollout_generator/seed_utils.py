"""
Utils for instantiating reasoning trees
within a forest based on a given dataset.
- Instantiate a tree given 
    - a dataset entry, and 
    - a prompt template
"""
from typing import Any, Dict, Mapping

from .._experiment_configs.interfaces import RolloutConfig
from ..utils.tree_utils import ReasoningTree, ReasoningTreeNode


def initialize_tree_from_dataset_entry(
    dataset_entry: Mapping, rollout_config: RolloutConfig
) -> ReasoningTree:
    """
    Instantiate a reasoning tree given a dataset entry
    and a template for the root prompt.

    Root node data are compatible with the ChatCompletion API.

    Params:
        dataset_entry: Mapping, where each key represents a feature.
        rollout_config: instance of RolloutConfig.
    """

    root_prompt = rollout_config.get_root_prompt(dataset_entry)
    ground_truth = rollout_config.get_label(dataset_entry)

    tree_attributes = {
        "rollout_worker_config": rollout_config.rollout_worker_config._asdict(),
        "source": dataset_entry,
        "ground_truth": ground_truth,
        "forest_info": rollout_config.forest_info,
        "dataset_feature_names": rollout_config.dataset_feature_names,
        "actions_config": rollout_config.actions_config.to_dict(),
    }

    root_data = {"role": "user", "content": root_prompt}
    root_node_attributes = {
        "num_children": rollout_config.root_num_children,
        "node_id": "0",
    }
    root_node = ReasoningTreeNode(root_node_attributes, root_data)

    # Instantiate reasoning tree
    reasoning_tree = ReasoningTree(attributes=tree_attributes, nodes={"0": root_node})

    return reasoning_tree


def validate_config(forest_config: Dict[str, Any]):
    """
    Validate forest config.

    Params:
        forest_config: Dict, from config.json.

    Raises:
        AssertionError or KeyError if config is invalid.

    TODO: replace rollout worker config with dataclass.
    """
    assert isinstance(forest_config["root_template"], str)
    assert isinstance(forest_config["default_num_children"], int)

    for key in ["rollout_delimiter_pattern", "prediction_pattern"]:
        assert isinstance(forest_config["rollout_config"][key], str)
