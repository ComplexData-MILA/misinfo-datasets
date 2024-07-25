"""
Tests for utils that build a forest from a 
given dataset and prompt template.
"""
import os
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional

import pytest

from ..._experiment_configs.interfaces import RolloutConfig, RolloutWorkerConfig
from ...utils.submodule_utils import get_commit_hash, has_uncommitted_changes
from ..seed_utils import initialize_tree_from_dataset_entry

SUBMODULE_PATH = "prm_pipeline/_experiment_configs/"

# To avoid confusion, do not use a `.py` file extension.
TEST_FILE_NAME = "rollout/test-file"


class ExampleRolloutConfig(RolloutConfig):
    root_num_children: int = 7

    rollout_worker_config = RolloutWorkerConfig(
        rollout_delimiter_pattern=r"Thought: \d+",
        prediction_pattern=r": \(([0-9])\)",
        max_branching_altitude=10,
        max_num_retries=10,
        max_num_children=2,
    )

    forest_info: Dict[str, Any] = {
        "prompt_template": "Example prompt template",
    }
    dataset_feature_names: List[str] = ["text", "label"]

    @staticmethod
    def get_dataset_iterator(split: str) -> Iterator[Mapping]:
        return [
            {"split": split, "label": "A", "text": "Text 1"},
            {"split": split, "label": "B", "text": "Text 2"},
        ].__iter__()

    @staticmethod
    def serialize_dataset_item(dataset_entry: Mapping) -> Dict[str, Any]:
        """
        Create a JSON-serializable representation of one item
        from the dataset (e.g., for reference in the tree attributes.)

        Params:
            dataset_entry: Mapping, see get_dataset_iterator.

        Returns:
            JSON-serializable representation of the given
            dataset item.
        """
        return dict(dataset_entry)

    @staticmethod
    def get_root_prompt(dataset_entry: Mapping) -> str:
        """
        Return the question to the LLM about the given item
        from the dataset.

        Params:
            dataset_entry: Mapping, see get_dataset_iterator.

        Returns:
            a string denoting the initial instruction for the language model
        """
        template = "Example prompt for text: {text}"
        return template.format(**dataset_entry)

    @staticmethod
    def get_label(dataset_entry: Mapping) -> Optional[Any]:
        """
        If applicable, return the ground truth label for this
        particular item in the dataset. Note that some datasets
        are unlabelled- return None in these cases.

        Params:
            dataset_entry: Mapping, see get_dataset_iterator.

        Returns:
            A comparable label for the dataset item, or
            None if not applicable for this task.
        """
        label_lookup = {"A": 0, "B": 1}
        return label_lookup.get(dataset_entry["label"], -1)


def test_initialize_tree():
    """
    Test instantiate tree on an example dataset for an example template.
    """
    rollout_config = ExampleRolloutConfig()
    dataset_iterator = rollout_config.get_dataset_iterator("test")
    dataset_items = list(dataset_iterator)
    assert len(dataset_items) == 2
    dataset_item = dataset_items[0]

    tree = initialize_tree_from_dataset_entry(dataset_item, rollout_config)

    # Verify tree attributes
    assert (
        tree.attributes["rollout_worker_config"]
        == rollout_config.rollout_worker_config._asdict()
    )
    assert tree.attributes["ground_truth"] == 0
    assert tree.attributes["source"] == dict(dataset_item)
    assert tree.attributes["forest_info"] == dict(rollout_config.forest_info)
    assert tree.attributes["dataset_feature_names"] == (
        rollout_config.dataset_feature_names
    )

    # Verify root node attributes
    assert len(tree.nodes.values()) == 1
    root_node = list(tree.nodes.values())[0]
    assert root_node.data["role"] == "user"
    assert root_node.data["content"] == "Example prompt for text: Text 1"
    assert root_node.attributes["num_children"] == rollout_config.root_num_children


def test_git_has_uncommitted_changes():
    test_file_path = os.path.join(SUBMODULE_PATH, TEST_FILE_NAME)
    assert not os.path.exists(test_file_path), (
        "path `{}` exists; "
        "Please use a test file name that "
        "won't be confused with regular files".format(test_file_path)
    )

    assert not has_uncommitted_changes(SUBMODULE_PATH, TEST_FILE_NAME)

    # Create a test file and check again, before deleting the test file.
    with open(test_file_path, "w") as _:
        assert has_uncommitted_changes(SUBMODULE_PATH, TEST_FILE_NAME)

    os.remove(test_file_path)
    assert not has_uncommitted_changes(SUBMODULE_PATH, TEST_FILE_NAME)


def test_git_get_commit_hash():
    commit_hash = get_commit_hash(SUBMODULE_PATH)

    print("commit_hash: {}".format(commit_hash))
    assert re.match(r"^[0-9a-z]{9}$", commit_hash) is not None
