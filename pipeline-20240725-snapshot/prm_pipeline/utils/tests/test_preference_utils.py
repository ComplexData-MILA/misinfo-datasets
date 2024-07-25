import datetime
import json
from typing import Optional

import pytest

from ..preference_utils import (
    Preference,
    PreferenceScore,
    get_pairwise_preferences,
    validate_preference,
)
from ..tree_utils import ForestName, NodeId, ReasoningTree, TreeId
from .test_tree_utils import get_example_reasoning_tree

forest_name: ForestName = "example_forest"
tree_id: TreeId = 0
example_timestamp = datetime.datetime(2023, 12, 28, 12, 12)


@pytest.fixture()
def example_preference() -> Preference:
    return Preference(
        node_ids=["1", "2"],
        preference_scores={"1": 0.0, "2": 1.0},
        label_source="example",
        forest_name=forest_name,
        tree_id=tree_id,
        timestamp=example_timestamp,
    )


@pytest.fixture()
def example_preference_triplet() -> Preference:
    return Preference(
        node_ids=["10", "11", "12"],
        preference_scores={"10": 0.0, "11": 0.5, "12": 1.0},
        label_source="example",
        forest_name=forest_name,
        tree_id=tree_id,
        timestamp=example_timestamp,
    )


@pytest.fixture()
def example_reasoning_tree() -> ReasoningTree:
    return get_example_reasoning_tree(tree_id, forest_name)


def test_preference_serialization_deserialization(example_preference):
    """
    Ensure equivalence when a Preference instance is serialized as JSON
    and instantiated from the JSON representation.
    """
    preference_reference: Preference = example_preference
    preference_data_reference = preference_reference.to_dict()
    preference_data_serialized = json.dumps(preference_data_reference)
    print(preference_data_serialized)

    preference_data = json.loads(preference_data_serialized)
    preference = Preference.from_dict(preference_data)

    assert isinstance(preference.timestamp, datetime.datetime)
    assert preference == preference_reference


def test_validate_preference(example_preference, example_reasoning_tree):
    error_message = validate_preference(example_preference, example_reasoning_tree)
    assert error_message is None


def test_validate_preference_int_datatype(example_preference, example_reasoning_tree):
    preference: Preference = example_preference._replace(
        preference_scores={"1": 0, "2": 1}
    )
    error_message = validate_preference(preference, example_reasoning_tree)
    assert error_message is None


def test_validate_preference_invalid_datatype(
    example_preference, example_reasoning_tree
):
    preference: Preference = example_preference._replace(
        preference_scores={"1": "0", "2": "1"}
    )
    error_message = validate_preference(preference, example_reasoning_tree)
    assert error_message is not None


def test_validate_preference_missing_nodes(example_preference, example_reasoning_tree):
    preference: Preference = example_preference._replace(preference_scores={"2": 1.0})
    error_message = validate_preference(preference, example_reasoning_tree)
    assert error_message is not None

    preference: Preference = example_preference._replace(
        preference_scores={"0": 0.0, "2": 1.0}
    )
    error_message = validate_preference(preference, example_reasoning_tree)
    assert error_message is not None


def test_validate_preference_negative_scores(
    example_preference, example_reasoning_tree
):
    preference: Preference = example_preference._replace(
        preference_scores={"1": -1.0, "2": 1.0}
    )
    error_message = validate_preference(preference, example_reasoning_tree)
    assert error_message is not None


def test_validate_preference_different_predecessors(
    example_preference, example_reasoning_tree
):
    preference: Preference = example_preference._replace(
        node_ids=["1", "5"], preference_scores={"1": 1.0, "5": 1.0}
    )
    error_message = validate_preference(preference, example_reasoning_tree)
    assert error_message is not None


def test_get_pairwise_preferences_unfiltered(example_preference_triplet):
    pairwise_preferences_unfiltered = get_pairwise_preferences(
        example_preference_triplet
    )
    assert len(pairwise_preferences_unfiltered) == 3

    for pairwise_preference in pairwise_preferences_unfiltered:
        assert pairwise_preference.more_preferred != pairwise_preference.less_preferred


def test_get_pairwise_preferences_filtered(example_preference_triplet):
    def _filtering_function(
        score_0: PreferenceScore, score_1: PreferenceScore
    ) -> Optional[bool]:
        if abs(score_0 - score_1) > 0.5:
            return score_0 > score_1

    pairwise_preferences_filtered = get_pairwise_preferences(
        example_preference_triplet, _filtering_function
    )
    assert len(pairwise_preferences_filtered) == 1

    pairwise_preference = pairwise_preferences_filtered[0]
    assert pairwise_preference.more_preferred == "12"
    assert pairwise_preference.less_preferred == "10"
