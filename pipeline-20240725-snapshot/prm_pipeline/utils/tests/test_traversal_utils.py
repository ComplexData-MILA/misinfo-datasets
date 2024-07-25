"""
Unit tests for tree traversal coordination utils.
"""
from typing import Dict, List, Tuple

import pytest

from prm_pipeline.utils.traversal_utils import (
    ForestTraversalCoordinator,
    NodeID,
    TreeID,
)


@pytest.fixture()
def forest_traversal_coordinator():
    example_nodes: Dict[Tuple[TreeID, NodeID], int] = {
        (1, "0"): 0,
        (1, "3"): 0,
        (1, "1"): 2,
        (1, "2"): 1,
        (2, "1"): 2,
    }

    return ForestTraversalCoordinator(initial_nodes=example_nodes)


def _verify_get_node_output(
    forest_traversal_coordinator: ForestTraversalCoordinator,
    reference_output_values: List,
):
    for output_reference in reference_output_values:
        if output_reference is not None:
            assert forest_traversal_coordinator.get_node() == output_reference
        else:
            assert forest_traversal_coordinator.get_node() is None

    assert forest_traversal_coordinator.get_node() is None


def test_forest_traversal_max_width_limit(forest_traversal_coordinator):
    """
    Verify return values of get_node in
    forest traversal controller follow max_width.
    """

    _verify_get_node_output(
        forest_traversal_coordinator,
        [(1, "1")] * 2 + [(1, "2")] * 1 + [(2, "1")] * 2 + [None],
    )


def test_forest_traversal_add_node(forest_traversal_coordinator):
    """
    Verify return values of get_node in
    forest traversal controller are correct after
    adding new nodes.
    """
    assert forest_traversal_coordinator.get_node() == (1, "1")

    forest_traversal_coordinator.add_node((1, "3"), 1)
    forest_traversal_coordinator.add_node((2, "2"), 2)

    assert forest_traversal_coordinator.get_node() == (1, "1")

    _verify_get_node_output(
        forest_traversal_coordinator,
        [(1, "2")] * 1 + [(1, "3")] * 1 + [(2, "1")] * 2 + [(2, "2")] * 2 + [None],
    )
