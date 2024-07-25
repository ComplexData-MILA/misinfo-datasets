"""
Utils for testing tree utils.

- Test JSON serialization and deserialization.
"""

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

from prm_pipeline.utils.tree_utils import (
    Cache,
    ForestTraversalManager,
    NodeId,
    NodeTuple,
    PathElement,
    Prediction,
    ReasoningTree,
    ReasoningTreeIdTuple,
    ReasoningTreeNode,
    get_altitude_map,
    get_next_node_id,
    get_value_estimate,
    invert_edges,
)

from prm_pipeline.utils import Action, ActionsConfig


def get_example_reasoning_node(
    index: str = "0",
    num_children: Optional[int] = None,
    num_leaf_nodes: Optional[int] = None,
    leaf_value_sum: Optional[int] = None,
    prediction: Optional[Prediction] = None,
) -> ReasoningTreeNode:
    """
    Create example node for reasoning tree.
    """
    role = "assistant"
    if index == "0":
        role = "user"

    # Set default value of num_children if not specified.
    if num_children is None:
        try:
            node_index = int(index.split("_")[-1])

            if node_index == 6:
                # Emulate node that requires function calling.
                num_children = 1
            elif node_index > 2:
                num_children = 0
            else:
                num_children = 2
        except ValueError:
            num_children = 0

    node_attribute: Dict[str, Any] = {
        "node_id": index,
        "num_children": num_children,
    }

    for optional_key, value in [
        ("num_leaf_nodes", num_leaf_nodes),
        ("leaf_value_sum", leaf_value_sum),
        ("prediction", prediction),
    ]:
        if value is not None:
            node_attribute[optional_key] = value

    node_data: Dict[str, Any] = {
        "role": role,
        "content": "Example rollout for node {}. ".format(index),
    }
    node = ReasoningTreeNode(node_attribute, node_data)

    return node


@pytest.fixture()
def reasoning_tree_node():
    """
    Example node of a reasoning tree.
    """
    return get_example_reasoning_node("example")


def get_example_edges() -> Dict[NodeId, List[NodeId]]:
    # Prevents test cases from modifying
    # the edges of other test cases
    return {"0": ["1", "2"], "1": ["3", "4"], "2": ["5", "6"]}


def get_example_num_leaf_nodes() -> Dict[NodeId, int]:
    """
    Assume same edge layout as output of get_example_edges
    """
    return {"0": 4, "1": 2, "2": 2, "3": 1, "4": 1, "5": 1, "6": 1}


def get_example_leaf_value_sum() -> Dict[NodeId, int]:
    # Assume that leaf 3 is incorrect, but 4 ~ 6 are correct.
    return {"0": 3, "1": 1, "2": 2, "3": 0, "4": 1, "5": 1, "6": 1}


def get_example_leaf_predictions() -> Dict[NodeId, Prediction]:
    # Should be consistent with get_example_leaf_value_sum.
    return {"3": 0, "4": 1, "5": 1, "6": 1}


def _extract_attributes(path: List[PathElement], node_attribute_key: str) -> List[Any]:
    """
    Extract the given node_attribute from each node in the path.
    Return a list of attribute values, one for each node in the same
    order as the original path input.

    Params:
        path: list of PathElement
        node_attribute_key: str, name of attribute to extract.

    Returns:
        List of attribute values where nodes are in the same order
        as they are in path.
    """
    return [element.node.attributes[node_attribute_key] for element in path]


def get_example_reasoning_tree(
    tree_id: int = 0,
    forest_name: str = "default_forest",
    num_children_override: Optional[int] = None,
) -> ReasoningTree:
    """
    Example reasoning tree.

    Set num_children_override to an integer to override
    the number of children for all initial nodes of this tree.
    """
    tree_nodes = {}
    tree_num_leaf_nodes = get_example_num_leaf_nodes()
    tree_leaf_value_sum = get_example_leaf_value_sum()
    tree_leaf_predictions = get_example_leaf_predictions()
    for node_index in range(7):
        node = get_example_reasoning_node(
            str(node_index),
            num_children_override,
            tree_num_leaf_nodes[str(node_index)],
            tree_leaf_value_sum[str(node_index)],
            tree_leaf_predictions.get(str(node_index)),
        )
        tree_nodes[str(node_index)] = node

    tree_attributes = {
        "forest_name": forest_name,
        "tree_id": tree_id,
        "prompt_template": "Example template.",
        "actions_config": ActionsConfig(
            [Action(name="search", pattern=r"Question:[\s\n]*([^?]+\?)")]
        ).to_dict(),
    }

    tree = ReasoningTree(tree_attributes, tree_nodes, get_example_edges())
    return tree


@pytest.fixture()
def reasoning_tree():
    return get_example_reasoning_tree(0)


def test_tree_node_to_dict(reasoning_tree_node):
    """
    Verify format and content of dictionary representation of
    reasoning tree node.
    """
    dict_representation = reasoning_tree_node.to_dict()
    assert dict_representation == {
        "attributes": {"node_id": "example", "num_children": 0},
        "data": {"role": "assistant", "content": "Example rollout for node example. "},
    }


def test_tree_to_dict(reasoning_tree):
    """
    Test obtaining a JSON-serializable dictionary representation
    for the tree.
    """
    representation = reasoning_tree.to_dict()
    assert representation == {
        "nodes": {k: v.to_dict() for k, v in reasoning_tree.nodes.items()},
        "edges": {"0": ["1", "2"], "1": ["3", "4"], "2": ["5", "6"]},
        "attributes": {
            "forest_name": "default_forest",
            "tree_id": 0,
            "prompt_template": "Example template.",
        },
    }


def test_tree_node_from_dict():
    """
    Verify tree nodes are properly restored from dictionary.
    """
    dict_representation = {
        "attributes": {"node_id": "index_example"},
        "data": {"role": "assistant", "content": "Example rollout for node example. "},
    }
    tree_node = ReasoningTreeNode.from_dict(dict_representation)
    assert tree_node.attributes == {"node_id": "index_example"}
    assert tree_node.data == {
        "role": "assistant",
        "content": "Example rollout for node example. ",
    }


def test_tree_from_dict():
    """
    Test restoring tree from dictionary.
    """
    tree_representation = {
        "nodes": {
            "0": {
                "attributes": {"node_id": "index_0"},
                "data": {"role": "user", "content": "Example rollout for node 0. "},
            },
            "1": {
                "attributes": {"node_id": "index_1"},
                "data": {
                    "role": "assistant",
                    "content": "Example rollout for node 1. ",
                },
            },
        },
        "edges": {"0": ["1"]},
        "attributes": {
            "forest_name": "example_forest",
            "tree_id": 0,
            "prompt_template": "Example template.",
        },
    }
    tree = ReasoningTree.from_dict(tree_representation)
    assert tree.edges == {"0": ["1"]}
    assert tree.attributes == {
        "forest_name": "example_forest",
        "tree_id": 0,
        "prompt_template": "Example template.",
    }
    assert tree.nodes["0"].attributes == {"node_id": "index_0", "altitude": 1}
    assert tree.nodes["1"].attributes == {"node_id": "index_1", "altitude": 2}


class _ExampleTreeRetrieval:
    """
    Counter for number of times get_example_tree was invoked.

    Requests that returned None are also included in the count.
    """

    def __init__(self):
        self.num_invokations = 0

    def get_tree(self, key: ReasoningTreeIdTuple) -> Optional[ReasoningTree]:
        """
        Emulate database retrieval and deserialization of trees.

        Tree with tree_id 31 and forest_name "example_forest" would
        emulate "not found".
        """
        self.num_invokations += 1
        forest_name, tree_id = key

        if (tree_id == 31) and (forest_name == "example_forest"):
            return None

        forest_name, tree_id = key
        return get_example_reasoning_tree(tree_id=tree_id, forest_name=forest_name)


def _verify_tree_attributes(tree: Optional[ReasoningTree], forest_name, tree_id):
    assert isinstance(tree, ReasoningTree)
    assert tree.attributes["forest_name"] == forest_name
    assert tree.attributes["tree_id"] == tree_id


def test_tree_cache():
    """
    Test local reasoning tree cache.

    Example cache capacity: 2

    Try retrieving trees in this order:
    """
    example_tree_retrieval_counter = _ExampleTreeRetrieval()
    tree_cache = Cache(
        capacity=2, get_item_uncached=example_tree_retrieval_counter.get_tree
    )

    example_forest_name = "example_forest"

    # 1: example_forest/101 (101)
    for _ in range(1):
        tree = tree_cache.get(("example_forest", 101))
        _verify_tree_attributes(tree, example_forest_name, 101)
        assert example_tree_retrieval_counter.num_invokations == 1

    # 2: example_forest/102 (101, 102)
    # 2: example_forest/102 (101, 102)
    # 2: example_forest/102 (101, 102)
    for _ in range(3):
        tree = tree_cache.get(("example_forest", 102))
        _verify_tree_attributes(tree, example_forest_name, 102)
        assert example_tree_retrieval_counter.num_invokations == 2

    # 2: example_forest/101 (102, 101)
    for _ in range(1):
        tree = tree_cache.get(("example_forest", 101))
        _verify_tree_attributes(tree, example_forest_name, 101)
        assert example_tree_retrieval_counter.num_invokations == 2

    # 3: example_forest/103 (103, 101)
    # 3: example_forest/103 (103, 101)
    for _ in range(2):
        tree = tree_cache.get(("example_forest", 103))
        _verify_tree_attributes(tree, example_forest_name, 103)
        assert example_tree_retrieval_counter.num_invokations == 3

    # 4: example_forest/102 (101, 102)
    tree = tree_cache.get(("example_forest", 102))
    _verify_tree_attributes(tree, example_forest_name, 102)
    assert example_tree_retrieval_counter.num_invokations == 4

    # 5: example_forest/101 (102, 101)
    tree = tree_cache.get(("example_forest", 101))
    _verify_tree_attributes(tree, example_forest_name, 101)
    assert example_tree_retrieval_counter.num_invokations == 5

    # 6: example_forest/31 (None) (103, 101)
    # 7: example_forest/31 (None) (103, 101)
    tree = tree_cache.get(("example_forest", 31))
    assert tree is None
    assert example_tree_retrieval_counter.num_invokations == 6

    tree = tree_cache.get(("example_forest", 31))
    assert tree is None
    assert example_tree_retrieval_counter.num_invokations == 7

    # 8: example_forest/103 (101, 103) (pop 102 again)
    tree = tree_cache.get(("example_forest", 103))
    _verify_tree_attributes(tree, example_forest_name, 103)
    assert example_tree_retrieval_counter.num_invokations == 8


def test_invert_edges():
    """
    Test edge inversion.
    """
    example_edges = get_example_edges()
    assert invert_edges(example_edges) == {
        "1": "0",
        "2": "0",
        "3": "1",
        "4": "1",
        "5": "2",
        "6": "2",
    }


def test_altitude():
    example_edges = get_example_edges()
    altitude_map = get_altitude_map(example_edges)

    # root node:
    assert altitude_map["0"] == 1

    for node_id_int in range(1, 2 + 1):
        assert altitude_map[str(node_id_int)] == 2

    # leaf nodes
    for node_id_int in range(3, 6 + 1):
        assert altitude_map[str(node_id_int)] == 3


def test_get_siblings(reasoning_tree: ReasoningTree):
    # root node has only itself as the output
    siblings = reasoning_tree.get_siblings("0")
    assert len(siblings) == 1
    assert siblings[0].attributes["node_id"] == "0"

    # middle node
    siblings = reasoning_tree.get_siblings("1")
    assert len(siblings) == 2
    assert set([node.attributes["node_id"] for node in siblings]) == set(["1", "2"])

    # leaf node
    siblings = reasoning_tree.get_siblings("5")
    assert len(siblings) == 2
    assert set([node.attributes["node_id"] for node in siblings]) == set(["5", "6"])


def test_is_leaf(reasoning_tree):
    for node_id_int in range(0, 2 + 1):
        assert not reasoning_tree.is_leaf(str(node_id_int))

    for node_id_int in range(3, 6 + 1):
        assert reasoning_tree.is_leaf(str(node_id_int))


def test_get_leaf_node_ids(reasoning_tree):
    leaf_node_ids_reference = ["3", "4", "5", "6"]
    leaf_node_ids = reasoning_tree.get_leaf_node_ids()

    assert set(leaf_node_ids) == set(leaf_node_ids_reference)


def _verify_path(path: List[PathElement], reference: List[NodeId]):
    """
    Assert that node_id along path match reference.
    """
    node_ids_actual = []
    for (node_id, node, siblings), node_id_reference in zip(path, reference):
        node_ids_actual.append(node_id)
        assert node_id == node_id_reference, node_ids_actual

        # Verify path validity
        assert node.attributes["node_id"] == node_id
        is_node_among_siblings = False
        for sibling in siblings:
            if node_id == sibling.attributes["node_id"]:
                is_node_among_siblings = True

        assert is_node_among_siblings, (node_id, node, siblings)


def test_get_walk(reasoning_tree):
    """
    Test getting a walk along a reasoning tree, both
    to the selected node and to the leaf.
    """
    # Root node
    _verify_path(reasoning_tree.get_walk("0"), ["0"])
    _verify_path(reasoning_tree.get_walk_to_leaf("0"), ["0", "1", "3"])

    # Node in between
    _verify_path(reasoning_tree.get_walk("2"), ["0", "2"])
    _verify_path(reasoning_tree.get_walk_to_leaf("2"), ["0", "2", "5"])

    # Leaf node
    _verify_path(reasoning_tree.get_walk("6"), ["0", "2", "6"])
    _verify_path(reasoning_tree.get_walk_to_leaf("6"), ["0", "2", "6"])


def test_add_node_to_root():
    """
    Test adding an initial node to the root node of a
    newly-initialized tree.
    """
    parent_node_id = "0"
    new_node_ids = []
    reasoning_tree = ReasoningTree(
        nodes={"0": get_example_reasoning_node(num_children=2)}
    )

    # Add to a node that has no children yet.
    initial_node_ids = list(reasoning_tree.nodes.keys())
    node = get_example_reasoning_node("1", num_leaf_nodes=1, leaf_value_sum=1)
    node_id = reasoning_tree.add_node(node, parent_node_id)

    assert node_id not in initial_node_ids
    assert node_id in reasoning_tree.nodes.keys()

    # Verify new node_id values are unique
    assert node_id not in new_node_ids
    new_node_ids.append(node_id)

    if node_id in reasoning_tree.edges.keys():
        assert len(reasoning_tree.edges[node_id]) == 0

    assert reasoning_tree.edges[parent_node_id] == [node_id]
    assert reasoning_tree.edges_inverted[node_id] == parent_node_id

    # Verify parent value attributes are correctly updated.
    assert reasoning_tree.nodes[parent_node_id]


def test_add_node(reasoning_tree):
    """
    Test adding a node to the tree.
    """
    # Add to a node with children.
    parent_node_id = "2"
    initial_node_ids = list(reasoning_tree.nodes.keys())
    node = get_example_reasoning_node("8")
    children_original = set(reasoning_tree.edges[parent_node_id])
    node_id = reasoning_tree.add_node(node, parent_node_id)

    assert node_id not in initial_node_ids
    assert node_id in reasoning_tree.nodes.keys()

    # Verify the children is the only newly-added node to the children list.
    assert (node_id not in reasoning_tree.edges.keys()) or len(
        reasoning_tree.edges[node_id]
    ) == 0
    children_new = set(reasoning_tree.edges[parent_node_id])
    assert node_id in children_new
    children_diff = children_new - children_original
    assert len(children_diff) == 1, (children_original, children_new)
    assert node_id in children_diff
    assert reasoning_tree.edges_inverted[node_id] == parent_node_id

    # Verify the newly added node is added to exactly one parent.
    for _parent_node_id, children in reasoning_tree.edges.items():
        if _parent_node_id != parent_node_id:
            assert node_id not in children


def _get_example_forest(forest_name: str) -> List[ReasoningTree]:
    """
    Generate a list of example reasoning trees.
    """
    reasoning_trees = []
    for tree_index in range(2):
        reasoning_tree = get_example_reasoning_tree(tree_index, forest_name)
        reasoning_trees.append(reasoning_tree)

    return reasoning_trees


def _get_num_children_example(tree_node: ReasoningTreeNode) -> int:
    """
    Example implementation of get_num_children.
    """

    print(tree_node.attributes)
    return tree_node.attributes.get("num_children", 0)


class _ExampleTreeUpload:
    """
    Emulate a method for uploading trees.
    """

    def __init__(self):
        self.trees_collected: List[Tuple[ReasoningTreeIdTuple, ReasoningTree]] = []

    def upload_tree(self, tree_id_tuple: ReasoningTreeIdTuple, tree: ReasoningTree):
        """
        Upload tree. (Example)

        Params:
            tree_id_tuple: forest_name, tree_id
            tree: tree to submit.
        """
        self.trees_collected.append((tree_id_tuple, tree))


def get_forest_traversal_manager(
    tree_cache: Optional[Cache[ReasoningTreeIdTuple, ReasoningTree]] = None,
    upload_tree_method: Optional[
        Callable[[ReasoningTreeIdTuple, ReasoningTree], None]
    ] = None,
):
    """
    Instantiate an example forest_traversal_manager.

    Provide optional tree_cache instance or upload_tree method
    to track element fetching and uploading.
    """

    if tree_cache is None:
        example_tree_retrieval_counter = _ExampleTreeRetrieval()
        tree_cache = Cache(
            capacity=2, get_item_uncached=example_tree_retrieval_counter.get_tree
        )

    if upload_tree_method is None:
        example_tree_collector = _ExampleTreeUpload()
        upload_tree_method = example_tree_collector.upload_tree

    forest_traversal_manager = ForestTraversalManager(
        tree_cache,
        _get_example_forest,
        _get_num_children_example,
        upload_tree_method,
    )

    return forest_traversal_manager


def test_forest_traversal_manager_retrieve():
    """
    Verify number of elements returned and unique elements returned.
    """
    output_tree_names: List[int] = []
    output_node_names: List[str] = []
    output_node_id_tuples: List[NodeTuple] = []

    job_name = "job_20231116a1"
    num_elements = 2 * 3 * 2 + 2 * 1 * 1

    forest_traversal_manager = get_forest_traversal_manager()

    for _ in range(num_elements):
        output = forest_traversal_manager.get_node_read_only("example_forest", job_name)
        traversal_coordinator = (
            forest_traversal_manager.forest_traversal_coordinators.get(
                ("example_forest", job_name)
            )
        )

        assert traversal_coordinator is not None
        print(
            "max_presentations: {}".format(
                traversal_coordinator.max_presentations,
            )
        )
        assert output is not None

        tree_id, tree_attributes, path = output

        node_id, node_selected, siblings = path[-1]
        output_tree_names.append(tree_id)
        output_node_id_tuples.append((tree_id, node_id))

    output = forest_traversal_manager.get_node_read_only("example_forest", job_name)
    assert output is None

    # _get_num_children_example sets the num_children of node 0 ~ 2 to 2,
    # node 6 to 1, and otherwise 0.
    assert len(output_tree_names) == num_elements
    assert len(output_node_id_tuples) == num_elements

    for tree_index in [0, 1]:
        assert output_tree_names.count(tree_index) == 2 * 3 + 1 * 1
        for node_index in [0, 1, 2]:
            assert output_node_id_tuples.count((tree_index, str(node_index))) == 2

        for node_index in [6]:
            assert output_node_id_tuples.count((tree_index, str(node_index))) == 1


def test_get_next_node_id():
    """
    Verify get_next_node_id correctly returns an available node id.
    """
    node_ids = ["0", "1", "", "example", "101"]
    node_id = get_next_node_id(node_ids)
    assert node_id not in node_ids
    node_ids.append(node_id)

    node_id = get_next_node_id(node_ids)
    assert node_id not in node_ids


def test_forest_traversal_manager_add():
    """
    Test adding node via forest traversal manager.
    """
    forest_name = "example_forest"

    example_tree_retrieval_counter = _ExampleTreeRetrieval()
    tree_cache = Cache(
        capacity=2, get_item_uncached=example_tree_retrieval_counter.get_tree
    )

    example_tree_collector = _ExampleTreeUpload()
    forest_traversal_manager = get_forest_traversal_manager(
        tree_cache, upload_tree_method=example_tree_collector.upload_tree
    )

    # Retrieve example parent node to add to.
    output = forest_traversal_manager.get_node_read_write(forest_name)
    assert output is not None

    selected_tree_id, tree_attributes, path = output

    parent_node_id, parent_node, siblings = path[-1]
    num_leaf_nodes_initial = _extract_attributes(path, "num_leaf_nodes")
    leaf_value_sum_initial = _extract_attributes(path, "leaf_value_sum")
    new_leaf_node = get_example_reasoning_node(
        "7", num_children=0, leaf_value_sum=0, num_leaf_nodes=1
    )

    # Default number of children: 0
    new_node_ids: List[NodeId] = forest_traversal_manager.add_nodes(
        forest_name, selected_tree_id, parent_node_id, [new_leaf_node]
    )
    assert len(new_node_ids) == 1
    new_node_id = new_node_ids[0]

    assert len(example_tree_collector.trees_collected) == 1
    (forest_name, tree_id_retrieved), tree = example_tree_collector.trees_collected[0]
    assert selected_tree_id == tree_id_retrieved

    assert new_node_id in tree.edges[parent_node_id]
    assert tree.edges_inverted[new_node_id] == parent_node_id

    for _node_id, children in tree.edges.items():
        if _node_id != parent_node_id:
            assert new_node_id not in children

    new_node_many_children = get_example_reasoning_node("8", num_children=12)

    # Example leaf node.
    new_leaf_node_longer_path = get_example_reasoning_node(
        "9", num_children=0, num_leaf_nodes=1, leaf_value_sum=1
    )

    # Add new_leaf_node_longer_path as a child of new_node_many_children
    new_node_ids = forest_traversal_manager.add_nodes(
        forest_name,
        selected_tree_id,
        parent_node_id,
        [new_node_many_children, new_leaf_node_longer_path],
    )
    assert len(new_node_ids) == 2
    node_id_many_children, node_id_no_children = new_node_ids

    # Same tree should be reused from cache.
    assert example_tree_retrieval_counter.num_invokations == 1

    # Verify nodes are added to traversal manager properly.
    nodes_retrieved = Counter()
    nodes_retrieved_other_tree = Counter()
    for _ in range(125):
        output = forest_traversal_manager.get_node_read_write(forest_name)
        if output is None:
            break

        tree_id, tree_attributes, path = output
        top_node_id, top_node, siblings = path[-1]

        if tree_id == selected_tree_id:
            nodes_retrieved[top_node_id] += 1
        else:
            nodes_retrieved_other_tree[top_node_id] += 1

    # Ensure differences are only in:
    # Node that was initial returned
    # Nodes that were added during the test.
    assert nodes_retrieved - nodes_retrieved_other_tree == Counter(
        # max 12 children, less one for new_leaf_node_longer_path,
        # which is added as a child of this node through add_nodes
        {node_id_many_children: 12 - 1}
    )
    assert nodes_retrieved_other_tree - nodes_retrieved == Counter(
        {
            parent_node_id: 1,
            new_node_id: 0,
            node_id_no_children: 0,
        }
    )

    # Verify that the value estimate attributes of all nodes
    # along the path to node_id_no_children are updated.
    updated_tree = forest_traversal_manager.reasoning_tree_cache.get(
        (forest_name, selected_tree_id)
    )
    assert updated_tree is not None
    updated_path = updated_tree.get_walk(parent_node_id)
    num_leaf_nodes_values = _extract_attributes(updated_path, "num_leaf_nodes")
    leaf_value_sum_values = _extract_attributes(updated_path, "leaf_value_sum")

    for path_element, num_leaf_nodes_value, num_leaf_nodes_initial in zip(
        path, num_leaf_nodes_values, num_leaf_nodes_initial
    ):
        assert num_leaf_nodes_value - num_leaf_nodes_initial == 2, path_element

    for path_element, leaf_value_sum_value, leaf_value_sum_initial in zip(
        path, leaf_value_sum_values, leaf_value_sum_initial
    ):
        assert (
            leaf_value_sum_value - leaf_value_sum_initial
            == new_leaf_node_longer_path.attributes["leaf_value_sum"]
            + new_leaf_node.attributes["leaf_value_sum"]
        ), path_element


def test_get_node_value_estimate():
    confidence_interval = get_value_estimate(num_leaf_nodes=3, leaf_value_sum=2)
    assert confidence_interval is not None
    lower_bound = confidence_interval.lower_bound
    upper_bound = confidence_interval.upper_bound
    interval_width = upper_bound - lower_bound
    assert (interval_width > 0) and (not np.allclose(interval_width, 0))
    assert np.allclose(confidence_interval.mean, (2 / 3))
    assert np.allclose((1 / 2) * (lower_bound + upper_bound), (2 / 3))

    example_node = ReasoningTreeNode({}, {})
    assert example_node.value_estimate is None

    example_node.attributes["num_leaf_nodes"] = 3
    example_node.attributes["leaf_value_sum"] = 2
    assert example_node.value_estimate == confidence_interval
