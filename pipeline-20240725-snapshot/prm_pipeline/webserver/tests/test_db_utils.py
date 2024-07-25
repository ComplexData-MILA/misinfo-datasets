from typing import List, Optional

import pytest

# Reuse tree fixtures from tree unit tests.
from prm_pipeline.utils.tests.test_tree_utils import (
    get_example_reasoning_node,
    get_example_reasoning_tree,
)
from prm_pipeline.utils.tests.test_preference_utils import (
    example_timestamp,
    example_preference,
)
from prm_pipeline.utils.preference_utils import Preference
from prm_pipeline.utils.tree_utils import NodeId, ReasoningTree, ReasoningTreeIdTuple
from prm_pipeline.webserver.db_utils import (
    MongoReasoningTreeDB,
    filter_query_dict,
    mongo,
)
from prm_pipeline.webserver.tests.fixtures import app


def test_filter_dict():
    """
    Ensure that filter_dict returns correct filtered values.
    """
    source_dictionary = {
        "key_included": "some_value",
        "key_not_included": "value",
        "key_empty_value": "",
        "key_included_int": "0",
    }
    key_whitelist = set(
        [
            "key_included",
            "key_empty_value",
            "other_key",
            "key_included",
            "key_included_int",
        ]
    )
    int_keys = set(["key_included_int", "other_key_int"])

    filtered_dictionary = filter_query_dict(source_dictionary, key_whitelist, int_keys)
    assert filtered_dictionary == {"key_included": "some_value", "key_included_int": 0}


def retrieve_tree_from_db_direct(
    tree_id_tuple: ReasoningTreeIdTuple, collection
) -> Optional[ReasoningTree]:
    """
    Retrieve tree with the given tree_id_tuple directly
    from DB, returning None if not found.

    Params:
        tree_id_tuple: ReasoningTreeIdTuple,
        collections: MongoDB collections

    Returns:
        ReasoningTree, or None if not found.
    """
    query_filter = MongoReasoningTreeDB._get_tree_query(tree_id_tuple)
    tree_match = collection.find_one(query_filter)

    if tree_match is None:
        return

    tree_retrieved = MongoReasoningTreeDB._extract_tree(tree_match)
    return tree_retrieved


@pytest.fixture
def reasoning_tree_db(app):
    assert app is not None
    assert mongo.db is not None
    return MongoReasoningTreeDB(mongo.db, cache_capacity=128)


def test_reasoning_tree_db_upload_retrieve_tree(
    reasoning_tree_db: MongoReasoningTreeDB,
):
    """
    Test uploading and retrieving a reasoning tree using
    reasoning_tree_db.
    """
    assert mongo.db is not None
    collection = mongo.db[reasoning_tree_db._TREES_MONGO_COLLECTION_NAME]
    assert collection.count_documents({"forest_name": "example_forest"}) == 0

    # Create example trees
    forest_names = ["example_forest"] * 2 + ["some_other_forest"]
    tree_ids = [0, 2, 1]
    num_children_values = [5, 1, 2]

    tree_id_tuples = [
        (forest_name, n) for forest_name, n in zip(forest_names, tree_ids)
    ]
    trees = [
        get_example_reasoning_tree(tree_id, forest_name, num_children_override=0)
        for tree_id, forest_name in zip(tree_ids, forest_names)
    ]

    # Verify there is no such tree in DB yet.
    for tree_id_tuple in tree_id_tuples:
        assert retrieve_tree_from_db_direct(tree_id_tuple, collection) is None

    # Insert tree using method from wrapper
    for tree, tree_id_tuple in zip(trees, tree_id_tuples):
        reasoning_tree_db.upload_tree(tree_id_tuple, tree)

    # Verify tree properties are matched,
    # reading directly from DB and via wrapper
    for tree_id_tuple, tree in zip(tree_id_tuples, trees):
        tree_retrieved_direct = retrieve_tree_from_db_direct(tree_id_tuple, collection)
        assert tree_retrieved_direct is not None
        assert tree.attributes == tree_retrieved_direct.attributes

        tree_retrieved_wrapper = reasoning_tree_db._get_tree(tree_id_tuple)
        assert tree_retrieved_wrapper is not None
        assert tree_retrieved_wrapper.attributes == tree.attributes, tree_id_tuple

        tree_retrieved_cached = reasoning_tree_db.cache.get(tree_id_tuple)
        assert tree_retrieved_cached is not None
        assert tree_retrieved_cached.attributes == tree.attributes

    # Insert a new node into each tree.
    # Upload updated version of each tree.
    reasoning_node_identifiers: List[NodeId] = ["101", "102", "21"]
    reasoning_node_parents: List[NodeId] = ["0", "2", "1"]
    new_nodes = [
        get_example_reasoning_node(node_id, num_children)
        for node_id, num_children in zip(
            reasoning_node_identifiers, num_children_values
        )
    ]

    new_node_ids: List[NodeId] = []
    for tree_id_tuple, parent_id, new_node in zip(
        tree_id_tuples, reasoning_node_parents, new_nodes
    ):
        # Add node to cached instance of the tree.
        # Important: the tree must be retrieved from the cache,
        # and should not be the instance in this unit test.
        tree_retrieved = reasoning_tree_db.cache.get(tree_id_tuple)
        assert tree_retrieved is not None

        new_node_id = tree_retrieved.add_node(new_node, parent_id)
        new_node_ids.append(new_node_id)

        # Submit tree to DB using wrapper.
        reasoning_tree_db.upload_tree(tree_id_tuple, tree_retrieved)

    # Verify retrieved trees are valid.
    for tree_id_tuple, tree, parent_id, new_node_id, new_node in zip(
        tree_id_tuples, trees, reasoning_node_parents, new_node_ids, new_nodes
    ):
        # Retrieve tree directly from DB
        tree_retrieved = reasoning_tree_db.cache.get(tree_id_tuple)
        assert tree_retrieved is not None
        assert new_node_id in tree_retrieved.edges_inverted.keys()
        assert tree_retrieved.edges_inverted[new_node_id] == parent_id
        assert tree_retrieved.nodes[new_node_id].attributes == new_node.attributes

    # Try retrieving all the trees from the "forest"
    trees_retrieved = list(reasoning_tree_db.get_forest(forest_names[0]))
    assert len(trees_retrieved) == 2

    tree_ids_retrieved = []
    # Note that the attribute dict isn't compatible with hashset.
    for tree_retrieved in trees_retrieved:
        tree_ids_retrieved.append(tree_retrieved.attributes["tree_id"])

    assert set(tree_ids_retrieved) == set(tree_ids[:2])


def _verify_preference_equivalence(
    preferences_0: List[Preference],
    preferences_1: List[Preference],
    verify_ids: bool = True,
):
    """
    Verify equivalence between two lists of preferences
    regardless of ordering.
    """
    object_ids_0 = [preference.object_id for preference in preferences_0]
    object_ids_1 = [preference.object_id for preference in preferences_1]

    assert set(object_ids_0) == set(object_ids_1)


def test_mongo_db_upload_retrieve_preference(
    reasoning_tree_db: MongoReasoningTreeDB, example_preference: Preference
):
    """
    Insert preferences into MongoDB and retrieve these instances.
    """
    example_preferences = [
        example_preference._replace(
            object_id="0",
            node_ids=["1", "2"],
            preference_scores={"1": 0.0, "2": 1.0},
            label_source="source_a",
        ),
        example_preference._replace(
            node_ids=["5", "6"],
            preference_scores={"5": 0.0, "6": 1.0},
            label_source="source_a",
        ),
        example_preference._replace(
            node_ids=["1", "2"],
            preference_scores={"1": 0.0, "2": 1.0},
            label_source="source_b",
        ),
    ]

    object_ids = []
    for preference in example_preferences:
        object_id = reasoning_tree_db.upload_preference(preference)
        object_ids.append(object_id)

    # Retrieve preferences and compare with original values.
    source_a_preferences = reasoning_tree_db.get_preferences(
        {"label_source": "source_a"}
    )
    assert len(source_a_preferences) == 2
    assert set([preference.object_id for preference in source_a_preferences]) == set(
        object_ids[:2]
    )

    source_b_preferences = reasoning_tree_db.get_preferences(
        {"label_source": "source_b"}
    )
    assert len(source_b_preferences) == 1
    assert set([preference.object_id for preference in source_b_preferences]) == set(
        object_ids[2:]
    )
