from typing import List

import pytest
from flask.testing import FlaskClient

from prm_pipeline.utils import ActionsConfig
from prm_pipeline.utils.tests.test_tree_utils import get_example_reasoning_node
from prm_pipeline.utils.tree_utils import ReasoningTreeNode
from prm_pipeline.webserver.db_utils import MongoReasoningTreeDB
from prm_pipeline.webserver.serialization_utils import RolloutSubmission, RolloutTask
from prm_pipeline.webserver.tests.fixtures import app
from prm_pipeline.webserver.tests.test_db_utils import (
    test_reasoning_tree_db_upload_retrieve_tree as _populate_reasoning_tree_db,
)


@pytest.fixture()
def client(app) -> FlaskClient:
    return app.test_client()


def add_rollouts(client, num_requests: int = 2):
    """
    model_identifier: `model: str`
    example prompt from dataset: `example: str`
    step-by-step reasoning: `reasoning: list of str`
    prediction of class label: `prediction: int`
    ground truth class label: `ground_truth: int`
    worker identifier: `worker: str`
    """
    for index in range(num_requests):
        rollout_payload = {
            "dataset_name": "example_dataset",
            "model": "vicuna-7b-v1.5",
            "example": "Example {} from dataset".format(index),
            "reasoning": ["Step 1", "Step 2"],
            "prediction": 0,
            "ground_truth": 0,
            "worker": "example-hostname/20231018/085001",
        }
        response = client.post("/rollout", json=rollout_payload)
        assert response.status_code < 300

    print("Added {} rollouts".format(num_requests))


def test_add_rollout(client):
    add_rollouts(client, num_requests=2)


def retrieve_rollouts(client, num_retrievals: int, batch_size: int) -> List[int]:
    """
    Retrieve rollouts num_retrievals times.
    Return rollouts_id_retrieved.
    """
    rollouts_id_retrieved = []
    for _ in range(num_retrievals):
        response = client.get(
            "/rollouts?prm_version={}&batch_size={}".format(
                "example_labelling_scheme", batch_size
            )
        )
        assert response.status_code < 300
        assert isinstance(response.json, list)

        for rollout in response.json:
            rollout_id = rollout["_id"]
            assert rollout_id not in rollouts_id_retrieved
            rollouts_id_retrieved.append(rollout_id)

    return rollouts_id_retrieved


@pytest.mark.parametrize("batch_size", [1, 8])
def test_get_rollouts(
    client,
    batch_size: int,
    num_example_rollouts: int = 12,
    num_additional_rollouts: int = 6,
):
    """
    Test retrieving rollouts, making sure the returned values
    do not contain duplicates.
    """
    add_rollouts(client, num_requests=num_example_rollouts)

    rollouts_id_retrieved = retrieve_rollouts(
        client, num_example_rollouts + 5, batch_size
    )
    assert len(set(rollouts_id_retrieved)) == num_example_rollouts

    # Verify that when new rollouts are added, these are properly retrieved.
    add_rollouts(client, num_requests=num_additional_rollouts)

    rollouts_id_retrieved += retrieve_rollouts(
        client, num_additional_rollouts + 2, batch_size
    )

    assert (
        len(set(rollouts_id_retrieved))
        == num_example_rollouts + num_additional_rollouts
    )


def test_add_preference(client, num_example_rollouts: int = 12):
    """
    Test adding AI preference to the database,
    one for each example rollout.

    Batch size is 1 by default (vLLM handles collation.)
    """
    example_prm_version = "example_labelling_scheme"

    # Create example rollouts.
    add_rollouts(client, num_requests=num_example_rollouts)

    """
    - rollout identifier: `rollout_id: int`
    - step-by-step rollout reward predictions: `prm_output: List[float]`
    - prm version identifier: `prm_version: str`
    - worker identifier: `worker: str`
    """
    for index in range(num_example_rollouts):
        rollout_query_response = client.get(
            "/rollouts?prm_version={}&batch_size={}".format(example_prm_version, 1)
        )

        assert isinstance(rollout_query_response.json, list)
        assert len(rollout_query_response.json) == 1
        rollout_id = rollout_query_response.json[0]["id"]

        example_predictions = [index, rollout_id, index + rollout_id / 1000]

        data = {
            "rollout": rollout_query_response.json[0],
            "rollout_id": rollout_id,
            "prm_output": example_predictions,
            "prm_version": example_prm_version,
            "worker_identifier": "example_worker/20231018-125001",
        }
        response = client.post("/process_reward_label", json=data)
        assert response.status_code < 300


def retrieve_process_reward_labels(
    client, num_retrievals: int, batch_size: int
) -> List[int]:
    """
    Retrieve process_reward_labels num_retrievals times.
    Return prm_label_ids_retrieved.
    """
    prm_label_ids_retrieved = []
    for _ in range(num_retrievals):
        response = client.get(
            "/process_reward_labels?rollout.dataset_name={}&batch_size={}".format(
                "example_dataset", batch_size
            )
        )
        assert response.status_code < 300
        assert isinstance(response.json, list)
        assert len(response.json) <= batch_size

        for prm_label in response.json:
            prm_label_ids_retrieved.append(prm_label["id"])

    return prm_label_ids_retrieved


@pytest.mark.parametrize("batch_size", [1, 9])
def test_get_preferences(client, batch_size: int, num_examples: int = 12):
    """
    Test retrieving process reward preferences from the webserver.
    """

    test_add_preference(client, num_examples)

    prm_label_ids_retrieved = retrieve_process_reward_labels(
        client, num_examples + 5, batch_size
    )
    assert len(set(prm_label_ids_retrieved)) == num_examples


def test_rollout_task_request(app, client):
    """
    Test requesting rollout task from webserver.
    """
    # Reuse logic from db tests to populate forest.
    reasoning_tree_db: MongoReasoningTreeDB = app.config["REASONING_TREE_DB"]
    _populate_reasoning_tree_db(reasoning_tree_db)
    assert len(reasoning_tree_db.cache.cache.items()) == 3

    # Verify more than one paths are retrieved.
    tree_ids = []
    top_node_id_list = []
    for _ in range(12):
        response = client.get("/rollout_task?forest_name={}".format("example_forest"))

        if len(response.json.keys()) == 0:
            break

        rollout_task = RolloutTask(**response.json)
        rollout_context = rollout_task.rollout_context
        tree_id = rollout_task.tree_id
        assert tree_id in [0, 2]

        assert isinstance(rollout_context[-1], dict)
        tree_ids.append(tree_id)

        action_config = ActionsConfig.from_dict(
            rollout_task.tree_attributes["actions_config"]
        )
        assert len(action_config.actions) == 1
        assert action_config.actions[0].name == "search"

        # Note that node_id is unique within each tree, but not across
        # trees in the same forest.
        top_node_id = (tree_id, rollout_context[-1]["node_id"])
        assert top_node_id not in top_node_id_list
        top_node_id_list.append((tree_id, top_node_id))

    assert len(tree_ids) == 5 + 1
    assert len(set(tree_ids)) == 2
    assert len(set(top_node_id_list)) == 2


def test_rollout_task_submit(app, client):
    """
    Test submitting new reasoning steps to webserver.
    """
    # Reuse logic from db tests to populate forest.
    reasoning_tree_db: MongoReasoningTreeDB = app.config["REASONING_TREE_DB"]
    _populate_reasoning_tree_db(reasoning_tree_db)
    assert len(reasoning_tree_db.cache.cache.items()) == 3

    forest_name = "example_forest"
    response = client.get("/rollout_task?forest_name={}".format(forest_name))

    rollout_task = RolloutTask(**response.json)

    num_children = 1
    new_node_parent = get_example_reasoning_node(num_children=num_children)
    new_node_leaf = get_example_reasoning_node(num_children=0, leaf_value_sum=1)
    payload = RolloutSubmission(
        tree_id=rollout_task.tree_id,
        ancestor_node_id=rollout_task.parent_node_id,
        nodes=[new_node_parent, new_node_leaf],
    ).to_dict()

    response = client.post(
        "/rollout_task?forest_name={}".format(forest_name), json=payload
    )
    new_node_ids = response.json["node_ids"]

    # Verify the changes are submitted to
    # both reasoning_tree_db and backing store
    tree_id_tuple = (forest_name, rollout_task.tree_id)
    for tree in [
        reasoning_tree_db.cache.get(tree_id_tuple),
        reasoning_tree_db.cache._get_item_uncached(tree_id_tuple),
    ]:
        assert tree is not None

        for new_node_id in new_node_ids:
            assert new_node_id in tree.nodes.keys()

        # Parent
        assert tree.nodes[new_node_ids[0]].attributes["num_children"] == num_children

        # Leaf
        assert tree.nodes[new_node_ids[1]].attributes["num_children"] == 0


def test_enumerate_and_retrieve_trees(app, client):
    """
    Test retrieving (forest_name, tree_id) pairs.

    Test retrieving a particular tree given one such pair.
    """
    # Reuse logic from db tests to populate forest.
    reasoning_tree_db: MongoReasoningTreeDB = app.config["REASONING_TREE_DB"]
    _populate_reasoning_tree_db(reasoning_tree_db)

    # Ensure previous tests did not interfere with the database state
    # of this run.
    assert len(reasoning_tree_db.cache.cache.items()) == 3

    forest_names_response = client.get("/forests")
    forest_names_with_example = forest_names_response.json
    assert isinstance(forest_names_with_example, list)

    tree_id_tuples = []

    # Enumerate trees in each forest separately
    for forest_name_example in forest_names_with_example:
        forest_name = forest_name_example["forest_name"]
        enumerate_tree_response = client.get(
            "/rollout_trees?forest_name={}".format(forest_name)
        )
        assert enumerate_tree_response.is_json

        tree_id_tuples.extend(
            [
                (tree_dict["forest"], tree_dict["id"])
                for tree_dict in enumerate_tree_response.json
            ]
        )

    assert len(set(tree_id_tuples)) == len(reasoning_tree_db.cache.cache.items())
    for key in reasoning_tree_db.cache.cache.keys():
        assert key in tree_id_tuples

    tree_id_tuple = tree_id_tuples[0]
    forest_name, tree_id = tree_id_tuple
    retrieve_tree_response = client.get(
        "/rollout_tree?forest_name={}&tree_id={}".format(forest_name, tree_id)
    )
    assert retrieve_tree_response.status_code == 200

    retrieve_filtered_trees_response = client.get(
        "/rollout_trees?forest_name={}&tree_id={}".format(forest_name, tree_id)
    )
    assert retrieve_tree_response.status_code == 200
    retrieved_filtered_trees = retrieve_filtered_trees_response.json
    assert len(retrieved_filtered_trees) == 1
    assert retrieved_filtered_trees[0]["tree_attributes"]["tree_id"] == tree_id
    assert retrieved_filtered_trees[0]["tree_attributes"]["forest_name"] == forest_name

    tree_reference = reasoning_tree_db.cache.get((forest_name, tree_id))
    assert tree_reference is not None
    assert tree_reference.attributes == retrieve_tree_response.json["attributes"]

    high_altitude_branches_response = client.get(
        "/tree_utils/get_long_branches?"
        "forest_name={}&tree_id={}&min_num_nodes=0".format(forest_name, tree_id)
    )
    top_node_ids = []
    for high_altitude_branch in high_altitude_branches_response.json:
        # De-serialize using reference method for consistency.
        top_node = ReasoningTreeNode.from_dict(high_altitude_branch[-1])
        top_node_id = top_node.attributes["node_id"]
        top_node_ids.append(top_node_id)

        # Verify only top (leaf) nodes are returned.
        children_ids = tree_reference.edges.get(top_node_id, [])
        assert len(children_ids) == 0

    assert set(top_node_ids) == set(["3", "4", "5", "6", "7"])


def test_get_walk_to_leaf(app, client):
    """
    Test retrieving a walk given (forest_name, tree_id, node_id).
    """
    # Reuse logic from db tests to populate forest.
    reasoning_tree_db: MongoReasoningTreeDB = app.config["REASONING_TREE_DB"]
    _populate_reasoning_tree_db(reasoning_tree_db)

    # Ensure previous tests did not interfere with the database state
    # of this run.
    assert len(reasoning_tree_db.cache.cache.items()) == 3

    enumerate_tree_response = client.get("/rollout_trees?forest_name=example_forest")
    assert enumerate_tree_response.is_json

    tree_id_tuples = [
        (tree_dict["forest"], tree_dict["id"])
        for tree_dict in enumerate_tree_response.json
    ]
    tree_id_tuple = tree_id_tuples[0]

    # Root node
    node_id = "0"
    forest_name, tree_id = tree_id_tuple
    path_response = client.get(
        "/rollout_trees/path"
        "?forest_name={}&tree_id={}&node_id={}".format(forest_name, tree_id, node_id)
    )
    assert path_response.status_code == 200

    path_serialized = path_response.json

    # TODO: de-serialize using method from PathElement.
    node_ids = []
    num_siblings_list = []
    sibling_ids_list = []
    for path_element in path_serialized:
        node_ids.append(path_element["node_id"])
        num_siblings = len(path_element["siblings"])
        num_siblings_list.append(num_siblings)
        sibling_ids_list.append(
            [sibling["attributes"]["node_id"] for sibling in path_element["siblings"]]
        )

    assert min(num_siblings_list) == 1
    assert max(num_siblings_list) > 1
