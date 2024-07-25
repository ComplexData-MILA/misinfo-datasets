import os
from threading import Lock

import pymongo
from flask import Blueprint, current_app, jsonify, request, send_file

from ..utils import ForestTraversalManager, PathElement
from .db_utils import (
    MongoReasoningTreeDB,
    filter_query_dict,
    get_next_index,
    mongo,
    sequential_retrieve,
)
from .serialization_utils import RolloutSubmission, RolloutTask, serialize_id

bp = Blueprint(
    "main", __name__, static_url_path="/assets", static_folder="../../static/assets"
)

TREE_SUMMARIZATION_SCHEME = {
    "forest": "$forest_name",
    "id": "$tree_id",
    "tree_attributes": "$data.attributes",
    "root_data": "$data.nodes.0.data",
}

# Mongo aggregation specs
# for generating summary info about *one* tree
# in each forest.
FOREST_SUMMARIZATION_PIPELINE = {
    "$group": {
        "_id": "$forest_name",
        "forest_name": {"$first": "$forest_name"},
        "example_tree": {"$first": TREE_SUMMARIZATION_SCHEME},
    }
}

# Mongo aggregation specs
# for generating summary info about *all* trees across each forests
# (be sure to apply a "$match" filter before this one)
# Returns one {_id, trees} instance for each forest.
TREE_ENUMERATION_PIPELINE = {
    "$group": {
        "_id": "$forest_name",
        "tree_summaries": {"$addToSet": TREE_SUMMARIZATION_SCHEME},
    }
}

# id for efficient and paginated retrieval
forest_manager_initialization_lock = Lock()
rollout_cursor_lock = Lock()
prm_label_cursor_lock = Lock()


FILTER_KEYS_BASE = {
    "rollout": ["rollout_model_identifier", "dataset_name"],
    "process_reward_labels": [
        "rollout.dataset_name",
        "rollout.model_identifier",
        "prm_version",
        "template_info.labelling_scheme_version",
    ],
    "preferences": ["forest_name", "label_source", "node_id"],
}

FILTER_KEYS = {**FILTER_KEYS_BASE}
for resource_name, filter_keys in FILTER_KEYS_BASE.items():
    env_key = "WEBSERVER_{}_FILTER_KEYS".format(resource_name.upper())
    env_value = os.environ.get(env_key)
    if env_value is not None:
        FILTER_KEYS[resource_name].extend(env_value.split(","))


def get_forest_traversal_manager(app_config) -> ForestTraversalManager:
    """
    Initialize forest traversal manager with reasoning_tree_db
    from app config if yet not initialized.

    This implementation is thread-safe.

    Params:
        TODO: replace current_app.config references with app_config
        app_config: Flask app_fonfig dictionary.
            REASONING_TREE_DB should have been initialized.

    Returns:
        ForestTraversalManager
    """
    reasoning_tree_db: MongoReasoningTreeDB = current_app.config["REASONING_TREE_DB"]

    with forest_manager_initialization_lock:
        # Note that the rollout workers are responsible
        # of setting num_children before uploading the tree node.
        forest_traversal_manager: ForestTraversalManager = current_app.config.get(
            "ROLLOUT_FOREST_MANAGER",
            ForestTraversalManager(
                reasoning_tree_db.cache,
                reasoning_tree_db.get_forest,
                lambda tree_node: tree_node.attributes.get("num_children", 0),
                reasoning_tree_db.upload_tree,
            ),
        )
        current_app.config["ROLLOUT_FOREST_MANAGER"] = forest_traversal_manager

    return forest_traversal_manager


# Endpoint: Rollout task
@bp.route("/rollout_task", methods=["GET"])
def rollout_task_request():
    forest_traversal_manager = get_forest_traversal_manager(current_app.config)
    forest_name = request.args["forest_name"]

    job_info = forest_traversal_manager.get_node_read_write(forest_name)
    if job_info is None:
        return jsonify({})

    tree_id, tree_attributes, rollout_context = job_info
    rollout_task = RolloutTask(
        tree_id, tree_attributes, PathElement.serialize_path(rollout_context)
    )
    return jsonify(rollout_task._asdict()), 200


@bp.route("/rollout_task", methods=["POST"])
def rollout_task_submit():
    forest_traversal_manager = get_forest_traversal_manager(current_app.config)
    forest_name = request.args["forest_name"]

    payload = request.json
    assert payload is not None

    rollout_submission = RolloutSubmission.from_dict(payload)
    new_node_ids = forest_traversal_manager.add_nodes(
        forest_name,
        rollout_submission.tree_id,
        rollout_submission.ancestor_node_id,
        rollout_submission.nodes,
    )

    response = {"node_ids": new_node_ids}
    return jsonify(response), 200


# Endpoint: Rollout Generator
@bp.route("/rollout", methods=["POST"])
def rollout():
    assert mongo.db is not None
    collection_name = "rollout"
    collection = mongo.db[collection_name]
    collection.create_index([("id", pymongo.ASCENDING)], unique=True)

    data = request.get_json()
    data["id"] = get_next_index(mongo.db, collection_name)

    collection.insert_one(data).inserted_id

    return jsonify({"id": data["id"]}), 200


@bp.route("/rollouts", methods=["GET"])
def get_rollouts():
    """
    If no rollout is available, the response will be an
    empty list. The client is responsible for retrying
    after a certain rate limit interaval.

    Requests with the same prm_version will receive
    different results from the database.
    """
    assert mongo.db is not None

    prm_version = request.args.get("prm_version", "")
    limit = int(request.args.get("batch_size", 8))

    additional_filters = filter_query_dict(
        dict(request.args), set(FILTER_KEYS["rollout"])
    )

    # Fetch rollouts from database, thread-safe.
    rollouts = sequential_retrieve(
        mongo.db,
        "rollout",
        key=prm_version,
        limit=limit,
        lock=rollout_cursor_lock,
        additional_queries=additional_filters,
    )

    rollouts_serializable = list(map(serialize_id, rollouts))
    return jsonify(rollouts_serializable)


@bp.route("/process_reward_label", methods=["POST"])
def process_reward_label():
    """
    Store process reward label in database.
    Create a unique index for this
    """
    assert mongo.db is not None
    collection_name = "process_reward_label"
    collection = mongo.db[collection_name]
    collection.create_index([("id", pymongo.ASCENDING)], unique=True)

    data = request.get_json()
    data["id"] = get_next_index(mongo.db, collection_name)

    collection.insert_one(data).inserted_id

    return jsonify({"id": data["id"]}), 200


@bp.route("/process_reward_labels", methods=["GET"])
def get_process_reward_labels():
    """
    If no process reward label is available, the response
    will be an empty list. The client is responsible for retrying
    after a certain rate limit interaval.
    """
    assert mongo.db is not None
    prm_version = request.args.get("prm_version", "")
    limit = int(request.args.get("batch_size", 8))

    additional_filters = filter_query_dict(
        dict(request.args), key_whitelist=set(FILTER_KEYS["process_reward_labels"])
    )

    # Fetch prm_labels from database, thread-safe.
    process_reward_labels = sequential_retrieve(
        mongo.db,
        "process_reward_label",
        key=prm_version,
        limit=limit,
        lock=prm_label_cursor_lock,
        additional_queries=additional_filters,
    )

    prm_label_serializable = list(map(serialize_id, process_reward_labels))
    return jsonify(prm_label_serializable)


@bp.route("/forests")
def enumerate_forest_names_with_examples():
    """
    Return a list of forest_name,
    with one example for each unique forest_name.
    """
    assert mongo.db is not None

    # Contains ObjectId which needs to be turned into a
    # string before serializing.
    forest_names_with_example_raw = mongo.db.reasoning_trees.aggregate(
        [FOREST_SUMMARIZATION_PIPELINE]
    )
    forest_names_with_example = list(forest_names_with_example_raw)

    return jsonify(forest_names_with_example)


@bp.route("/reasoning_trees")
@bp.route("/rollout_trees")
def enumerate_tree_ids():
    """
    Return a list of (forest_name, tree_id) pairs.
    """

    assert mongo.db is not None

    assert "forest_name" in request.args.keys(), "forest_name is required"

    query = filter_query_dict(
        request.args, set(["forest_name", "tree_id"]), set(["tree_id"])
    )

    # Returns a list of {id: forest_name, tree_summaries: [list of tree summaries]}
    # One for each forest in the result
    tree_id_pairs_raw = mongo.db.reasoning_trees.aggregate(
        [{"$match": query}, TREE_ENUMERATION_PIPELINE]
    )
    tree_summaries = list(tree_id_pairs_raw)

    # Verify that only one forest is selected.
    # (_id would be set to forest_name)
    assert len(tree_summaries) == 1, [summary["_id"] for summary in tree_summaries]

    return jsonify(tree_summaries[0]["tree_summaries"])


def _get_reasoning_tree(request_args):
    """
    Retrieve reasoning tree from database given
    GET parameters specified in request.
    """
    forest_name = request_args.get("forest_name")
    tree_id_str = request_args.get("tree_id")

    if (forest_name is None) or (tree_id_str is None):
        raise ValueError("Both forest_name and tree_id are required.")

    tree_id = int(tree_id_str)
    assert mongo.db is not None

    # Retrieve from forest traversal manager cache since DB might not be
    # up to date.
    forest_traversal_manager = get_forest_traversal_manager(current_app.config)
    tree = forest_traversal_manager.reasoning_tree_cache.get((forest_name, tree_id))

    if tree is None:
        raise KeyError("Tree not found: ({})".format((forest_name, tree_id)))

    return tree


@bp.route("/reasoning_tree")
@bp.route("/rollout_tree")
def retrieve_reasoning_tree():
    """
    Retrieve a particular rollout tree given (forest_name, tree_id).
    """
    tree = _get_reasoning_tree(request.args)
    return jsonify(tree.to_dict())


@bp.route("/reasoning_trees/path")
@bp.route("/rollout_trees/path")
def get_walk_to_leaf():
    """
    Retrieve a path between root and a leaf that goes
    through the given node (forest_name, tree_id, node_id).

    Provides remote access to tree.get_walk_to_leaf.
    """
    tree = _get_reasoning_tree(request.args)
    node_id = request.args.get("node_id")

    if node_id is None:
        raise ValueError("GET parameter `node_id` is required")

    path = tree.get_walk_to_leaf(node_id)
    path_serialized = PathElement.serialize_path(path)

    return jsonify(path_serialized)


@bp.route("/tree_utils/get_long_branches")
def get_long_branches():
    """
    Return a list of paths with least min_num_nodes steps.
    Only paths to leaf nodes will be included.
    """
    min_num_nodes = int(request.args.get("min_num_nodes", 7))
    forest_name = request.args.get("forest_name")
    tree_id_str = request.args.get("tree_id")

    if (forest_name is None) or (tree_id_str is None):
        raise ValueError("Both forest_name and tree_id are required.")

    tree_id = int(tree_id_str)
    assert mongo.db is not None

    # Retrieve from forest traversal manager cache since DB might not be
    # up to date.
    forest_traversal_manager = get_forest_traversal_manager(current_app.config)
    tree = forest_traversal_manager.reasoning_tree_cache.get((forest_name, tree_id))
    if tree is None:
        raise KeyError("Tree not found: ({})".format((forest_name, tree_id)))

    high_altitude_paths = []
    # Include only paths to leaf nodes
    # satisfying the minimum altitude requirement.
    for node_id, altitude in tree.altitude_map.items():
        if tree.is_leaf(node_id) and (altitude >= min_num_nodes):
            path = tree.get_walk(node_id)
            high_altitude_paths.append([element.node.to_dict() for element in path])

    return jsonify(high_altitude_paths)


@bp.route("/preferences")
def get_preferences():
    """
    Returns a list of serialized preferences according to the given query.
    """
    query = filter_query_dict(dict(request.args), set(FILTER_KEYS["rollout"]))
    reasoning_tree_db: MongoReasoningTreeDB = current_app.config["REASONING_TREE_DB"]

    preferences = reasoning_tree_db.get_preferences(query)
    preference_serialized = [preference.to_dict() for preference in preferences]
    return jsonify(preference_serialized)


@bp.route("/", defaults={"path": ""})
@bp.route("/<path:path>")
def index(path):
    return send_file(os.path.realpath("static/index.html"))
