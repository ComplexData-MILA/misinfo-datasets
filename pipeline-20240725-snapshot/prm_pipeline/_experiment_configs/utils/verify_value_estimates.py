"""
Utils for testing a given value estimate experiment config.

Usage: assuming that example_config.py is stored under 
prm_pipeline/_experiment_configs/value_estimates/
```bash
$ source .env && python3 -m \
 prm_pipeline._experiment_configs.utils.verify_value_estimates \
 --config_name=example_config \
 --forest_name=example_forest \
 --node_ids=0,1 
```
"""

import argparse
import datetime
import importlib
import json
import logging
import os
from collections import Counter
from typing import AbstractSet, List, Optional

from pymongo import MongoClient

from ...utils import NodeId, Preference
from ...utils.preference_utils import _PairwisePreference, get_pairwise_preferences
from ...webserver.db_utils import MongoReasoningTreeDB
from ..interfaces import ValueEstimateConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--forest_name", required=True)
    parser.add_argument("--reasoning_tree_id", default=0, type=int)
    parser.add_argument("--node_ids")
    args = parser.parse_args()

    config_name = args.config_name
    config_filename = config_name + ".py"
    forest_name = args.forest_name
    reasoning_tree_id = args.reasoning_tree_id

    tree_id_tuple = (forest_name, reasoning_tree_id)

    mongo_server_url = os.environ["MONGO_DB_SERVER"]
    mongo_db_name = os.environ["MONGO_DB_NAME"]

    value_estimate_module = importlib.import_module(
        "prm_pipeline._experiment_configs.value_estimates." + config_name,
    )
    value_estimate_config: ValueEstimateConfig = value_estimate_module.Config

    # Retrieve example reasoning tree from the given forest.
    mongo_client = MongoClient(mongo_server_url)
    assert mongo_client is not None
    database = mongo_client[mongo_db_name]

    reasoning_tree_db = MongoReasoningTreeDB(database, 1)

    reasoning_tree = reasoning_tree_db._get_tree(tree_id_tuple)
    if reasoning_tree is None:
        raise ValueError("reasoning tree {} is not found".format(tree_id_tuple))

    node_ids: List[NodeId] = list(reasoning_tree.nodes.keys())
    if args.node_ids is not None:
        node_ids = args.node_ids.split(",")

    # Process a node only if neither this node nor its siblings
    # has been processed.
    nodes_visited: AbstractSet[NodeId] = set()
    all_preferences: List[Preference] = []
    for node_id in node_ids:
        if node_id in nodes_visited:
            continue

        sibling_ids = reasoning_tree.get_sibling_ids(node_id)
        nodes_visited = nodes_visited.union(sibling_ids)

        path = reasoning_tree.get_walk(node_id)
        preferences = value_estimate_config.get_preferences(path)
        all_preferences.extend(preferences)

    if len(all_preferences) == 0:
        raise ValueError("No preferences was generated for this tree.")

    # Report number of nodes tracked in each preference instance,
    # as well as the number of preferences created from each preference instance.
    num_nodes_statistics = Counter()
    pairwise_preferences_statistics = Counter()
    all_pairwise_preferences: List[_PairwisePreference] = []
    for preference in all_preferences:
        preference = preference._replace(
            forest_name=forest_name,
            tree_id=reasoning_tree_id,
            label_source=value_estimate_config.label_source_prefix,
            timestamp=datetime.datetime.now(),
        )
        num_nodes_tracked = len(preference.node_ids)
        num_nodes_statistics[num_nodes_tracked] += 1

        pairwise_preferences = get_pairwise_preferences(
            preference, value_estimate_config.pairwise_preference_criteria
        )
        all_pairwise_preferences.extend(pairwise_preferences)
        pairwise_preferences_statistics[len(pairwise_preferences)] += 1

    print(json.dumps(preference.to_dict(), indent=2))
    print("len(preferences): {}".format(len(preferences)))
    print("len(all_pairwise_preferences): {}".format(len(all_pairwise_preferences)))
    print("num_nodes_statistics: {}".format(num_nodes_statistics))
    print(
        "pairwise_preferences_statistics "
        "(num_preferences: num_instances): {}".format(pairwise_preferences_statistics)
    )
