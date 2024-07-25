"""
Instatiates MongoReasoningTreeDB to retrieve trees from a specific run and saves them in json format

source .env  &&
python3 \
 -m prm_pipeline.metrics.scripts.generate_json_from_run \
 --forest_name example_forest-123ac0789
"""

import argparse
import re
import pandas as pd
import os

from pymongo import MongoClient
from typing import Any, Optional

from ...webserver.app import get_db_parameters
from ...webserver.db_utils import MongoReasoningTreeDB
from typing import Any, Dict, List

from ..utils.metrics_utils import (
    get_tree_accuracy,
    get_prediction_folded,
)


def construct_searches_dict(tree) -> List[Dict[str, Any]]:
    leaf_node_ids = tree.get_leaf_node_ids()
    trial_searches = []

    for leaf_node_id in leaf_node_ids:
        path_to_leaf = tree.get_walk(leaf_node_id)
        searches = []
        final_evaluation = tree.nodes[leaf_node_id].attributes.get("prediction", None)

        for path_element in path_to_leaf:
            node_id = path_element.node_id
            node = tree.nodes[node_id]
            action_response = node.attributes.get("action_response")

            if action_response:
                search_query = action_response["query"]
                responses = action_response["response"]

                search_info = {
                    "search_query": search_query,
                    "response": responses["response"],
                }
                searches.append(search_info)

        # Associate final evaluation with the search path
        trial_searches.append({"prediction": final_evaluation, "searches": searches})

    return trial_searches


URL_PATTERN = r".+/([a-z0-9_\-\.]+)/?[^/]+$"
parser = argparse.ArgumentParser()
parser.add_argument("--forest_name", required=True)
parser.add_argument("--num_folds", type=int, default=5)

args = parser.parse_args()


url_match = re.match(URL_PATTERN, args.forest_name)
if url_match is not None:
    forest_name = url_match.group(1)
else:
    forest_name = args.forest_name


num_folds = args.num_folds


mongo_server, db_name = get_db_parameters(testing=False)
client = MongoClient(mongo_server)
db = client[db_name]


reasoning_tree_db = MongoReasoningTreeDB(db, -1)
trees = reasoning_tree_db.get_forest(forest_name)


results = []

for tree in trees:
    liar_new = tree.attributes.get("source")
    label: Optional[Any] = tree.attributes.get("ground_truth")

    predictions = []
    tree_accuracies = []
    search_results = []

    for fold_index in range(num_folds):
        predictions.append(get_prediction_folded(tree, fold_index))
        tree_accuracies.append(get_tree_accuracy(tree, fold_index, num_folds))

    tree_data = {
        **liar_new,
        "ground_truth": tree.attributes.get("ground_truth"),
        "predictions": predictions,
        "average_tree_accuracies": tree_accuracies,
        "search_results": construct_searches_dict(tree),
    }
    results.append(tree_data)

df_results = pd.DataFrame(results)

directory = "data"
file_name = f"{forest_name}.csv"
if not os.path.exists(directory):
    os.makedirs(directory)

print(df_results[["statement", "predictions"]].head())
df_results.to_json(os.path.join(directory, file_name))
