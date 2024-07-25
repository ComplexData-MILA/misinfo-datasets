"""
Report confidence interval of prediction accuracies (sample means)
of all reasoning trees in a forest.

If a forest include N trees, let X_1 ... X_N represent whether
the prediction for each tree is correct- a Bernoulli random variable.

In a particular tree, whether each prediction (a leaf node) is correct 
represents an independent observation of whether that tree is correct.

Assumptions: 
- X_1 ... X_N are i.i.d.
- leaf nodes in a given tree are independent from one another.

Overview:
- Retrieve all N trees in the forest.
- Obtain "sample mean" for each tree given its leaf nodes.
- Apply the Central Limit Theorem to obtain the confidence interval
    for the average value of the sample means.

Example:
```bash
source .env  # load MongoDB configs
python3 \
 -m prm_pipeline.metrics.scripts.accuracy_clt \
 --forest_name example_forest-123ac0789
```
"""

import argparse
from collections import Counter
import re
from typing import Any, List, Optional

import numpy as np
from pymongo import MongoClient
from tqdm.auto import tqdm

from ...webserver.app import get_db_parameters
from ...webserver.db_utils import MongoReasoningTreeDB
from ..utils.metrics_utils import (
    get_tree_accuracy,
    get_prediction_folded,
    get_f1_safe,
    get_confidence_interval,
)

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
print("Retrieved trees; len(trees): {}".format(len(trees)))

sample_means = np.zeros((num_folds,))
f1_scores: List[Optional[float]] = []
num_examples = 0
num_examples_parsed = 0
model_counter = Counter()

for fold_index in range(num_folds):
    predictions = []
    labels = []
    sample_means_fold: List[float] = []
    for reasoning_tree in tqdm(trees, ncols=75):
        tree_accuracy: Optional[float] = get_tree_accuracy(
            reasoning_tree, fold_index, num_folds
        )
        prediction, model_name = get_prediction_folded(reasoning_tree, fold_index)
        label: Optional[Any] = reasoning_tree.attributes.get("ground_truth")
        model_counter[model_name] += 1

        num_examples += 1
        predictions.append(prediction)
        labels.append(label)

        if tree_accuracy is not None:
            sample_means_fold.append(tree_accuracy)
            num_examples_parsed += 1
        else:
            sample_means_fold.append(0.0)

    sample_means[fold_index] = np.asarray(sample_means_fold).mean().item()
    f1_score = get_f1_safe(labels, predictions)
    f1_scores.append(f1_score)


# the confidence interval is for the mean of len(trees) means.

print(
    "Forest {}\n"
    "Accuracy: {}\n"
    "based on {}/{} examples.".format(
        forest_name,
        get_confidence_interval(sample_means),
        num_examples_parsed,
        num_examples,
    )
)


if all([f1_score is not None for f1_score in f1_scores]):
    f1_scores_array = np.asarray(f1_scores)
    f1_score_confidence_interval: str = get_confidence_interval(f1_scores_array)
    print("F1: {}".format(get_confidence_interval(f1_scores_array)))
else:
    print("F1 score is unavailable.")

print("Models: {}".format(model_counter))