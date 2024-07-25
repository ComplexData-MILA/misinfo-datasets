"""
Evaluate quality of the value estimation.

- Retrieve all trees in forest.
- Obtain popular vote from get_tree_beam_search_prediction.
- Evaluate whether popular vote is correct.
- Report average accuracy.

Example:
```bash
source .env  # load MongoDB configs
python3 \
 -m prm_pipeline.metrics.scripts.beam_search \
 --forest_name=example_forest-123ac0789 \
 --num_beam_values=1,2,5,10,15
```
"""
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from pymongo import MongoClient
from scipy.stats import t as student_t

from ...utils.enums import ValueFunctionOptions
from ...utils.tree_utils import Prediction
from ...webserver.app import get_db_parameters
from ...webserver.db_utils import MongoReasoningTreeDB
from ..utils.metrics_utils import (
    get_tree_beam_search_prediction,
    value_function_labelled,
)

mongo_server, db_name = get_db_parameters(testing=False)
client = MongoClient(mongo_server)
db = client[db_name]

parser = argparse.ArgumentParser()
parser.add_argument("--forest_name", required=True)
parser.add_argument("--num_beam_values", default="1,2,5,10,15")
parser.add_argument(
    "--value_function_option",
    required=True,
    choices=[choice.value for choice in ValueFunctionOptions],
)
args = parser.parse_args()
forest_name = args.forest_name
num_beam_values = [int(num_beams) for num_beams in args.num_beam_values.split(",")]
value_function_option = ValueFunctionOptions(args.value_function_option)

if value_function_option is ValueFunctionOptions.LABELLED:
    value_function = value_function_labelled
else:
    value_function = lambda _: 0.0

reasoning_tree_db = MongoReasoningTreeDB(db, -1)
trees = reasoning_tree_db.get_forest(forest_name)
print("Retrieved trees; len(trees): {}".format(len(trees)))

# Map num_beam to a list of popular predictions,
# one for each tree in the forest.
predictions: Dict[int, List[Prediction]] = defaultdict(list)

# Map num_beams to the accuracy ratio.
prediction_accuracy: Dict[int, str] = {}

for num_beams in num_beam_values:
    accuracy_list: List[bool] = []
    for tree in trees:
        beam_search_output = get_tree_beam_search_prediction(
            tree, value_function, num_beams
        )

        prediction = beam_search_output.popular_prediction
        ground_truth = tree.attributes["ground_truth"]

        if prediction is not None:
            predictions[num_beams].append(prediction)
            accuracy_list.append(prediction == ground_truth)

    sample_means = np.asarray(accuracy_list)
    avg_sample_mean = np.mean(sample_means)

    # standard error of average sample mean
    sem = np.std(sample_means, ddof=1)
    t_score = student_t.ppf(0.975, df=len(sample_means) - 1)

    prediction_accuracy[num_beams] = "{:.1f}% ({:.7f}, {:.7f})".format(
        avg_sample_mean.item() * 100,
        (avg_sample_mean - t_score * sem).item(),
        (avg_sample_mean + t_score * sem).item(),
    )

print("prediction_accuracy:", prediction_accuracy)
