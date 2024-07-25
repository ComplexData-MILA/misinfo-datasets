"""
Logic for computing:
- majority vote accuracy with filtering enabled.

Example:
```bash
python3 \
 -m prm_pipeline.metrics.scripts.weighted \
 --prm_version=lmsys/vicuna-13b-v1.5/example-20231016a1 \
 --aggregation="minimum"
```

Refer to prm_pipeline/utils/enums.py for "aggregation" options.
"""
import argparse
import logging
import os
from typing import Any, Dict

import pandas as pd
import pymongo
from tqdm.auto import tqdm

from ...utils import AggregationMethods
from ..utils.db_utils import retrieve
from ..utils.metrics_utils import (
    get_aggregated_preferences,
    get_categories,
    get_category_metrics,
    split_rollout_rewards,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Maps AI "prediction" to dataset label.
LABELS_MAP = {
    0: "false",
    1: "half-true",
    2: "mostly-true",
    3: "true",
    4: "barely-true",
    5: "pants-fire",
}

# Map label to correct prediction label.
CORRECT_PREDICTION_MAP = {
    "false": 0,
    "half-true": 1,
    "mostly-true": 1,
    "true": 1,
    "barely-true": 0,
    "pants-fire": 0,
}

PREDICTION_MAP = {
    -1: "More-Info",
    0: "False",
    1: "True",
}

parser = argparse.ArgumentParser()
parser.add_argument("--prm_version", required=True)
parser.add_argument("--dataset_name")
parser.add_argument("--rollout_model_identifier")
parser.add_argument("--aggregation")
parser.add_argument("--cache_path", default="cache/metrics")
parser.add_argument("--reset_cache", default=False, type=bool)
args = parser.parse_args()

if args.aggregation is not None:
    aggregation = AggregationMethods(args.aggregation)
else:
    aggregation = None

logger.info("Aggregating step-level process_rewards with {}".format(aggregation))

client = pymongo.MongoClient(os.environ.get("MONGO_DB_SERVER"))
db = client[os.environ["MONGO_DB_NAME"]]
query = {
    "prm_version": args.prm_version,
}

if args.dataset_name is not None:
    query["rollout.dataset_name"] = args.dataset_name

if args.rollout_model_identifier is not None:
    query["rollout.model_identifier"] = args.rollout_model_identifier

logger.info("query: {}".format(query))

# Retrieve process labels with caching enabled
process_reward_elements = retrieve(
    db, "process_reward_label", query, args.cache_path, args.reset_cache
)
logger.info(
    "Retrieved process supervision labels. Number of elements: {}".format(
        len(process_reward_elements)
    )
)

# Split process reward elements into rollouts and process reward labels.
rollouts, process_rewards = split_rollout_rewards(process_reward_elements)

# Note that process_rewards is a list of list of floats.
# Each rollout maps to a list of floats, one for each step.
# Apply the selected reduction method to obtain one aggregated
# reward for the entire rollout.
aggregated_preferences = get_aggregated_preferences(process_rewards, aggregation)
assert len(rollouts) == len(process_rewards)

"""
Average reasoning accuracy 
weighted with aggregated reward labels.
- For each ground_truth category
"""

# Retrieve a list of all ground_truth values
# mentioned in the rollouts retrieved.
ground_truth_categories = get_categories(rollouts)

# Initialize an empty list to store metrics for the categories.
# Metrics for each category is stored as a list.
all_metrics = []

# For each category:
for category in tqdm(ground_truth_categories, ncols=75):
    # Look up the user-friendly name of the category (e.g., pants-fire)
    friendly_name = LABELS_MAP.get(category, str(category))

    # Look up label map (True or False) for the category
    correct_prediction = CORRECT_PREDICTION_MAP.get(friendly_name, -1)

    # Calculate metrics for this category in the selected rollouts
    metrics: Dict[str, Any] = get_category_metrics(
        rollouts, aggregated_preferences, category, [correct_prediction], [-1, None]
    )

    # Include the user-friendly name for this category in the metrics dictionary.
    metrics["category"] = "{} ({})".format(friendly_name, correct_prediction)

    # Add metrics for this category to the metrics list.
    all_metrics.append(metrics)

# Store metrics as a Pandas table for prettier formatting.
all_metrics_df = pd.DataFrame(all_metrics).set_index("category").sort_index()
print(all_metrics_df)
all_metrics_df.to_markdown("output.md")

# Report average accuracy over all categories.
print(all_metrics_df.mean())
