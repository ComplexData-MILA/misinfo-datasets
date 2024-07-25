"""
Logic for computing:
- average accuracy of reasoning traces for each ground_truth category.

Example:
```bash
python3 \
    -m prm_pipeline.metrics.scripts.unweighted \
    --model_identifier=lmsys/vicuna-13b-v1.5/example-20231018a2 \
    --max_rollouts=21000
```
"""
import argparse
import logging
import os
from typing import Any, Dict

import pandas as pd
import pymongo
from tqdm.auto import tqdm

from ..utils.db_utils import retrieve
from ..utils.metrics_utils import get_categories, get_category_metrics

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
parser.add_argument("--model_identifier", required=True)
parser.add_argument("--dataset_name", default="liar")
parser.add_argument("--max_rollouts", default=-1, type=int)
parser.add_argument("--cache_path", default="cache/metrics")
parser.add_argument("--reset_cache", default=False, type=bool)
args = parser.parse_args()

client = pymongo.MongoClient(os.environ.get("MONGO_DB_SERVER"))
db = client[os.environ["MONGO_DB_NAME"]]
query = {
    "model_identifier": args.model_identifier,
    "dataset_name": args.dataset_name,
}
logger.info("query: {}".format(query))

# Retrieve rollouts with caching enabled
rollouts = retrieve(
    db, "rollout", query, args.cache_path, args.reset_cache, args.max_rollouts
)
logger.info("Retrieved rollouts. Number of elements: {}".format(len(rollouts)))

"""
Average reasoning accuracy 
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
        rollouts, None, category, [correct_prediction], [-1, None]
    )

    # Include the user-friendly name for this category in the metrics dictionary.
    metrics["category"] = "{} ({})".format(friendly_name, correct_prediction)

    # Add metrics for this category to the metrics list.
    all_metrics.append(metrics)

# Store metrics as a Pandas table for prettier formatting.
all_metrics_df = pd.DataFrame(all_metrics).set_index("category")
print(all_metrics_df.to_markdown())

# Report average accuracy over all categories.
print(all_metrics_df.mean())
