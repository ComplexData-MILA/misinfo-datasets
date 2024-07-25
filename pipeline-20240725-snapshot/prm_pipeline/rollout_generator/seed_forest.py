"""
Script for adding root nodes to a forest.

Overview:
- Load dataset.
- Apply prompt template to create base prompt
    for each item in the dataset.
- Instantiate reasoning tree node instance 
    for the root node of each tree.
- Instantiate reasoning tree 
    with root node added and no edges.
- Submit reasoning tree to database.

Parameters:
- forest_name: str, Name of forest
- dataset_name: str, path to HuggingFace dataset
- dataset_split: str, split of dataset to include
- forest_config: str, path to forest prompt config json

Example:
source testing.env && python3 -m \
 prm_pipeline.rollout_generator.seed_forest \
 --dataset_config_name=example-dataset \
 --dataset_split=train
"""

import argparse
import importlib
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import List

from pymongo import MongoClient
from tqdm.auto import tqdm

from .._experiment_configs.interfaces import RolloutConfig
from ..utils.submodule_utils import get_commit_hash, has_uncommitted_changes
from ..webserver.db_utils import MongoReasoningTreeDB
from .seed_utils import initialize_tree_from_dataset_entry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

submodule_path = "prm_pipeline/_experiment_configs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config_name", required=True)
    parser.add_argument("--forest_name_suffix")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_num_workers", type=int, default=16)
    args = parser.parse_args()

    dataset_config_name = args.dataset_config_name
    dataset_config_relative_path = "rollout/{}.py".format(dataset_config_name)

    mongo_client = MongoClient(os.environ["MONGO_DB_SERVER"])
    db = mongo_client[os.environ["MONGO_DB_NAME"]]
    reasoning_tree_db = MongoReasoningTreeDB(db, 1)

    # Experiment provenance tracking:
    # Verify experiment version control status before loading configs.
    if has_uncommitted_changes(submodule_path, dataset_config_relative_path):
        raise RuntimeError(
            "in the submodule {}, please git-commit "
            "pending changes in {} before continuing.".format(
                os.path.join(submodule_path, dataset_config_relative_path),
                submodule_path,
            )
        )

    experiment_commit_hash: str = get_commit_hash(submodule_path)
    dataset_module = importlib.import_module(
        "prm_pipeline._experiment_configs.rollout." + dataset_config_name
    )
    rollout_config: RolloutConfig = dataset_module.Config

    # Name forest based on name from config, config git version, and
    # the suffix supplied when invoking this script.
    forest_name: str = "{}-{}".format(dataset_config_name, experiment_commit_hash)
    if args.forest_name_suffix is not None:
        forest_name += "-" + args.forest_name_suffix
    logger.info("forest_name: {}".format(forest_name))

    if len(reasoning_tree_db.get_forest(forest_name)):
        raise ValueError(
            "{} is already registered in DB. "
            "Please specify a different --forest_name_suffix.".format(forest_name)
        )

    dataset_iterator = rollout_config.get_dataset_iterator(args.dataset_split)

    with ThreadPoolExecutor(max_workers=args.max_num_workers) as executor:
        futures: List[Future] = []
        # Iterate through dataset for each selected split
        for tree_id, dataset_entry in enumerate(
            tqdm(dataset_iterator, ncols=75, desc="Loading data")
        ):
            tree = initialize_tree_from_dataset_entry(
                dict(dataset_entry), rollout_config
            )
            tree.attributes["forest_info"]["dataset_split"] = args.dataset_split

            # tree_id is the same as the ordering of the dataset item
            # in the iterator.
            tree.attributes["tree_id"] = tree_id

            # Submit tree to backing store
            future = executor.submit(
                reasoning_tree_db.upload_tree, (forest_name, tree_id), tree
            )
            futures.append(future)

            if (args.limit is not None) and (tree_id + 1 >= args.limit):
                break

        for future in tqdm(
            as_completed(futures), total=len(futures), ncols=75, desc="Uploading"
        ):
            continue
