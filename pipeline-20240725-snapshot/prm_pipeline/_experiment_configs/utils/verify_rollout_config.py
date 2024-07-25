"""
Utils for testing a given rollout experiment config.

Usage: assuming that example_config.py is stored under 
prm_pipeline/_experiment_configs/rollouts/
```bash
$ python3 -m \
 prm_pipeline._experiment_configs.utils.verify_rollout_config \
 --config_name=example_config.py \
 --dataset_split=test
```
"""

import argparse
import importlib
import json
import logging
import os

from ..interfaces import RolloutConfig
from ...rollout_generator.seed_utils import initialize_tree_from_dataset_entry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--dataset_split", default="train")
    args = parser.parse_args()

    config_name = args.config_name
    config_filename = config_name + ".py"
    dataset_split = args.dataset_split

    dataset_module = importlib.import_module(
        "prm_pipeline._experiment_configs.rollout." + config_name,
    )
    rollout_config: RolloutConfig = dataset_module.Config

    # Test each method specified in the dataset config interface.
    dataset_iterator = rollout_config.get_dataset_iterator(dataset_split)

    dataset_item = next(iter(dataset_iterator))
    dataset_item_serializable = rollout_config.serialize_dataset_item(dataset_item)
    logger.info(
        "dataset_item_serializable keys:"
        " \n{}\n".format(dict(dataset_item_serializable).keys())
    )
    dataset_item_serialized = json.dumps(dataset_item_serializable)
    logger.info("dataset_item_serialized: \n{}\n".format(dataset_item_serialized))

    root_prompt = rollout_config.get_root_prompt(dataset_item)
    logger.info("root_prompt: \n{}\n".format(root_prompt))

    label = rollout_config.get_label(dataset_item)
    logger.info("label ({}): \n{}\n".format(type(label), label))

    tree = initialize_tree_from_dataset_entry(dataset_item, rollout_config)
    logger.info("tree.attributes: {}".format(json.dumps(tree.attributes, indent=2)))