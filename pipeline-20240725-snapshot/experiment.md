# Generating Preference Dataset

- Initialize Forest
- Populate Forest
- Create Synthetic Preferences
- Export Preference Dataset

## Initialize Forest

### Forest Configurations

Create a branch in the submodule `prm_pipeline/_experiment_configs`.

E.g., `liar-new-summarized/with-summaries`

Create rollout and dataset configs under `prm_pipeline/_experiment_configs/rollout/`.
Use only a-z and underscore `_` for the filename since this file needs to be imported as a Python module.

Your config should be a class called `Config` and should sub-class "RolloutConfig" from `prm_pipeline._experiment_configs.interfaces`. Implement the following methods for your dataset:

- get_dataset_iterator
- get_root_prompt
- get_label

Additional methods are useful for visualization and for your own reference. Consider implementing them as well:

- serialize_dataset_item

Modify rollout_worker_config accordingly.

Link to [example](https://github.com/<REDACTED>/_experiment_configs/blob/51a69959b978c7cac7c024c71ba76b4d8899aa01/rollout/gsm8k_baseline.py).

Use the following utility to verify your configuration on an example. Be sure to replace config_name with the name of your module.

```bash
python3 -m prm_pipeline._experiment_configs.utils.verify_rollout_config \
--config_name your_module_name \
--dataset_split test
```

### Seeding the forest

To copy items from your dataset to the forest DB using your configurations, run the following command from a machine with direct access to the Mongo database.

```bash
source .env && python3 -m prm_pipeline.rollout_generator.seed_forest \
--dataset_config_name your_module_name \
--dataset_split test \
--forest_name_suffix optional_but_recommended
```

In the command, `dataset_config_name` should be the same as the file name part (without the `.py` file name extension) of your experiment config.

In the output of the command, look for a string called "forest_name" as you will need this value to retrieve your forest in subsequent steps.

### Inspecting the root nodes

Visit (controller_url)/forest_name to explore the newly-created forest.

By default, the text fields are renderred with markdown. Be sure to read a number of examples carefully.

## Populating the Forest

On each worker node, launch the following to populate your forest:

- vLLM or FastCaht OpenAI-compatible inference backend.
- rollout_worker.

After spinning up the vLLM worker, update `.env` to provide OPENAI-related configs.

Spin up the rollout worker with the following command:

```bash
source .env && python3 -m prm_pipeline.rollout_generator.worker \
--num_processes 12 \
--forest_name name_of_your_forest_from_previous_step
```

You can keep track of the progress by visiting (controller_url)/forest_name. You might need to refresh the page to retrieve the most recent rollouts.

The rollout worker instances need to be stopped manually.

### Obtaining Preliminary Metrics

On a server with direct access to the Mongo database, launch the following to get an estimate of the accuracy assuming that the value estimates are exact. The script below will compute _exact_ value for each node using the ground_truth labels. During the beam search, the utility will keep the top N beams with the highest overall value.

The number of beams and the aggregation strategy for getting the overall value of each beam are command-line arguments.

```bash
source .env && \
python3 -m prm_pipeline.metrics.scripts.beam_search \
--num_beam_values 1,5 \
--value_function_option labelled \
--forest_name name_of_your_forest_from_previous_step
```