# Automated Evaluation of Misinfo Dataset

## Install Dependencies

Install astral-uv: [Installation methods](https://docs.astral.sh/uv/getting-started/installation/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Environment Variables

For automated evaluation, set up OpenAI or OpenAI-compatible LLM API.

```bash
# .env
# export OPENAI_API_BASE=...
export OPENAI_API_KEY="sk-"
```

## Run Evaluation

```bash
source .env && \
uv run -m misinfo_data_eval.entrypoint \
--source_dataset_path hf://ComplexDataLab/Misinfo_Datasets@ce06269:liar_new:test \
--evaluator_model_name gpt-4o-mini-2024-07-18 \
--max_concurrency 32 \
--limit 72
```

Example output:

```
Loading from HF hub: ComplexDataLab/Misinfo_Datasets
Revision: ce06269
Name of data subset: liar_new
Name of data split: test
len(dataset): 392
100%|██████████████████████████████████████| 72/72 [00:08<00:00,  8.49it/s]
100%|██████████████████████████████████████| 72/72 [00:12<00:00,  5.72it/s]
Evaluating Feasibility: 100%|████████████████| 2/2 [00:21<00:00, 10.54s/it]
Counter({'feasible, requires search': 50, 'not feasible even with search': 18, 'feasible, no search required': 4})
```
