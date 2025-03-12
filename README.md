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

## Run LLM Feasibility Evaluation

```bash
source .env && \
uv run -m misinfo_data_eval.entrypoint \
--source_dataset_path hf://ComplexDataLab/Misinfo_Datasets@ce06269:liar_new:test \
--evaluate_feasibility \
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

## Run Evaluation on a dataset where tweet_id is available

```bash
source .env && \
uv run -m misinfo_data_eval.entrypoint \
--source_dataset_path hf://ComplexDataLab/Misinfo_Datasets@ce06269:twitter15:train \
--evaluate_temporal_correlation \
--max_concurrency 32 \
--limit -1
```

Example output:

```
Loading from HF hub: ComplexDataLab/Misinfo_Datasets
Revision: ce06269
Name of data subset: twitter15
Name of data split: train
len(dataset): 1043
1043
1043
1043
0
              precision    recall  f1-score   support

        True      1.000     1.000     1.000       261

    accuracy                          1.000       261
   macro avg      1.000     1.000     1.000       261
weighted avg      1.000     1.000     1.000       261
```

```python
{
  "True": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 261.0
  },
  "accuracy": 1.0,
  "macro avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 261.0
  },
  "weighted avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 261.0
  }
}
```
