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
```

```python
{
  "feasible, requires search": 50,
  "not feasible even with search": 18,
  "feasible, no search required": 4
}
```
## Run LLM Web Retrieval Accuracy Analysis

```bash
source .env && \
uv run -m misinfo_data_eval.entrypoint \
--source_dataset_path hf://ComplexDataLab/Misinfo_Datasets@ce06269:liar_new:test \
--evaluate_factuality \
--max_concurrency 32 \
--limit 72
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
len(dataset) filtered by veracity is not unknown: 793
len(dataset) filtered by tweet_id is not unknown: 793
0
              precision    recall  f1-score   support

           0      0.838     0.886     0.861        70
           1      0.936     0.907     0.921       129

    accuracy                          0.899       199
   macro avg      0.887     0.896     0.891       199
weighted avg      0.901     0.899     0.900       199
```

```python
{
  "0": {
    "precision": 0.8378378378378378,
    "recall": 0.8857142857142857,
    "f1-score": 0.8611111111111112,
    "support": 70.0
  },
  "1": {
    "precision": 0.936,
    "recall": 0.9069767441860465,
    "f1-score": 0.9212598425196851,
    "support": 129.0
  },
  "accuracy": 0.8994974874371859,
  "macro avg": {
    "precision": 0.8869189189189189,
    "recall": 0.896345514950166,
    "f1-score": 0.8911854768153982,
    "support": 199.0
  },
  "weighted avg": {
    "precision": 0.9014705962243652,
    "recall": 0.8994974874371859,
    "f1-score": 0.9001019973005887,
    "support": 199.0
  }
}
```


## Run Keyword Correlation Analysis

```bash
uv run -m misinfo_data_eval.entrypoint \
--source_dataset_path hf://ComplexDataLab/Misinfo_Datasets@ce06269:liar_new:test \
--keyword_analysis \
--limit 72
```

Example output:

```
[nltk_data] Downloading package punkt to /home/ubuntu/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package punkt_tab to /home/ubuntu/nltk_data...
[nltk_data]   Package punkt_tab is already up-to-date!
[nltk_data] Downloading package stopwords to /home/ubuntu/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
len(dataset) filtered by veracity is not unknown: 392
{
  "veracity_counts": {
    "0": 327,
    "1": 65
  },
  "veracity_proportions": {
    "0": 83.41836734693877,
    "1": 16.581632653061224
  },
  "top_keywords": {
    "\u201c": 268, "\u201d": 265, "\u2019": 91, "shows": 46, "biden": 42, "covid19": 26, "video": 26, "people": 23, "joe": 22, "president": 22, "says": 21, "new": 20, "photo": 18, "us": 18, "trump": 17, "ukraine": 16, "million": 13, "vaccines": 13, "\u2018": 13, "2020": 13, "covid": 12, "one": 12, "said": 11, "children": 11, "bill": 11, "election": 11, "year": 10, "pelosi": 9, "america": 9, "used": 9, "americans": 9, "would": 9, "voted": 9, "state": 9, "prices": 9, "abortion": 9, "tax": 9, "vaccine": 8, "agents": 8, "states": 8
  },
  "confusion_matrix": [
    [
      77,
      5
    ],
    [
      14,
      2
    ]
  ],
  "macro_f1_random_forest": 0.5320432269414426,
  "macro_f1_random_baseline": 0.4716490053913367
}
```