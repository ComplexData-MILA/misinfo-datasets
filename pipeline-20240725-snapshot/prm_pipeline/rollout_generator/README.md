## Rollout Generator
Example from project root folder:
```bash
source .env && python3 -m prm_pipeline.rollout_generator.worker \
    --model_name="codellama/CodeLlama-34b-Instruct-hf" \
    --template_json="prm_pipeline/rollout_generator/example_rollout_config.json" \
    --dataset_name="/path/to/dataset" \
    --num_processes=128
```

Additional examples:

```bash
source .env && python3 -m prm_pipeline.rollout_generator.worker \
    --model_name="lmsys/vicuna-13b-v1.5" \
    --template_json="prm_pipeline/rollout_generator/templates/json-20231020a1.json" \
    --dataset_name="liar" \
    --num_processes=128 \
    --num_repeats=16
```

```bash
source .env && python3 -m prm_pipeline.rollout_generator.worker \
    --model_name="lmsys/vicuna-13b-v1.5" \
    --template_json="prm_pipeline/rollout_generator/templates/definitive-20231019a1.json" \
    --dataset_name="liar" \
    --num_processes=128 \
    --num_repeats=16
```

```bash
source .env && python3 -m prm_pipeline.rollout_generator.worker \
 --model_name="lmsys/vicuna-13b-v1.5" \
 --template_json="prm_pipeline/rollout_generator/templates/example-20231027a1.json" \
 --dataset_name="data/liar-new-20231029a1" \
 --num_processes=128 \
 --num_repeats=16
```

```bash
source .env && python3 -m prm_pipeline.rollout_generator.worker \
 --model_name="lmsys/vicuna-13b-v1.5" \
 --template_json="prm_pipeline/rollout_generator/templates/example-20231029b1.json" \
 --dataset_name="data/liar-new-20231029a1" \
 --num_processes=128 \
 --num_repeats=16
```

Environment Variables
- OPENAI_API_BASE
- OPENAI_API_KEY
- WEBSERVER_API_BASE

CLI Arguments
- Model name will be forwarded to the OpenAI API

Example Template:
```json
{
    "rollout_config_version": "example-20231018a2",
    "template": "Below is a dubious statement: \n\n{statement} \n\nYour task is to present a careful step-by-step analysis of why and why not this statement should be considered as true or false. Present your analysis in this format: \n```\n<u>Step 1: Understanding the statement</u>\n...\n``` \nAFTER presenting a thorough analysis, you should summarize your reasoning and state whether you think this statement is True or False. To do so, end your response with a summary and then `<u>Verdict: True (code: 1)</u>` for True, or `<u>Verdict: False (code: 0)</u>` for False. You may also state `<u>Verdict: More info required (code: -1)</u>` if you need more information. Since you might change your mind as you write down your reasoning, you must not state your conclusion until you have presented a thorough step-by-step analysis. \n\n",
    "reasoning_delimiter_pattern": "</?u>|Step|```",
    "prediction_pattern": "code: ([-0-1]+)"
}
```

### Overview
- Load dataset of prompt and ground-truth labels
- Asynchronously: 
    - invoke the OpenAI ChatCompletion API to generate reasoning
    - Split each reasoning trace into steps with heuristics
    - Send rollout to remote server

### Specifications
