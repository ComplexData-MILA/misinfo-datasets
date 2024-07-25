# Webserver for Active Learning
Note that 0 is for not preferred, 1 is for preferred, and 2 is for neutral.

```bash
source .env && WEBSERVER_PORT=25565 python3 -m prm_pipeline.webserver.wsgi
```

## API Interfaces
### Rollout Generator
POST /rollout
- Data: JSON
    - Dictionary
        - model_identifier: `model: str`
        - example prompt from dataset: `example: str`
        - step-by-step reasoning: `reasoning: list of str`
        - prediction of class label: `prediction: int`
        - ground truth class label: `ground_truth: int`
        - worker identifier: `worker: str`
- Response: 200 
    - Database id: `int`.


### Process Reward Model inference controller
GET /prm_adapters
- Response: JSON
    - Return list of dictionaries, one for each process reward model adapter. Attribute of each dictionary:
        - prm_version identifier: `prm_version: str`

GET /prm_adapter?prm_version={prm_version:str}
- Response: Bytes
    - Serialized LoRA adapter for the specific process reward model version

GET /rollout?prm_version={prm_version:str}
- Response: JSON
    - List of rollouts that haven't yet been labelled with the specific process reward model version: `List of dictionaries` 
        - attributes are a superset of attributes in the payload of POST /rollout, with an additional attribute storing rollout database id: `id: int`.
    - Might hang if all rollouts have been labelled with this version of the process reward model
- Note: the webserver should mark a rollout as checked-out for this prm_version until some timeout after sending that rollout to an inference controller.

POST /process_reward_label
- Data: JSON
    - Dictionary
        - rollout identifier: `rollout_id: int`
        - step-by-step rollout reward predictions: `prm_output: List[float]`
        - prm version identifier: `prm_version: str`
        - worker identifier: `worker: str`
        - (optional) explanations: `explanations: List[str]` 
- Response: 200
    - Database id: `int`
- Note: the webserver should delete the checked-out marker from this rollout


### AI Preference generator
GET /rollout?prm_version={labelling_scheme_version:str}&active_learning=True
- `labelling_scheme_version: str` identifies the labeller and labelling scheme: prompt, etc.  
- Response: JSON
    - Same format as when active_learning=False.
    - Prioritize reasoning traces where process reward model predictions disagree with ground truth labels.

POST /process_reward_label
- Same as process reward model inference, except that `prm_version` should be the same as `labelling_scheme_version`.


### Process Reward Model training loop
GET /process_reward_labels?prm_version={labelling_scheme_version:str}
- Response: JSON of process reward label database keys for this prm_version.
    - List of integers

GET /process_reward_label?keys={keys_list_json:str}
- `keys` is a JSON-serialized list of reward label dataset keys for the labels to retrieve.
- Response: list of dictionaries
    - Same format as process_reward_label POST content.

POST /prm_adapter?prm_version={prm_version:str}&base_model={base_model_name:str}&worker={worker_name:str}
- Data: Bytes
    - Send serialized process reward model LoRA adapters
- Response: 200
    - Path to adapter file on the server.