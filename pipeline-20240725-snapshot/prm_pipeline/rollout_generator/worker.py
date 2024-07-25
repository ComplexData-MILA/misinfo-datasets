"""
- Load dataset of prompt and ground-truth labels
- Asynchronously: 
    - invoke the OpenAI ChatCompletion API to generate reasoning
    - Send rollout to remote server

Example:
```bash
python3 -m prm_pipeline.rollout_generator.worker \
 --forest_name=example-dataset/example-model \
 --num_processes=128
```
"""
import argparse
import datetime
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from socket import gethostname
from typing import Dict
from urllib.parse import urljoin

import requests

from ..webserver.serialization_utils import RolloutTask
from .functions import generate_reasoning, generate_search
from ..utils.action_utils import ActionsConfig, process_actions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def retrieve_and_submit(
    extra_attributes: Dict[str, str],
    webserver_api_base: str,
    forest_name: str,
) -> bool:
    """
    Retrieve a rollout job from controller web server.
    Generate the next rollout step with the LLM, and
    submit the result back to the controller.

    Run this function in parallel across multiple threads.
    One concurrent chat_completion request per invokation.

    TODO: implement function calling.
    (e.g., retrieval; chat_completion would also be a function)

    Params:
        extra_attributes: added to node attributes.
        model_name: str, forwarded to chat_completion.
        webserver_api_base

    Returns:
        boolean.
        - True, if a rollout was submitted.
        - False, if no rollout job is available.
    """
    try:
        # Retrieve rollout task from controller server
        task_request_response = requests.get(
            urljoin(
                webserver_api_base, "/rollout_task?forest_name={}".format(forest_name)
            )
        )

    except requests.exceptions.ConnectionError as e:
        logger.info(e)
        return False

    if not task_request_response.ok:
        print(
            "Error in HTTP response from controller: {}".format(task_request_response)
        )

    if len(task_request_response.json().keys()) == 0:
        return False

    rollout_task = RolloutTask(**task_request_response.json())

    # TODO: automatically select function to invoke based on task_request_response.
    endpoint_name = os.environ.get("ENDPOINT_NAME", "Completion")
    action_name = (
        rollout_task.rollout_context[-1]["attributes"].get("action", {}).get("name")
    )

    if action_name == "search":
        rollout_submit_payload = generate_search(rollout_task)
    else:
        rollout_submit_payload = generate_reasoning(
            rollout_task, endpoint_name  # type: ignore
        )

    if rollout_submit_payload is None:
        # TODO: Implement retry when generation is incomplete.
        return False

    actions_config_data = rollout_task.tree_attributes.get("actions_config")
    if actions_config_data is not None:
        actions_config = ActionsConfig.from_dict(actions_config_data)
        new_nodes_updated = process_actions(
            rollout_submit_payload.nodes, actions_config
        )
        rollout_submit_payload = rollout_submit_payload._replace(
            nodes=new_nodes_updated
        )

    # Add rollout work info to node attributes.
    for node in rollout_submit_payload.nodes:
        for attribute, attribute_value in extra_attributes.items():
            node.attributes[attribute] = attribute_value

    rollout_submit_payload_dict: Dict = rollout_submit_payload.to_dict()

    # Send serialized reasoning tree node to controller server.
    try:
        task_submission_response = requests.post(
            urljoin(
                webserver_api_base, "/rollout_task?forest_name={}".format(forest_name)
            ),
            json=rollout_submit_payload_dict,
        )

        if not task_submission_response.ok:
            return False

        return True

    except requests.exceptions.ConnectionError as e:
        logger.info(e)
        return False


# Asynchronously process the prompts with N workers
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest_name", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=128)
    args = parser.parse_args()

    webserver_api_base = os.environ.get("WEBSERVER_API_BASE", "http://localhost:25565/")
    logger.info("webserver_api_base: {}".format(webserver_api_base))

    # Extra attributes for the webserver
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    extra_attributes = {
        "worker": "{}/{}".format(gethostname(), timestamp),
    }

    num_requested: int = 0
    num_submitted: int = 0
    num_skipped: int = 0
    print()

    with ThreadPoolExecutor(args.num_processes) as executor:
        futures = []
        while True:
            future = executor.submit(
                retrieve_and_submit,
                extra_attributes,
                webserver_api_base,
                args.forest_name,
            )
            futures.append(future)
            num_requested += 1

            if len(futures) < args.num_processes:
                print("\rRequested: {}".format(num_requested), end="")

            while len(futures) >= args.num_processes:
                new_futures = []
                for future in futures:
                    future: Future
                    if future.done():
                        is_submitted = future.result()
                        if is_submitted:
                            num_submitted += 1
                        else:
                            num_skipped += 1
                            num_requested -= 1
                            time.sleep(10)

                        print(
                            "\rSubmitted/Skipped/Requested: "
                            "{:5}/{:5}/{:5}".format(
                                num_submitted, num_skipped, num_requested
                            ),
                            end="",
                        )
                    else:
                        new_futures.append(future)

                futures = new_futures
