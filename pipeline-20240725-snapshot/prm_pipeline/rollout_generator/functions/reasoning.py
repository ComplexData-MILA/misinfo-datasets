import logging
import os
from typing import Any, Dict, List, Literal, NamedTuple, Optional

import cohere
import openai

from ..._experiment_configs.interfaces import RolloutWorkerConfig
from ...utils import ReasoningTreeNode
from ...utils.templating import split_with_delimiter
from ...utils.text_generation import get_conversation_context, get_conversation_prompt
from ...webserver.serialization_utils import RolloutSubmission, RolloutTask
from ..data_utils import get_prediction, PredictionType

logger = logging.getLogger(__name__)

endpoint_options = ["ChatCompletion", "Completion", "CohereChat", "CohereChatSearch"]

cohere_role_vocab = {"user": "USER", "assistant": "CHATBOT"}
role_vocab_lookup = {
    "CohereChat": cohere_role_vocab,
    "CohereChatSearch": cohere_role_vocab,
}


class ReasoningTraceParserOutput(NamedTuple):
    rollout_nodes: List[ReasoningTreeNode]
    prediction: Optional[PredictionType]


def generate_reasoning(
    rollout_task: RolloutTask, endpoint_name: str
) -> Optional[RolloutSubmission]:
    """
    Generate reasoning via Completion LLM API.

    Design Note: LLM model name is retrieved directly from env var.

    Params:
        rollout_task: Dict
        - tree_id: int
        - tree_attributes: Dict
            - rollout_config: Dict
                - rollout_delimiter: str
                - rollout_delimiter_pattern: str
                - prediction_pattern: str
        - rollout_context: List of Dict
            - node_id: str
            - node_attributes: Dict
            - node_data: Dict
                - role: str, "user" (first message) or "assistant"
                - content: str

    Returns:
        RolloutSubmission,
        or None if the generation is invalid (esp. OpenAI Exception)
    """
    model_name = os.environ["MODEL_NAME"]
    assert endpoint_name in endpoint_options

    # Parse rollout task to obtain LLM context of messages
    rollout_context = rollout_task.rollout_context
    rollout_config_dict = rollout_task.tree_attributes["rollout_worker_config"]
    rollout_config = RolloutWorkerConfig(**rollout_config_dict)
    role_vocab = role_vocab_lookup.get(endpoint_name, {})
    conversation_context = get_conversation_context(rollout_context, role_vocab)

    assert len(rollout_task.rollout_context) > 0
    parent_node = rollout_task.rollout_context[-1]
    parent_altitude = parent_node["attributes"]["altitude"]

    parser_output: Optional[ReasoningTraceParserOutput] = None
    num_tokens: Dict[str, int] = {}

    # Retry for up to max_num_retries in case of LLM API error or
    # no prediction being matched in output.
    for retry_index in range(rollout_config.max_num_retries):
        assert endpoint_name in endpoint_options
        try:
            if endpoint_name == "ChatCompletion":
                model_response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=conversation_context,
                    max_tokens=1024,
                    **rollout_config.llm_completion_kwargs
                )

                assert isinstance(model_response, dict)
                reasoning_text = model_response["choices"][0]["message"]["content"]
                num_tokens["context"] = model_response["usage"]["prompt_tokens"]
                num_tokens["output"] = model_response["usage"]["completion_tokens"]
                num_tokens["total"] = model_response["usage"]["total_tokens"]

            elif endpoint_name == "Completion":
                prompt = get_conversation_prompt(conversation_context)
                model_response = openai.Completion.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=1024,
                    **rollout_config.llm_completion_kwargs
                )
                assert isinstance(model_response, dict)
                reasoning_text = model_response["choices"][0]["text"]

            elif endpoint_name.startswith("Cohere"):
                api_key = os.environ["COHERE_KEY"]
                client = cohere.Client(api_key)
                if "Search" in endpoint_name:
                    model_name = "_cohere_web_search"
                    connectors = [{"id": "web-search"}]
                else:
                    model_name = "_cohere"
                    connectors = None

                chat_history = []
                for message in conversation_context:
                    chat_history.append(
                        {"role": message["role"], "message": message["content"]}
                    )

                response = client.chat(
                    message=conversation_context[-1]["content"],
                    chat_history=chat_history,
                    connectors=connectors,
                )
                reasoning_text: str = response.text
                token_count = response.token_count
                if isinstance(token_count, dict):
                    num_tokens["context"] = token_count["prompt_tokens"]
                    num_tokens["output"] = token_count["response_tokens"]
                    num_tokens["total"] = token_count["total_tokens"]
                    num_tokens["billed"] = token_count["billed_tokens"]

        except Exception as e:
            logger.error("conversation_context: {}".format(conversation_context))
            logger.error("Encountered error; skipping this prompt. {}".format(e))
            continue

        if os.environ.get("PRINT_LLM_OUTPUT") == "True":
            print(prompt)
            print(reasoning_text)

        parser_output = _parse_reasoning_trace(
            reasoning_text, rollout_config, parent_altitude
        )

        if os.environ.get("DEBUG") is not None:
            parser_output.rollout_nodes[0].attributes["debug"] = {
                "prompt": prompt,
                "conversation_context": conversation_context,
                "reasoning_text": reasoning_text,
            }

        if parser_output.prediction is not None:
            break

    if (not parser_output) or (len(parser_output.rollout_nodes) == 0):
        # LLM API does not work for max_num_retries times
        return None

    parser_output.rollout_nodes[0].attributes["num_tokens"] = num_tokens

    # If no prediction is found after max_num_retries,
    # upload the rollout but prevent new children from
    # being added to these nodes.
    for node in parser_output.rollout_nodes:
        node.attributes["retry_index"] = retry_index
        node.attributes["model_name"] = model_name

        if parser_output.prediction is None:
            node.attributes["num_children"] = 0

    # If prediction is extracted, add value attributes to
    # the leaf node.
    leaf_node = parser_output.rollout_nodes[-1]
    if parser_output.prediction is not None:
        is_correct = (
            parser_output.prediction == rollout_task.tree_attributes["ground_truth"]
        )
        leaf_node.attributes["num_leaf_nodes"] = 1
        leaf_node.attributes["leaf_value_sum"] = int(is_correct)

    rollout_submission = RolloutSubmission(
        rollout_task.tree_id,
        rollout_task.parent_node_id,
        parser_output.rollout_nodes,
    )

    return rollout_submission


def _parse_reasoning_trace(
    reasoning_trace: str, rollout_config: RolloutWorkerConfig, parent_altitude: int
) -> ReasoningTraceParserOutput:
    """
    Parse reasoning trace from LLM with a given
    rollout configuration file, obtaining a list of
    reasoning tree nodes, one for each step in
    the reasoning process. Parse each step for prediction.
    Set max_num_children value for each step.

    The most recent node in the reasoning trace has no children.

    Nodes will have no children in the following situations:
    - altitude > max_branching_altitude
    - some node before this has a parsed prediction
    - no prediction could be extracted from this rollout

    Design choice: Skip nodes with blank content.

    Design choice: all nodes following a node with a valid
    "prediction" match will have the same prediction attribute,
    even if the node itself does not include a prediction match.

    Param:
        reasoning_trace: str, output from Completion LLM.
        rollout_config: Dict[str, str], rollout configuration
            from tree attributes:
            - rollout_delimiter_pattern: str, regexp, e.g., "\n\n"
            - prediction_pattern: str, regexp with a capture group
        parent_altitude: int, altitude of the node immediately before
            the first step of this reasoning trace.

    Returns:
        List[ReasoningTreeNodes]

    """
    rollout_delimiter_pattern = rollout_config.rollout_delimiter_pattern
    prediction_pattern = rollout_config.prediction_pattern
    max_branching_altitude = rollout_config.max_branching_altitude
    max_num_children = rollout_config.max_num_children

    reasoning_tree_nodes: List[ReasoningTreeNode] = []

    # Prediction in this node or some other node before this node.
    prediction = None

    reasoning_steps = split_with_delimiter(rollout_delimiter_pattern, reasoning_trace)

    altitude = parent_altitude + 1
    for reasoning_step in reasoning_steps:
        if len(reasoning_step) == 0:
            continue

        # If a node has no prediction match, but a predecessor has
        # a valid match, use the value from the predecessor.
        prediction_output = get_prediction(reasoning_step, prediction_pattern)
        if prediction_output is not None:
            prediction = prediction_output

        if (altitude <= max_branching_altitude) and (prediction is None):
            num_children = max_num_children
        else:
            num_children = 0

        node_attributes: Dict[str, Any] = {
            "num_children": num_children,
            "altitude": altitude,
        }
        if prediction is not None:
            node_attributes["prediction"] = prediction

        node_data = {
            "role": "assistant",
            "content": reasoning_step,
        }
        node = ReasoningTreeNode(node_attributes, node_data)
        reasoning_tree_nodes.append(node)
        altitude += 1

    # The most recent node in each trace is a leaf.
    if len(reasoning_tree_nodes) > 0:
        reasoning_tree_nodes[-1].attributes["num_children"] = 0

    return ReasoningTraceParserOutput(reasoning_tree_nodes, prediction)
