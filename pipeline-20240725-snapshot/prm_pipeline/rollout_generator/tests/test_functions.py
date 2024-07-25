import pytest

from prm_pipeline._experiment_configs.interfaces import RolloutWorkerConfig
from prm_pipeline.rollout_generator.functions import generate_reasoning, generate_search
from prm_pipeline.rollout_generator.functions.reasoning import _parse_reasoning_trace
from prm_pipeline.utils.tests.test_generation_utils import example_rollout_task
from prm_pipeline.utils.tests.test_action_utils import example_actions_config_serialized
from prm_pipeline.utils.tree_utils import NodeId, PathElement
from prm_pipeline.webserver.serialization_utils import RolloutTask
import json


@pytest.fixture()
def example_search_task(
    example_rollout_task, example_actions_config_serialized
) -> RolloutTask:
    context = [
        {
            "attributes": {"node_id": "0", "num_children": 1, "altitude": 1},
            "data": {
                "content": "Your task is to find out the scientific name of red pandas.",
                "role": "user",
            },
            "node_id": "0",
        },
        {
            "attributes": {
                "node_id": "1",
                "num_children": 1,
                "altitude": 2,
                "action": {
                    "type": "search",
                    "payload": ("What is the scientific name of red pandas?",),
                },
            },
            "data": {
                "content": "Question: What is the scientific name of red pandas?",
                "role": "assistant",
            },
            "node_id": "2",
        },
    ]

    attributes = example_rollout_task.tree_attributes
    attributes["actions_config"] = example_actions_config_serialized
    example_rollout_task = example_rollout_task._replace(
        tree_attributes=attributes, rollout_context=context
    )

    return example_rollout_task


@pytest.fixture
def stub_rollout_context(example_rollout_task) -> RolloutTask:
    # test ChatCompletion with rollout context consisting of only
    # message from user.
    stub_rollout_context = example_rollout_task.rollout_context[:1]
    stub_rollout_context = example_rollout_task._replace(
        rollout_context=stub_rollout_context
    )
    return stub_rollout_context


def test_generate_reasoning(example_rollout_task):
    """
    Verify if the generate_reasoning function produces
    a reasoning node of the correct format.
    """
    output = generate_reasoning(example_rollout_task, "Completion")
    assert output is not None
    for node in output.nodes:
        print(node)

    assert len(output.nodes) == 6

    # Add the first output to the rollout task
    # and re-run the generation process to get
    # an alternative path.
    new_node = output.nodes[0]
    new_node_id: NodeId = "101"
    new_node.attributes["node_id"] = new_node_id
    new_path_element = PathElement(new_node_id, new_node, [new_node])
    updated_rollout_context = (
        example_rollout_task.rollout_context
        + PathElement.serialize_path([new_path_element])
    )

    example_rollout_task = example_rollout_task._replace(
        rollout_context=updated_rollout_context
    )
    reasoning_output = generate_reasoning(example_rollout_task, "Completion")
    assert reasoning_output is not None
    for node in reasoning_output.nodes:
        print(node)

    assert len(reasoning_output.nodes) == 5

    leaf_node_attributes = reasoning_output.nodes[-1].attributes
    assert leaf_node_attributes["num_leaf_nodes"] == 1
    assert leaf_node_attributes["num_leaf_nodes"] == 1


def test_generate_reasoning_chat_completion(stub_rollout_context):
    reasoning_output = generate_reasoning(stub_rollout_context, "ChatCompletion")
    assert reasoning_output is not None
    for node in reasoning_output.nodes:
        print(node)

    assert len(reasoning_output.nodes) == 8


def test_generate_reasoning_cohere_chat(example_rollout_task, stub_rollout_context):
    reasoning_output = generate_reasoning(stub_rollout_context, "CohereChatSearch")
    assert reasoning_output is not None
    for node in reasoning_output.nodes:
        print(node)

    assert len(reasoning_output.nodes) == 8

    reasoning_output = generate_reasoning(example_rollout_task, "CohereChatSearch")
    assert reasoning_output is not None
    for node in reasoning_output.nodes:
        print(node)

    assert len(reasoning_output.nodes) == 6


def test_parse_reasoning_trace():
    """
    Verify _parse_reasoning_trace correctly splits and formats
    reasoning traces from LLM Completion.
    """
    rollout_config = RolloutWorkerConfig(
        rollout_delimiter_pattern=r"Step\s+[0-9]:",
        prediction_pattern=r"\(([0-9])\)",
        max_branching_altitude=10,
        max_num_retries=10,
        max_num_children=2,
        llm_completion_kwargs={"temperature": 0.1},
    )
    reasoning_steps = [
        "Example step 0 \n",  # altitude: 10
        "Step 1: Example step 1 \n",  # altitude: 11
        "Step 2: Example conclusion: (1)",  # altitude: 12
    ]
    parser_output = _parse_reasoning_trace("".join(reasoning_steps), rollout_config, 9)

    assert parser_output is not None
    reasoning_tree_nodes = parser_output.rollout_nodes
    assert len(reasoning_tree_nodes) == 3
    for reasoning_node_parsed, reasoning in zip(
        reasoning_tree_nodes[:-1], reasoning_steps[:-1]
    ):
        assert reasoning_node_parsed.data["content"] == reasoning
        assert reasoning_node_parsed.data["role"] == "assistant"

    assert reasoning_tree_nodes[-1].data["content"] == reasoning_steps[-1]
    assert reasoning_tree_nodes[-1].attributes["prediction"] == 1

    num_children = [node.attributes["num_children"] for node in reasoning_tree_nodes]
    assert num_children == [2, 0, 0]

    altitude = [node.attributes["altitude"] for node in reasoning_tree_nodes]
    assert altitude == [10, 11, 12]


def test_parse_reasoning_trace_after_prediction():
    """
    Verify _parse_reasoning_trace correctly sets num_children value
    for unnecessary reasoning steps after prediction.
    """
    rollout_config = RolloutWorkerConfig(
        rollout_delimiter_pattern=r"Step\s+[0-9]:",
        prediction_pattern=r"\(([0-9])\)",
        max_branching_altitude=10,
        max_num_retries=10,
        max_num_children=2,
    )
    reasoning_steps = [
        "Example step 0 \n",  # altitude: 7
        "Step 1: Example conclusion: (1)",  # altitude: 8
        "Step 2: Example extra step",  # altitude: 9
    ]
    parser_output = _parse_reasoning_trace("".join(reasoning_steps), rollout_config, 6)

    assert parser_output is not None
    reasoning_tree_nodes = parser_output.rollout_nodes

    num_children = [node.attributes["num_children"] for node in reasoning_tree_nodes]
    predictions = [node.attributes.get("prediction") for node in reasoning_tree_nodes]
    assert num_children == [2, 0, 0]
    # prediction value should be copied from predecessor
    assert predictions == [None, 1, 1]


def test_generate_search(example_search_task):
    rollout_submission = generate_search(example_search_task)
    assert rollout_submission is not None
    assert len(rollout_submission.nodes) == 1

    search_output_node = rollout_submission.nodes[0]
    assert search_output_node.data["role"] == "assistant"
    assert search_output_node.data["source"] == "search"
    assert isinstance(search_output_node.data["content"], str)
    assert len(search_output_node.data["content"]) > 0
    print(rollout_submission.nodes[0])
    print(json.dumps(rollout_submission._asdict(), indent=2))
