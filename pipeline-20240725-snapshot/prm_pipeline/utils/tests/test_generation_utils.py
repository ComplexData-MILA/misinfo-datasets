"""
Unit tests for text generation utils include:
- ChatCompletion with support for continuing conversation
    where the most recent message is from the assistant,
    not the user.
- Formatting rollout task context as a list of messages. 
    (both for rollout and for ai_preference)
    
"""

import pytest

from prm_pipeline._experiment_configs.interfaces import RolloutWorkerConfig
from prm_pipeline.utils.templating import split_with_delimiter
from prm_pipeline.utils.text_generation import (
    get_rollout_prompt,
    get_conversation_context,
)
from prm_pipeline.webserver.serialization_utils import RolloutTask


@pytest.fixture
def example_rollout_task() -> RolloutTask:
    return RolloutTask(
        tree_attributes={
            "forest_name": "example_forest",
            "prompt_template": "Example template.",
            "ground_truth": 0,
            "rollout_worker_config": RolloutWorkerConfig(
                # Mode might mistake "..." as part of the output
                # format we requested.
                rollout_delimiter_pattern=r"\s*Hello|Task",
                prediction_pattern=r"Status: \(([0-9])\)",
                # Maximum altitude beyond which no children should be added.
                max_branching_altitude=5,
                # Maximum number of rollouts retries permitted per task,
                # including initial one.
                max_num_retries=10,
                max_num_children=2,
                llm_completion_kwargs={"temperature": 0.1},
            )._asdict(),
            "tree_id": 0,
        },
        tree_id=0,
        rollout_context=[
            {
                "attributes": {"node_id": "0", "num_children": 1, "altitude": 1},
                "data": {
                    "content": 'Your task is to say "Hello! (n)" seven times, '
                    "Add two new lines after each output. "
                    'After that, (after line breaks) say "Task Complete; Status: (0)"'
                    "For example: \n\nHello! (1)\n\n\n\nHello! (7)\n\n"
                    "Task Complete; Status: (0)",
                    "role": "user",
                },
                "node_id": "0",
            },
            {
                "attributes": {"node_id": "1", "num_children": 1, "altitude": 2},
                "data": {
                    "content": "Hello! (1)\n\n",
                    "role": "assistant",
                },
                "node_id": "2",
            },
            {
                "attributes": {"node_id": "6", "num_children": 1, "altitude": 3},
                "data": {
                    "content": "Hello! (2)\n\n",
                    "role": "assistant",
                },
                "node_id": "7",
            },
        ],
    )


def test_get_rollout_context(example_rollout_task):
    """
    Test obtaining a rollout context for a rollout task.
    """
    rollout_context = example_rollout_task.rollout_context
    rollout_prompt = get_rollout_prompt(rollout_context)

    # Verify relative position of nodes texts
    root_prompt = rollout_context[0]["data"]["content"]
    previous_index = rollout_prompt.index(root_prompt)

    for rollout_step in rollout_context[1:]:
        # Compare position of text from the current step
        # with position of text from the step before.
        text = rollout_step["data"]["content"]
        text_index = rollout_prompt.index(text)
        assert text_index > previous_index

        previous_index = text_index


def test_get_rollout_context_messages(example_rollout_task):
    """
    Test obtaining message list compatible
    with ChatCompletion.
    """
    rollout_context = example_rollout_task.rollout_context
    conversation_context = get_conversation_context(
        rollout_context, {"user": "USER", "assistant": "CHATBOT"}
    )
    conversation_context_ref = [
        {"role": "USER", "content": rollout_context[0]["data"]["content"]},
        {
            "role": "CHATBOT",
            "content": rollout_context[1]["data"]["content"]
            + rollout_context[2]["data"]["content"],
        },
    ]
    assert conversation_context == conversation_context_ref

    rollout_context = [
        {"data": {"role": "user", "content": "user message 1"}},
        {"data": {"role": "assistant", "content": "assistant message 1"}},
        {"data": {"role": "user", "content": "user message 2"}},
        {"data": {"role": "assistant", "content": ""}},
    ]
    conversation_context = get_conversation_context(rollout_context)
    conversation_context_ref = [
        {"role": "user", "content": rollout_context[0]["data"]["content"]},
        {"role": "assistant", "content": rollout_context[1]["data"]["content"]},
        {"role": "user", "content": rollout_context[2]["data"]["content"]},
    ]
    assert conversation_context == conversation_context_ref


def test_split_with_delimiter():
    string = "example1with2trailing text"
    pattern = r"[0-9]"
    parts = split_with_delimiter(pattern, string)
    assert parts == ["example", "1with", "2trailing text"]

    string = "1example with2trailing text"
    pattern = r"[0-9]"
    parts = split_with_delimiter(pattern, string)
    assert parts == ["1example with", "2trailing text"]

    string = "example1without trailing text2"
    pattern = r"[0-9]"
    parts = split_with_delimiter(pattern, string)
    assert parts == ["example", "1without trailing text", "2"]

    string = "example without delimiter"
    pattern = r"[0-9]"
    parts = split_with_delimiter(pattern, string)
    assert parts == ["example without delimiter"]
