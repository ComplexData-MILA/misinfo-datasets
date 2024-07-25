"""
Tests for ensuring that ChatCompletion generation resumes as if
there hasn't been an interruption in the conversation.
"""
import json
from typing import Dict, List

import pytest
from fastchat.conversation import Conversation, SeparatorStyle, get_conv_template

from prm_pipeline.utils.text_generation import get_conversation_prompt


@pytest.fixture()
def example_message_basic():
    """
    Example conversation, with the most recent message
    from the user.
    """
    messages = [
        {"role": "user", "content": "Example user prompt 0."},
    ]
    return messages


@pytest.fixture()
def example_message_reference():
    """
    Example conversation, with the most recent message
    from the user.
    """
    messages = [
        {"role": "user", "content": "Example user prompt 0."},
        {"role": "assistant", "content": "Example assistant output 0."},
        {"role": "user", "content": "Example user prompt 1."},
        {"role": "assistant", "content": "Example assistant output 1."},
        {"role": "user", "content": "Example user prompt 2."},
    ]
    return messages


@pytest.fixture()
def example_message_run_on():
    """
    Example conversation where the most recent message is
    from the assistant- an undefined behavior. The handling
    of this type of messages in vLLM might deviate from the
    intended behaivor.
    """
    messages = [
        {"role": "user", "content": "Example user prompt 0."},
        {"role": "assistant", "content": "Example assistant output 0."},
        {"role": "user", "content": "Example user prompt 1."},
        {"role": "assistant", "content": "Example assistant output 1."},
        {"role": "user", "content": "Example user prompt 2."},
        {"role": "assistant", "content": "Example assistant output 2."},
    ]
    return messages


def get_fastchat_example_prompt_output(
    messages: List[Dict[str, str]], add_blank_message: bool = True
) -> str:
    """
    Retrieve output from fastchat templating.

    Params:
        add_blank_message: boolean, whether to add a blank assistant
            message to output. The default value of True would produce
            the same behavior as the reference vLLM implementation.

    Returns:
        str: prompt representing the conversation.
    """

    conv = get_conv_template("vicuna_v1.1")

    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    for message in messages:
        msg_role = message["role"]
        if msg_role == "system":
            conv.system_message = message["content"]
        elif msg_role == "user":
            conv.append_message(conv.roles[0], message["content"])
        elif msg_role == "assistant":
            conv.append_message(conv.roles[1], message["content"])  # type: ignore
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    # Add a blank message for the assistant.
    if add_blank_message:
        conv.append_message(conv.roles[1], None)  # type: ignore
    prompt = conv.get_prompt()
    return prompt


def test_conversation_prompt_equivalence_reference(
    example_message_basic, example_message_reference
):
    """
    Verify equivalence with vLLM for the usual input format, where
    the most recent message is from "user".
    """
    for example_message in [example_message_basic, example_message_reference]:
        prompt_reference = get_fastchat_example_prompt_output(
            example_message, add_blank_message=True
        )
        prompt = get_conversation_prompt(example_message)
        assert prompt == prompt_reference, json.dumps(
            {"reference": prompt_reference, "actual": prompt}, indent=2
        )


def test_conversation_prompt_equivalence_run_on(example_message_run_on):
    """
    Verify equivalence with vLLM for the run-on input format, where
    the most recent message is from "assistant".
    """
    # Output should be equivalent to vLLM implementation
    # without the extra blank assistant message.
    prompt_reference = get_fastchat_example_prompt_output(
        example_message_run_on, add_blank_message=False
    )[: -len("</s>")]
    prompt = get_conversation_prompt(example_message_run_on)
    assert prompt == prompt_reference, json.dumps(
        {"reference": prompt_reference, "actual": prompt}, indent=2
    )
