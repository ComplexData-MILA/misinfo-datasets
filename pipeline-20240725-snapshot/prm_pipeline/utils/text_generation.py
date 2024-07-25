"""
Utilities for interfacing with external LLMs for
text generation. 
- ChatCompletion wrapper.
- generating conversation prompts for the Vicuna model.
"""
import base64
from typing import Dict, List

import openai


def chat_competion(model_name, prompt) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
    except openai.APIError as e:
        print("Exception: {}".format(e), flush=True)
        return ""

    reasoning_trace = response["choices"][0]["message"]["content"]  # type: ignore
    return reasoning_trace


def get_conversation_prompt(conversation: List[Dict[str, str]]) -> str:
    """
    Given a "conversation" (a list of chat messages,)
    return a prompt compatible with Vicuna. This method
    will append a blank message with the "assistant" role.

    Design choices:
    - If two subsequent messages are from the same role,
        do not insert separator between them. Rather, concatenate
        directly.
    - As a result, the blank "assistant" message would not affect
        the prompt output if the last message of conversation is
        from the assistant.

    Params:
        Conversation: List of {"role": "", "content": ""} dicts.

    Return:
        str: prompt compatible with the Completion API.
    """
    prefix_lookup = {
        "system": "",
        "_first_user": " USER: ",  # First user message.
        "user": "</s>USER: ",  # All subsequent messages from user.
        "assistant": " ASSISTANT: ",
    }
    system_message = base64.b64decode(
        "QSBjaGF0IGJldHdlZW4gYSBjdXJpb3VzIHVzZX"
        "IgYW5kIGFuIGFydGlmaWNpYWwgaW50ZWxsaWdl"
        "bmNlIGFzc2lzdGFudC4gVGhlIGFzc2lzdGFudC"
        "BnaXZlcyBoZWxwZnVsLCBkZXRhaWxlZCwgYW5k"
        "IHBvbGl0ZSBhbnN3ZXJzIHRvIHRoZSB1c2VyJ3"
        "MgcXVlc3Rpb25zLg=="
    ).decode("utf-8")

    # Add "system" message and a blank "assistant" message
    # to the conversation. Replace the "user" role in the first
    # message with "_first_user".
    conversation = [
        {"role": "system", "content": system_message},
        {**conversation[0], "role": "_first_user"},
        *conversation[1:],
        {"role": "assistant", "content": ""},
    ]

    # Initialize buffer for tracking the role of the previous message.
    prev_role = ""

    # Output buffer
    output_buffer: List[str] = []

    for message_dict in conversation:
        author_role = message_dict["role"]
        message_content = message_dict["content"]

        # Append prefix only if role differs.
        # Otherwise, concatenate role and message.
        if author_role == prev_role:
            message_formatted = message_content
        else:
            prev_role = author_role
            prefix = prefix_lookup[author_role]
            message_formatted = "{prefix}{message}".format(
                prefix=prefix, message=message_content
            )

        output_buffer.append(message_formatted)

    return "".join(output_buffer).rstrip(" ")


def get_conversation_context(
    rollout_context: List[Dict[str, Dict[str, str]]], role_vocab: Dict[str, str] = {}
) -> List[Dict[str, str]]:
    """
    Given the task description from GET /rollout_task,
    produce a "conversation" context consisting of a list of messages.
    Concatenate consecutive messages from the same role.

    Params:
        rollout_context: List of Dict
        role_vocab: optionally, map one role to another.
            e.g., {"user": "USER", "assistant": "CHATBOT}

    Returns:
        List of "messages", where each message is a dictionary
            with key "role" and "content".
    """

    if len(rollout_context) == 0:
        raise ValueError("rollout_context is empty.")

    concatenated_messages = []
    first_message = rollout_context[0]["data"]

    pending_role = first_message["role"]
    pending_content = first_message["content"]

    # Keep concatenating messages until role switches.
    for context_step in rollout_context[1:]:
        message = context_step["data"]

        if pending_role == message["role"]:
            pending_content += message["content"]
        else:
            # role switches
            concatenated_message = {
                "role": role_vocab.get(pending_role, pending_role),
                "content": pending_content,
            }
            concatenated_messages.append(concatenated_message)

            pending_role = message["role"]
            pending_content = message["content"]

    # trailing message
    concatenated_message = {
        "role": role_vocab.get(pending_role, pending_role),
        "content": pending_content,
    }
    concatenated_messages.append(concatenated_message)

    return list(
        filter(lambda message: len(message["content"]) > 0, concatenated_messages)
    )


def get_rollout_prompt(rollout_context: List[Dict[str, Dict[str, str]]]) -> str:
    """
    Given the task description from GET /rollout_task,
    produce a conversation history. Concatenate consecutive
    messages from the same role.

    Params:
        rollout_context: List of Dict

    Returns:
        str: prompt for the Completion API.
    """
    concatenated_messages = get_conversation_context(rollout_context)
    prompt = get_conversation_prompt(concatenated_messages)
    return prompt
