"""
Shared utilities for handling templates.
- Generating prompts.
- Parsing responses.
"""

import json
import re
from typing import Any, Dict, List, Optional

JSON_LIST_PATTERN = r"(\[[^\]].+\])"
PLACEHOLDER_PATTERN = r"\{\{[^\}]+\}\}"


def substitute(template: str, replacements: Dict[str, Any]) -> str:
    for placeholder, value in replacements.items():
        template = template.replace("{{" + placeholder + "}}", str(value))
        template = template.replace("{{" + placeholder + ".json}}", json.dumps(value))

    placeholder_matches = re.findall(PLACEHOLDER_PATTERN, template)
    if len(placeholder_matches) > 0:
        raise ValueError(
            "Some placeholders in template are yet to be replaced: {}".format(
                placeholder_matches
            )
        )

    return template


def split_reasoning(reasoning: str, delimiter_regexp: Optional[str]) -> List[str]:
    """
    Heuristics for splitting the reasoning trace.
    If delimiter_regexp is None, look for JSON lists and
    try parsing them.
    """
    output = []

    if delimiter_regexp is not None:
        for reasoning_step in re.split(delimiter_regexp, reasoning):
            reasoning_filtered = reasoning_step.lstrip().rstrip()
            if len(reasoning_filtered) > 0:
                output.append(reasoning_filtered)

    # Otherwise, look for JSON lists
    else:
        json_list_matches = re.findall(JSON_LIST_PATTERN, reasoning.replace("\n", ""))
        if len(json_list_matches) > 0:
            try:
                output = json.loads(json_list_matches[0])
            except json.JSONDecodeError:
                return []

    return output


def split_with_delimiter(delimiter_pattern: str, string: str) -> List[str]:
    """
    Split `string` with `delimiter_pattern`, but include
    delimiters in output *before* each part.

    For example:
    "Overview. Step 1. Example. Step 2. Example."
    with pattern "Step [0-9]\\.\\s+" would be split into
    ["Overview. ", "Step 1. Example. ", "Step 2. Example"]

    Params:
        string: str, string to split.
        delimiter_pattern: str, regular expression pattern of delimiters.

    Returns:
        List of string parts, including delimiter before each part.
    """
    parts: List[str] = []

    # Positions of the first and last character of the next part.
    part_start_index = 0
    part_end_index = 0

    delimiter_matches = list(re.finditer(delimiter_pattern, string))
    for delimiter_match in delimiter_matches:
        part_end_index, _ = delimiter_match.span()
        part = string[part_start_index:part_end_index]
        if len(part) > 0:
            parts.append(part)

        part_start_index = part_end_index

    # Add any trailing item to list.
    trailing_text = string[part_start_index:]
    if len(trailing_text) > 0:
        parts.append(trailing_text)

    return parts
