"""
Paste example output in example-output.txt under project root.
"""
import os

import pytest

from prm_pipeline.utils.templating import split_reasoning


@pytest.fixture()
def llm_output():
    input_file = "example-output.txt"
    if not os.path.exists(input_file):
        return "[]"

    with open(input_file, "r") as input_file:
        return input_file.read()


def test_split_reasoning(llm_output):
    parsed = split_reasoning(llm_output, None)
    print(parsed)
