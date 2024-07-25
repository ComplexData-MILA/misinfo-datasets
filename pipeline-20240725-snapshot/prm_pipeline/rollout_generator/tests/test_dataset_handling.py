import json

import pytest

from prm_pipeline.rollout_generator.data_utils import get_prediction, load_dataset_dict


@pytest.fixture(scope="session")
def dataset_dict(hf_dataset_name: str = "liar"):
    dataset_dict = load_dataset_dict(hf_dataset_name)
    return dataset_dict


@pytest.fixture()
def dataset_entry(dataset_dict):
    return dataset_dict["test"][0]


@pytest.fixture(scope="session")
def template_data():
    with open(
        "prm_pipeline/rollout_generator/example_rollout_template.json", "r"
    ) as template_file:
        template_data = json.load(template_file)

    return template_data


def test_prompt_templating(dataset_entry, template_data):
    template_string = template_data["template"]
    example_statement = dataset_entry["statement"]

    prompt = template_string.format(statement=example_statement)
    print("prompt: \n```\n{}\n```".format(prompt))

    assert len(prompt) > 0


def test_get_prediction():
    example_response = "Example text <em>1</em>"
    example_prediction = get_prediction(example_response, r"<em>(-?[0-9]+)</em>")
    assert isinstance(example_prediction, int)
    assert example_prediction == 1

    example_response = "Example text <em>1.0</em>"
    example_prediction = get_prediction(example_response, r"<em>(-?[0-9\.]+)</em>")
    assert isinstance(example_prediction, float)
    assert example_prediction == 1.0

    example_response = "Example text <em>1.0.1</em>"
    example_prediction = get_prediction(example_response, r"<em>(-?[0-9\.]+)</em>")
    assert isinstance(example_prediction, str)
    assert example_prediction == "1.0.1"
