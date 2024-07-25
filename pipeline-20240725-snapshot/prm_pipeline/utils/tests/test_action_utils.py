import pytest

from .. import ReasoningTreeNode
from ..action_utils import ActionsConfig, process_actions

example_thought_message = """
What is a red panda? 
I need to refer to an external source for this information.
"""

example_search_message = """\
Therefore, my question is: Question: 
What is the scientific name of red pandas?

Answer: 
I need to refer to an external source for this information.
"""

example_search_match = "Question: \nWhat is the scientific name of red pandas?"

example_max_altitude_placeholder = "All things considered"

example_payload = "Answer in one sentence: What is the scientific name of red pandas?"


@pytest.fixture()
def example_actions_config_serialized():
    return {
        "actions": [
            {
                "name": "jupyter",
                "pattern": r"Python:[\s\n]*```(?P<code>[^`]+)```",
                "query_templates": ["{code}"],
            },
            {
                "name": "search",
                "pattern": r"Question:[\s\n]*(?P<question>[^?]+\?)",
                "max_altitude": 5,
                "max_altitude_placeholder": example_max_altitude_placeholder,
                "query_templates": ["Answer in one sentence: {question}"],
            },
        ]
    }


@pytest.fixture()
def example_actions_config(example_actions_config_serialized):
    return ActionsConfig.from_dict(example_actions_config_serialized)


def test_serialize_actions_config(example_actions_config):
    actions_config_serialized = example_actions_config.to_dict()
    assert isinstance(actions_config_serialized["actions"], list)
    assert len(actions_config_serialized["actions"]) == 2


@pytest.fixture()
def example_new_nodes():
    return [
        ReasoningTreeNode(
            attributes={"num_children": 0, "altitude": 1},
            data={"role": "assistant", "content": example_thought_message},
        ),
        ReasoningTreeNode(
            attributes={"num_children": 0, "altitude": 2},
            data={"role": "assistant", "content": example_search_message},
        ),
        ReasoningTreeNode(
            attributes={"num_children": 0, "altitude": 3},
            data={
                "role": "assistant",
                "content": "After searching wikipedia, "
                "I found the following info about red pandas.",
            },
        ),
    ]


def test_get_action(example_actions_config):
    """
    Test retrieving an action given the name of the action.
    """
    for action_name in ["search", "jupyter"]:
        action_retrieved = example_actions_config.get_action_by_name(action_name)
        assert action_retrieved.name == action_name


def test_parse_action(example_actions_config):
    node_replacement = example_actions_config.parse_message(example_search_message, 1)
    assert node_replacement is not None
    node_replacement: ReasoningTreeNode

    action = node_replacement.attributes["action"]
    assert action["name"] == "search"
    assert action["payload"] == (example_payload,)

    assert node_replacement.data["content"] == example_search_match


def test_parse_action_overheight(example_actions_config):
    node_replacement = example_actions_config.parse_message(example_search_message, 7)
    assert node_replacement is not None
    node_replacement: ReasoningTreeNode

    action = node_replacement.attributes.get("action", {})
    assert action == {}

    assert node_replacement.data["content"] == example_max_altitude_placeholder


def test_parse_action_no_match(example_actions_config):
    assert example_actions_config.parse_message(example_thought_message, 1) is None


def test_process_action(example_actions_config, example_new_nodes):
    """
    Make sure non-action nodes before action nodes are copied over,
    action nodes are replaced, and subsequent nodes are not included.
    """
    new_nodes_processed = process_actions(example_new_nodes, example_actions_config)
    assert len(new_nodes_processed) == 2
    assert new_nodes_processed[0].data["content"] == example_thought_message
    assert new_nodes_processed[0].attributes["num_children"] == 0
    assert new_nodes_processed[1].data["content"] == example_search_match
    assert new_nodes_processed[1].attributes["action"]["name"] == "search"
    assert new_nodes_processed[1].attributes["action"]["payload"] == (example_payload,)
    assert new_nodes_processed[1].attributes["num_children"] == 2
