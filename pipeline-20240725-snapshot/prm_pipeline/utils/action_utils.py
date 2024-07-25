"""
Utils for parsing rollout for actions. 
"""

import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from . import ReasoningTreeNode


class Action(NamedTuple):
    """
    Config describing one particular action.

    pattern: regular expression with capture groups.
    max_altitude: int, instead of invoking this action,
        nodes that match the pattern but exceeds max_altitude
        would be replaced with a new node with content
        max_altitude_placeholder.
    query_templates: list of templates with substitute placeholders
        for regexp capture groups from "pattern".
    response_role: messages from this action will be labeled with
        this particular role (e.g., "user" or "assistant".)
    """

    name: str
    pattern: str
    query_templates: Tuple[str,] = ("{}",)
    response_template: str = "{}"

    # set this from within the action function.
    response_role: str = "assistant"
    max_altitude: Optional[int] = None
    max_altitude_placeholder: Optional[str] = None
    max_altitude_placeholder_role: str = "assistant"

    def match_and_replace(
        self, message_content: str, altitude: int
    ) -> Optional[ReasoningTreeNode]:
        """
        Given a message and altitude info,
        replace the message if matched.

        Params:
            message_content: content of message to match.
            altitude: of the given node.

        Returns:
            new message node if matched.
            None, otherwise.
        """
        match = re.search(self.pattern, message_content)
        if match is None:
            return

        payload_arguments = []
        for query_template in self.query_templates:
            payload_argument = query_template.format_map(match.groupdict())
            payload_arguments.append(payload_argument)

        message_match = match.group(0)
        action_data = {"name": self.name, "payload": tuple(payload_arguments)}

        # Apply action only if within max_altitude.
        if (self.max_altitude is not None) and altitude > self.max_altitude:
            node_data = {
                "role": self.max_altitude_placeholder_role,
                "source": "max_altitude",
                "content": self.max_altitude_placeholder,
            }
            node_attributes = {
                "_action_debug": {
                    **action_data,
                    "altitude": altitude,
                    "max_altitude": self.max_altitude,
                },
            }
        else:
            node_data = {"role": "assistant", "content": message_match}
            node_attributes = {"action": action_data}

        return ReasoningTreeNode(attributes=node_attributes, data=node_data)


class ActionsConfig(NamedTuple):
    """
    Config describing actions (e.g., search) for a tree.
    """

    actions: List[Action] = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionsConfig":
        actions = [Action(**action_dict) for action_dict in data["actions"]]
        return ActionsConfig(actions)

    def to_dict(self) -> Dict[str, Any]:
        return {"actions": [action._asdict() for action in self.actions]}

    def get_action_by_name(self, action_name: str) -> Optional[Action]:
        """
        Retrieve action with the given name.

        Returns None if not found.
        """
        action_map = {action.name: action for action in self.actions}
        return action_map.get(action_name)

    def parse_message(
        self, message_content: str, altitude: int
    ) -> Optional[ReasoningTreeNode]:
        """
        Parse message for action matches.

        Params:
            message_content: message (string) to parse.

        Returns:
            Updated ReasoningTreeNode if found.
            Otherwise, return None.
        """
        for action in self.actions:
            updated_node = action.match_and_replace(message_content, altitude)
            if updated_node is not None:
                return updated_node


def process_actions(
    new_nodes: List[ReasoningTreeNode], actions_config: ActionsConfig
) -> List[ReasoningTreeNode]:
    """
    Parse new nodes in rollout_submit_payload for special
    nodes to create. If matched, edit the matched node and
    discard all subsequent nodes in the list of new nodes.

    Params:
        new_nodes: newly-created nodes to parse.
        actions_config: ActionConfig.

    Returns:
        updated list of new nodes.
    """
    updated_nodes = []

    # Keep copying nodes until an action is matched.
    # When that happens, replace the matched node with
    # an "action" node.
    for node in new_nodes:
        message = node.data["content"]
        altitude = node.attributes["altitude"]
        node_replacement = actions_config.parse_message(message, altitude)
        if node_replacement is not None:
            node_replacement = node_replacement._replace(
                attributes={
                    **node.attributes,
                    **node_replacement.attributes,
                    "num_children": max(2, node.attributes.get("num_children", 2)),
                }
            )
            updated_nodes.append(node_replacement)
            break
        else:
            updated_nodes.append(node)

    return updated_nodes
