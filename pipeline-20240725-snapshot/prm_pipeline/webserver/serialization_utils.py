"""
Utilities for generating JSON responses.
"""
from typing import Any, Dict, List, NamedTuple, Tuple

from bson import ObjectId

from ..utils.tree_utils import NodeId, ReasoningTreeNode


def serialize_id(input_item: Dict) -> Dict:
    """
    Return a copy of "item" where any bson.ObjectId is replaced
    with the string representation of its value.
    """
    output = {}
    for key, value in input_item.items():
        if isinstance(value, ObjectId):
            output[key] = str(value)

        else:
            output[key] = value

    return output


class RolloutTask(NamedTuple):
    """
    Represents a rollout task instance
    for both rollout and AI preference generation.
    """

    tree_id: int
    tree_attributes: Dict
    rollout_context: List[Dict[str, Any]]

    @property
    def parent_node_id(self) -> NodeId:
        """
        Return the node_id of the parent node.
        (Most recent node in the rollout context)
        """
        assert len(self.rollout_context) > 0
        most_recent_node = self.rollout_context[-1]
        return most_recent_node["node_id"]


class RolloutSubmission(NamedTuple):
    """
    Represents the value submitted to `POST /rollout_task`

    To instantiate from decoded JSON, use from_dict().
    """

    tree_id: int
    ancestor_node_id: str
    nodes: List[ReasoningTreeNode]

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "RolloutSubmission":
        """
        Load from dictionary.

        Parse nodes into a list of reasoning tree node items.
        """
        nodes = []
        for node_dict in payload["nodes"]:
            node = ReasoningTreeNode.from_dict(node_dict)
            nodes.append(node)

        payload_parsed = {**payload, "nodes": nodes}
        return RolloutSubmission(**payload_parsed)

    def to_dict(self) -> Dict[str, Any]:
        """
        Create JSON-serializable dictionary representation of self,
        where each node is serialized using node.to_dict()
        """
        payload = self._asdict()
        nodes_payload = [node.to_dict() for node in self.nodes]
        payload["nodes"] = nodes_payload

        return payload
