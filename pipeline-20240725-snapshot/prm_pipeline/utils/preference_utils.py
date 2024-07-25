import datetime
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from .tree_utils import ForestName, NodeId, ReasoningTree, TreeId

PreferenceScore = float


class Preference(NamedTuple):
    """
    Nodes with higher preference scores are more preferrable.
    """

    node_ids: List[NodeId]
    preference_scores: Dict[NodeId, PreferenceScore]

    # object_id is set automatically when retrieved from DB.
    object_id: Optional[str] = None

    # Placeholder values; must be populated before submitting to DB.
    forest_name: Optional[ForestName] = None
    tree_id: Optional[TreeId] = None

    label_source: Optional[str] = None
    timestamp: Optional[datetime.datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Produce JSON-serializable dictionary representation of self.
        """
        data = self._asdict()
        for key, value in data.items():
            if (value is None) and (key != "object_id"):
                raise ValueError("attribute must not be None: {}".format(key))

        assert self.timestamp is not None
        data["timestamp"] = self.timestamp.isoformat()

        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Preference":
        """
        Instantiate Preference object from a JSON-serializable
        dictionary representation similar to the output of _asdict.

        Replaces timestamp with a datetime.datetime instance.
        """
        timestamp = datetime.datetime.fromisoformat(data["timestamp"])
        parsed_data = {
            **data,
            "timestamp": timestamp,
        }
        object_id = data.get("_id")
        if object_id is not None:
            parsed_data["object_id"] = str(object_id)
        else:
            parsed_data["object_id"] = None

        parsed_data.pop("_id", None)
        return Preference(**parsed_data)


def validate_preference(
    preference: Preference, reasoning_tree: ReasoningTree
) -> Optional[str]:
    """
    Verify preference for the following types of data errors:
    - node_ids is empty
    - Some node_id is missing in the "preference_score" dictionary
    - The value in the "preference" dictionary is negative for some node_id
    - The predecessor node_id of some node_id is different from
        others in the the given Preference instance.

    Params:
        preference: Preference instance.
        reasoning_tree: Reasoning tree referred to in the preference.

    Returns:
        String (error message) if invalid.
        Otherwise, return None.
    """

    num_node_ids = len(preference.node_ids)
    if num_node_ids <= 1:
        return "node_ids must include at least 2 elements; current: {}".format(
            num_node_ids
        )

    # all node_id in node_ids should also be in preference_scores, and vice versa
    num_node_ids = len(preference.node_ids)
    assert isinstance(preference.preference_scores, dict)
    num_preference_scores = len(preference.preference_scores.keys())
    if num_node_ids != num_preference_scores:
        return (
            "Length of node_ids {} != length of preference_scores {}: {} vs {}".format(
                num_node_ids,
                num_preference_scores,
                preference.node_ids,
                list(preference.preference_scores.keys()),
            )
        )

    # node_ids with missing preference score value.
    missing_node_ids: List[NodeId] = []
    for node_id in preference.node_ids:
        if node_id not in preference.preference_scores.keys():
            missing_node_ids.append(node_id)

    if len(missing_node_ids) > 0:
        return (
            "The following node_id(s) are in node_ids "
            "but not preference_scores: {}".format(missing_node_ids)
        )

    # preference scores should be non-negative.
    invalid_preference_scores: Dict[NodeId, Any] = {}
    for node_id, preference_score in preference.preference_scores.items():
        if not isinstance(preference_score, (float, int)):
            invalid_preference_scores[node_id] = str(type(preference_score))
        elif preference_score < 0:
            invalid_preference_scores[node_id] = preference_score

    if len(invalid_preference_scores.items()) > 0:
        return (
            "Preference scores must be non-negative float values. "
            "The following values are invalid: {}".format(invalid_preference_scores)
        )

    # all nodes must have the same predecessor.
    predecessor_node_ids = {
        node_id: reasoning_tree.edges_inverted[node_id]
        for node_id in preference.node_ids
    }

    distinct_predecessor_node_ids = set(predecessor_node_ids.values())
    if len(distinct_predecessor_node_ids) > 1:
        return (
            "All nodes inde node_ids must share the same predecessor. "
            "distinct_predecessor_node_ids: {}; predecessor_node_ids: {}".format(
                distinct_predecessor_node_ids, predecessor_node_ids
            )
        )


class _PairwisePreference(NamedTuple):
    """
    Preference between exactly two reasoning nodes.
    """

    more_preferred: NodeId
    less_preferred: NodeId


def get_pairwise_preferences(
    preference: Preference,
    criteria: Callable[
        [PreferenceScore, PreferenceScore], Optional[bool]
    ] = lambda score_1, score_2: score_1
    > score_2,
) -> List[_PairwisePreference]:
    """
    Generate a list of pairwise preferences given a Preference instance.

    Design choice: criteria is part of the experiment code and should be
        version controlled alongside other experiment config logic,
        separate from the production logic.

    Params:
        preference: Perference instance with N node_ids.
        criteria: (PreferenceScore, PreferenceScore) -> int?
            Given two preference scores, return True if the first one is
            to be preferred, False if the second one is preferred, or None
            if this pair is to be excluded.

    Returns:
        List of (1/2) * N * (N - 1) _PairwisePreference instances.
    """
    num_options = len(preference.node_ids)
    pairwise_preferences: List[_PairwisePreference] = []

    for index_0 in range(num_options):
        for index_1 in range(index_0 + 1, num_options):
            node_id_0 = preference.node_ids[index_0]
            node_id_1 = preference.node_ids[index_1]
            preference_score_0 = preference.preference_scores[node_id_0]
            preference_score_1 = preference.preference_scores[node_id_1]

            prefer_0 = criteria(preference_score_0, preference_score_1)

            if prefer_0 == True:
                pairwise_preferences.append(_PairwisePreference(node_id_0, node_id_1))
            elif prefer_0 == False:
                pairwise_preferences.append(_PairwisePreference(node_id_1, node_id_0))

    return pairwise_preferences
