from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Optional, Tuple

from prm_pipeline.utils import (
    ActionsConfig,
    NodeId,
    PathElement,
    Preference,
    PreferenceScore,
)


class RolloutWorkerConfig(NamedTuple):
    rollout_delimiter_pattern: str
    prediction_pattern: str

    # Maximum altitude beyond which no children should be added.
    max_branching_altitude: int

    # Maximum number of rollouts retries (including initial one)
    # permitted per task when prediction_pattern isn't matched.
    max_num_retries: int

    # Default number of children per node
    max_num_children: int

    # Additional kwargs for openai.Completion.create
    llm_completion_kwargs: Dict[str, Any] = {}


class RolloutConfig:
    # Number of children for the root node.
    root_num_children: int = 2

    forest_name_suffix: str = "example-dataset"

    actions_config: ActionsConfig = ActionsConfig(actions=[])

    rollout_worker_config: RolloutWorkerConfig = RolloutWorkerConfig(
        rollout_delimiter_pattern=r"Thought: \d+",
        prediction_pattern=r": \(([0-9])\)",
        max_branching_altitude=10,
        max_num_retries=10,
        max_num_children=2,
    )

    # Information about this forest configuration.
    # For the convenience of human labellers.
    forest_info: Dict[str, Any] = {"prompt_template": ""}
    dataset_feature_names: List[str] = [""]

    @staticmethod
    def get_dataset_iterator(split: str) -> Iterator[Mapping]:
        """
        Handles data loading. For example: download data using the
        HuggingFace `datasets` library and specify dataset-specific
        configs.

        Params:
            split: name of dataset split.

        Returns:
            an iterator of mappings from column name to value.
            e.g., a list of dictionaries, one for each data item.
        """
        raise NotImplementedError()

    @staticmethod
    def serialize_dataset_item(dataset_entry: Mapping) -> Dict[str, Any]:
        """
        Create a JSON-serializable representation of one item
        from the dataset (e.g., for reference in the tree attributes.)

        Params:
            dataset_entry: Mapping, see get_dataset_iterator.

        Returns:
            JSON-serializable representation of the given
            dataset item.
        """
        return dict(dataset_entry)

    @staticmethod
    def get_root_prompt(dataset_entry: Mapping) -> str:
        """
        Return the question to the LLM about the given item
        from the dataset.

        Params:
            dataset_entry: Mapping, see get_dataset_iterator.

        Returns:
            a string denoting the initial instruction for the language model
        """
        raise NotImplementedError()

    @staticmethod
    def get_label(dataset_entry: Mapping) -> Optional[Any]:
        """
        If applicable, return the ground truth label for this
        particular item in the dataset. Note that some datasets
        are unlabelled- return None in these cases.

        Params:
            dataset_entry: Mapping, see get_dataset_iterator.

        Returns:
            A comparable label for the dataset item, or
            None if not applicable for this task.
        """
        return None


class ValueEstimateConfig:
    """
    Version-tracking for value estimation heuristics.
    """

    label_source_prefix: str = ""

    @staticmethod
    def get_preferences(path: List[PathElement]) -> List[Preference]:
        """
        Return relative preferences between the top node of the given path
        and the sibling nodes of that top node.

        The top node does not need to be a leaf.

        The caller is responsible for adding info such as forest_name and tree_id
        into each Preference.

        Params:
            path: Path to the given node, return value of tree.get_walk.
                Includes siblings at each level, particularly at the top level.
                See prm_pipeline.utils.tree_utils.

        Returns:
            list of one or more Preference instances.
        """
        # Example implementation:
        # set preference score of each node to the value estimate, and
        # skip nodes where no alternative options are available.

        siblings = path[-1].siblings
        if len(siblings) <= 1:
            return []

        preference_node_ids: List[NodeId] = []
        preference_scores: Dict[NodeId, PreferenceScore] = {}

        for sibling in siblings:
            if sibling.value_estimate is not None:
                node_id = sibling.attributes["node_id"]
                preference_node_ids.append(node_id)
                preference_scores[node_id] = sibling.value_estimate.mean

        return [
            Preference(
                node_ids=preference_node_ids, preference_scores=preference_scores
            )
        ]

    @staticmethod
    def pairwise_preference_criteria(
        score_0: PreferenceScore, score_1: PreferenceScore
    ) -> Optional[bool]:
        """
        Given two preference scores, return True if the first one is
        to be preferred, False if the second one is preferred, or None
        if this pair is to be excluded.

        See prm_pipeline.utils.preference_utils.get_pairwise_preferences

        Params:
            score_0: PreferenceScore
            score_1: PreferenceScore

        Returns:
            True if score_0 is to be preferred,
            False if score_1 is to be preferred, or
            None if this pair is to be excluded.
        """
        if abs(score_0 - score_1) > 0.5:
            return score_0 > score_1

    @staticmethod
    def preference_dataset_quality_filter(dataset_entry: Mapping) -> bool:
        """
        Given an entry proposed for the synthetic preference dataset,
        return whether this entry should be included in the dataset.

        Params:
            dataset_entry: a row proposed for the preference dataset.

        Returns:
            True if this row should be added.
            Otherwise, False.
        """
        # Example: return True only if "accepted" differs from "rejected".
        return dataset_entry["accepted"] != dataset_entry["rejected"]
