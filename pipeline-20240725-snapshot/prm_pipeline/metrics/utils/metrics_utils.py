"""
Utils for reporting metrics.

Utils for aggregating multiple rollouts
- Majority voting (get_majority_votes)
    - with equal weight
    - with filtering based on preference

Utils for reporting rollout accuracy
- Accuracy for a given ground truth label 
    and the correct prediction (get_category_metrics)
- Accuracy for select categories (get_categories, get_category_metrics)

Utils for parsing preferences
- Aggregate preferences for each step to obtain overall 
    preference for rollout
- Obtain area-under-curve between process reward labels scores and
    whether a rollout is correct.
"""

from collections import Counter, defaultdict
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from tqdm.auto import tqdm

from ...utils import (
    AggregationMethods,
    NodeId,
    PathElement,
    Prediction,
    ReasoningTree,
    ReasoningTreeNode,
    ValueBias,
    get_value_estimate,
)

from prm_pipeline.utils.tree_utils import ForestName

RolloutList = List[Dict[str, Any]]
PRMLabelList = List[Dict[str, Any]]
PreferenceList = List[float]
ProcessReward = List[float]
V = TypeVar("V")


def _get_rollout_example_id(rollout: Dict[str, Any]) -> Optional[Any]:
    """
    Given a rollout from the Rollout collection,
    retrieve unique identifier of the associated example.

    Return None if no such identifier is found.

    Params:
        rollout: Dict[str, Any], an element from the rollout collection.

    Returns:
        example identifier, None if not found.
    """
    example_id = rollout["example"].get("_index")
    if example_id is None:
        for key in ["index", "example_id", "id"]:
            example_id = rollout["example"].get(key)
            if example_id is not None:
                break

    return example_id


def _group_weighted_votes(
    rollouts: RolloutList, preferences: PreferenceList
) -> Dict[str, List[Dict[str, float]]]:
    """
    Each rollout is associated with a prediction.

    Given a list of rollouts and a matching list of preferences
    (weights), group (prediction, preference) by example_id.

    Params:
        rollouts: List[Dict], list of rollouts Dict from DB.
        preferences: List[float], length should be equal to
            that of rollouts.

    Returns:
        Dict mapping example id to a list of dictionaries with
        attributes:
            - ground_truth
            - prediction
            - weight
    """
    # Initialize default dictionary mapping example_id to
    # a list of dictionaries with attributes "prediction"
    # and "weight" (from preferences).
    example_to_votes = defaultdict(list)

    # For each (rollout, preference) pair:
    for rollout, pref in zip(rollouts, preferences):
        # Extract example id.
        example_id = _get_rollout_example_id(rollout)
        assert example_id is not None

        # Extract numerical value of prediction and ground truth.
        prediction = rollout["prediction"]
        ground_truth = rollout["ground_truth"]

        # Append prediction and preference to the defaultdict.
        example_to_votes[example_id].append(
            {"ground_truth": ground_truth, "prediction": prediction, "weight": pref}
        )

    return example_to_votes


def get_majority_votes(
    rollouts: RolloutList, preferences: PreferenceList
) -> Dict[str, int]:
    """
    Each "example" is associated with more than one rollouts.
    Examples are uniquely indexed with an "id" attribute.

    Given a list of rollouts, compute majority vote for each
    "example".

    If preference is specified, weight the majority vote
    with respect to the preferences.

    Params:
        rollouts: List[Dict], list of rollouts Dict from DB.
        preferences: length should be equal to that of rollouts.

    Returns:
        Dict mapping example id to majority vote prediction.
    """
    # Group predictions and preferences by example_id.
    example_to_votes = _group_weighted_votes(rollouts, preferences)

    # Initialize dictionary for storing output. The
    # dictionary maps example_id to weighted prediction.
    output = {}

    # For each example id and corresponding predictions
    # in the defaultdict:
    for example_id, predictions in example_to_votes.items():
        # Logic for computing weighted majority vote:
        # Initialize default dictionary of float "0.0",
        # mapping prediction int label to sum of weighted votes.
        _weighted_votes = defaultdict(lambda: 0.0)

        # For each predictions for this example:
        for pred in predictions:
            # Add the weight of the prediction to the float counter
            # of the prediction label specified in this prediction.
            _weighted_votes[pred["prediction"]] += pred["weight"]

        # In _weighted_votes, find the dictionary key
        # (prediction label) with the maximum value.
        majority_vote = max(_weighted_votes, key=_weighted_votes.get)  # type: ignore

        # Set this prediction label as the value of
        # the example_id in the output dictionary.
        output[example_id] = majority_vote

    return output


def _get_example_ids_by_category(
    rollouts: RolloutList, ground_truth_categories: List[int]
) -> List[str]:
    """
    Given a list of rollouts from DB where each rollout includes the associated
    "example", return a list of example_id where `ground_truth` matches
    the ground_truth_category specified.

    Params:
        rollouts: list of rollouts from DB
        ground_truth_category: list of ground truth labels,
            where each label has the same type as `ground_truth` in rollout DB.

    Returns:
        List of `example_id` strings.
    """

    matched_example_ids = set()
    for rollout in rollouts:
        example_id = _get_rollout_example_id(rollout)
        ground_truth = rollout["ground_truth"]

        if ground_truth in ground_truth_categories:
            matched_example_ids.add(example_id)

    return list(matched_example_ids)


def get_category_metrics(
    rollouts: RolloutList,
    preferences: Optional[PreferenceList],
    ground_truth_category: int,
    correct_predictions: List[int],
    skipped_predictions: List[Any] = [],
) -> Dict[str, float]:
    """
    Given a list of rollouts, report metrics for examples where `ground_truth`
    is in the specified ground_truth_labels (e.g., 5 for "pants_fire".)

    Params:
        rollouts: list of rollouts from DB
        preferences: list of (aggregated) preferences, optional.
            Defaults to 1.0 for all if set to None.
            Should be equal to rollouts in length.
        ground_truth_categories: label with the same type as
            `ground_truth` in the rollout DB.
        correct_predictions: list of predictions correct
            for this ground_truth_label.
        skipped_predictions: list of predictions not to be included
            in accuracy calculations. Note that this list
            should not overlap with correct_predictions.

    Returns:
        Dict[str, float] with keys
        - accuracy
        - accuracy_full  # Ignoring `skipped_predictions`
        - count
        - count_full  # Ignoring `skipped_predictions`
        - num_correct
        - num_incorrect
        - num_incorrect_full  # Ignoring `skipped_predictions`
    """
    # (preferences default to 1)
    weighted = True
    if preferences is None:
        weighted = False
        preferences = [1.0] * len(rollouts)

    # Retrieve a list example_id to include in this metric.
    ids_included: List[str] = _get_example_ids_by_category(
        rollouts, [ground_truth_category]
    )

    # Compute weighted majority votes for all example_ids.
    majority_vote_by_id: Dict[str, int] = get_majority_votes(rollouts, preferences)

    preference_auc_by_id = {}
    if weighted:
        preference_auc_by_id = get_process_reward_auc(
            rollouts, preferences, correct_predictions
        )

    # Initialize counter for all predictions that are not skipped
    total_predictions = 0

    # Initialize "full" counter for all predictions, ignoring `skipped_predictions`
    total_full_predictions = 0

    # Initialize counter for all predictions that are correct
    correct_count = 0

    # Initialize counter for all predictions that are correct,
    # ignoring `skipped_predictions`
    correct_full_count = 0

    # Initialize buffer for auc values.
    auc_values = []

    # For each id in ids_included, retrieve the `majority_vote`` by the id:
    for id_ in ids_included:
        majority_vote = majority_vote_by_id.get(id_)

        # Add to the counters based on whether `majority_vote` is in
        # `correct_predictions` or `skipped_predictions`
        if majority_vote in correct_predictions:
            correct_count += 1
            correct_full_count += 1

        if majority_vote not in skipped_predictions:
            total_predictions += 1

        total_full_predictions += 1

        # Add auc value to tally only if auc value is valid (not None)
        auc = preference_auc_by_id.get(id_)
        if auc is not None:
            auc_values.append(auc)

    # Report metrics as a dictionary.
    metrics = {
        "accuracy": correct_count / total_predictions if total_predictions else 0,
        # "accuracy_full": correct_full_count / total_full_predictions
        # if total_full_predictions
        # else 0,
        "count": total_predictions,
        # "count_full": total_full_predictions,
        "num_correct": correct_count,
        "num_incorrect": total_predictions - correct_count,
        # "num_incorrect_full": total_full_predictions - correct_count,
    }

    if weighted:
        metrics["preference_auc"] = (
            sum(auc_values) / len(auc_values) if auc_values else None
        )

    return metrics


def get_categories(rollouts: RolloutList) -> List[int]:
    """
    Return a list of all unique `ground_truth` values mentioned in the rollouts.

    Params:
        rollouts: rollout list from DB, where each entry includes
            the `ground_truth` attribute.

    Returns:
        list of unique `ground_truth` values with the same type as
            ones from the rollout DB.
    """
    # Create set to store unique ground_truth values.
    unique_values = set()

    # For each rollout:
    for rollout in rollouts:
        # Add ground_truth value to unique values.
        unique_values.add(rollout["ground_truth"])

    # Return results as a list.
    return list(unique_values)


def get_aggregated_preference(
    process_rewards: ProcessReward, aggregation_method: AggregationMethods
) -> float:
    """
    Given a list of process rewards, compute aggregated value of reward
    for the entire rollout based on the given `aggregation_method`.
    Any non-numerical values (e.g., None) would be replaced with 0.

    Params:
        process_rewards: List of float, one for each step of the rollout.
        aggregation_method: an Enum option from ..utils.AggregationMethods.

    Returns:
        float: aggregated value. If process_reward is empty, return (-1)
    """
    process_rewards_filtered = []
    for step_reward in process_rewards[:-1]:
        if not isinstance(step_reward, (float, int)):
            step_reward = -1

        if not 0.0 <= step_reward <= 1.0:
            continue

        process_rewards_filtered.append(step_reward)

    if len(process_rewards_filtered) == 0:
        return -1

    if aggregation_method is AggregationMethods.AVERAGE:
        return sum(process_rewards_filtered) / len(process_rewards_filtered)

    elif aggregation_method is AggregationMethods.MINIMUM:
        return min(process_rewards_filtered)

    elif aggregation_method is AggregationMethods.SUMMATION:
        return min(process_rewards_filtered)

    else:
        raise NotImplementedError(
            "requested aggregation_method {}"
            " is not implemented".format(aggregation_method)
        )


def get_aggregated_preferences(
    list_of_process_rewards: List[ProcessReward],
    aggregation_method: Optional[AggregationMethods],
) -> List[float]:
    """
    Wrapper around get_aggregated_preference
    for a list of process reward elements.

    Params:
        process_rewards: List of lists of float.
        aggregation_method: an Enum option from ..utils.AggregationMethods.

    Returns:
        List[float]: list of aggregated values. If process_reward is empty, return (-1)
            Returns a list of 1 if aggregation_method is None.
    """
    if not isinstance(list_of_process_rewards, list):
        raise ValueError("list_of_process_rewards: {}".format(list_of_process_rewards))

    if aggregation_method is None:
        return [1] * len(list_of_process_rewards)

    preferences = []
    for process_rewards in list_of_process_rewards:
        preference = get_aggregated_preference(process_rewards, aggregation_method)
        preferences.append(preference)

    return preferences


def split_rollout_rewards(
    process_reward_elements: PRMLabelList,
) -> Tuple[RolloutList, List[ProcessReward]]:
    """
    Split a list of process_reward_elements retrieved from DB
    into a paired list of rollouts and reward lists.
    - rollouts are from the `rollout` attribute.
    - process rewards are from the `prm_output` attribute.

    Params:
        process_reward_elements: PRMLabelList,
            from "process_reward_label" collection.

    Returns:
        RolloutList, List[ProcessReward]
        the two lists are of equal length.
    """

    # Initialize buffer for rollout list and process reward list.
    rollouts: List[Dict] = []
    process_rewards: List[List[float]] = []

    # For each process_reward_label element:
    for process_reward_element in process_reward_elements:
        # Retrieve embedded rollout element and process reward.
        rollout = process_reward_element["rollout"]
        prm_output = process_reward_element["prm_output"]

        # Append the rollout and process reward to the appropriate list.
        rollouts.append(rollout)
        process_rewards.append(prm_output)

    # Return buffers.
    return rollouts, process_rewards


def get_process_reward_auc(
    rollouts: RolloutList,
    aggregated_preferences: PreferenceList,
    correct_predictions: Optional[List[int]] = None,
) -> Dict[str, Optional[float]]:
    """
    Given a list of rollouts and a matching list of preferences,
    determine whether the preferences is a good prediction for
    rollout == ground_truth. Set AUC to -1 for examples where the
    value of rollout == ground_truth is the same across all
    rollouts.

    Params:
        rollouts: List[Dict]
        aggregated_preferences: List[float]
            `rollouts` and `aggregated_preferences` should be
            equal in length e.g., from split_rollout_rewards
        correct_predictions: Optional[List[int]], use
            correct_predictions instead of ground_truth from
            rollouts if specified.

    Returns:
        a dictionary mapping example_id to average auc value
        for rollouts of that example.
    """
    # Group predictions, ground truth values, and preferences by example_id
    examples_to_weighted_votes = _group_weighted_votes(rollouts, aggregated_preferences)

    # Initialize output for storing average auc value for each example
    auc_by_example_id: Dict[str, Optional[float]] = {}

    # For each example_id and associated (rollout, preference) pairs:
    for example_id, weighted_votes in examples_to_weighted_votes.items():
        # Initialize buffer for storing scores and labels for each example
        scores = []
        labels = []

        # For each (rollout, preference) pairs:
        for weighted_vote in weighted_votes:
            preference = weighted_vote["weight"]
            prediction = weighted_vote["prediction"]
            ground_truth = weighted_vote["ground_truth"]

            if correct_predictions is not None:
                is_correct = prediction in correct_predictions
            else:
                is_correct = prediction == ground_truth

            if prediction is not None:
                scores.append(preference)
                labels.append(is_correct)

        # Compute auc value.
        # Set AUC to None in case (rollout == ground_truth)
        # is the same across all rollouts of this example.
        if len(np.unique(labels)) <= 1:
            auc_score = None
        else:
            auc_score = float(roc_auc_score(labels, scores))

        # Add average auc value to the auc_by_example dictionary.
        auc_by_example_id[example_id] = auc_score

    return auc_by_example_id


def value_function_labelled(
    walk: List[PathElement],
    aggregation_method: AggregationMethods = AggregationMethods.AVERAGE,
    value_bias: ValueBias = ValueBias.AVERAGE,
) -> float:
    """
    Return the value estimate based on num_leaf_nodes and
    leaf_value_sum attributes of the nodes.

    Params:
        walk: output of reasoning_tree.get_walk.
        aggregation_method: aggregation method for the values
            along the walk.
        value_bias: whether to bias the value estimate.

    Returns:
        Aggregated value of the walk, or 0.0 if any node
        along the path is missing num_leaf_nodes.
    """
    values: List[float] = []
    for path_element in walk:
        node_attributes = path_element.node.attributes
        value_confidence_interval = get_value_estimate(
            num_leaf_nodes=node_attributes.get("num_leaf_nodes", 0),
            leaf_value_sum=node_attributes.get("leaf_value_sum", 0),
        )
        if value_confidence_interval is None:
            return 0.0

        if value_bias is ValueBias.AVERAGE:
            value = value_confidence_interval.mean
        elif value_bias is ValueBias.UPPER:
            value = value_confidence_interval.upper_bound
        elif value_bias is ValueBias.LOWER:
            value = value_confidence_interval.lower_bound
        else:
            raise ValueError("Invalid value_bias: {}".format(value_bias))

        values.append(value)

    aggregated_value = get_aggregated_preference(values, aggregation_method)
    return aggregated_value


class BeamSearchOutput(NamedTuple):
    popular_prediction: Optional[Prediction]
    popularity_count: int
    beams_selected: List[Tuple[float, NodeId]]


def get_tree_beam_search_prediction(
    reasoning_tree: ReasoningTree,
    value_function: Callable[[List[PathElement]], float],
    num_beams: int,
) -> BeamSearchOutput:
    """
    Given a reasoning tree, return the popular vote of the
    most highly-valued nodes reached via a beam-search.

    Params:
        reasoning_tree: In-memory reasoning tree instance.
        value_function: given a path from root node to a node
            in the reasoning tree, return the value of the top
            node.
        num_beams: maximum number of beams to track.

    Returns:
        BeamSearchOutput.
    """
    # Each beam is represented as a tuple: (value, top node)
    # Populate beams with the root node.
    beams: List[Tuple[float, NodeId]] = [(0.0, reasoning_tree.ROOT_NODE_ID)]

    # On each step of the search, replace each beam where the top node
    # is not yet a leaf node with the children of that top node.
    # Truncate the list and keep only the K nodes with the highest value.
    # Stop when the top node of all beams are leaf nodes.
    num_nodes = len(reasoning_tree.nodes.keys())
    for _ in range(num_nodes):
        new_beams = []
        is_all_leaf = True

        # If the top node of a beam is not a leaf node,
        # replace that beam with one beam for each child node.
        # Truncate the list of beams.
        for beam_value, top_node_id in beams:
            child_node_ids = reasoning_tree.edges.get(top_node_id, [])

            if len(child_node_ids) == 0:
                new_beams.append((beam_value, top_node_id))
            else:
                is_all_leaf = False
                for child_node_id in child_node_ids:
                    child_path = reasoning_tree.get_walk(child_node_id)
                    child_value = value_function(child_path)
                    new_beams.append((child_value, child_node_id))

        new_beams_truncated = sorted(new_beams, reverse=True)[:num_beams]
        beams = new_beams_truncated

        if is_all_leaf:
            break

    predictions: List[Prediction] = []
    # Extract valid prediction from the leaf nodes in beams.
    for _, node_id in beams:
        node = reasoning_tree.nodes[node_id]
        prediction: Optional[Prediction] = node.attributes.get("prediction")
        if prediction is not None:
            predictions.append(prediction)

    if len(predictions) == 0:
        return BeamSearchOutput(
            popular_prediction=None, popularity_count=num_beams, beams_selected=beams
        )

    # TODO: implement support for weighing predictions by value.
    prediction_counter = Counter(predictions)
    popular_prediction, popularity_count = prediction_counter.most_common(1)[0]

    return BeamSearchOutput(
        popular_prediction, popularity_count=popularity_count, beams_selected=beams
    )


def get_tree_accuracy(
    reasoning_tree: ReasoningTree, subset_index: int = 0, subset_denominator: int = 1
) -> Optional[float]:
    """
    Return average accuracy of all predictions (leaf nodes)
    in the given reasoning tree.

    Params:
        reasoning_tree: ReasoningTree with "ground_truth" in attributes.
        subset_index and subset_denominator: int, a leaf is selected only if
            (leaf index in tree) % subset_denominator
            == subset_index % subset_denominator

    Returns:
        float, representing average accuracy of predictions (leaf nodes) in
            this reasoning tree.
        None, if there are no leaf node in this reasoning tree with a
             "prediction" in the node attributes.

    Raises:
        KeyError if "ground_truth" is not in the tree attributes.
    """
    ground_truth = reasoning_tree.attributes["ground_truth"]

    num_predictions: int = 0
    num_correct: int = 0

    leaf_node_ids = reasoning_tree.get_leaf_node_ids()
    for index, leaf_node_id in enumerate(leaf_node_ids):
        node = reasoning_tree.nodes[leaf_node_id]
        prediction = node.attributes.get("prediction")

        if index % subset_denominator != (subset_index % subset_denominator):
            continue

        if prediction is None:
            continue

        num_predictions += 1

        if prediction == ground_truth:
            num_correct += 1

    if num_predictions == 0:
        return None

    return num_correct / num_predictions


def get_prediction_folded(
    reasoning_tree: ReasoningTree, fold_index: int = 0
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Select the prediction matching fold_index.
    If there are fewer than fold_index predictions in this tree,
    raise KeyError.

    Params:
        reasoning_tree: ReasoningTree.
        fold_index: among all leaf nodes of this tree, return the prediction
            of the leaf node with index fold_index.

    Returns:
        prediction in the specified fold, and model name.
        None if the selected leaf node does not include a prediction.

    """
    leaf_node_ids = reasoning_tree.get_leaf_node_ids()
    leaf_node_id_selected = leaf_node_ids[fold_index % len(leaf_node_ids)]
    leaf_node_selected = reasoning_tree.nodes[leaf_node_id_selected]

    numerical_match = re.search(
        r"\|.*(\d+)",
        leaf_node_selected.data["content"],
        re.DOTALL,
    )
    numerical_prediction = None
    if numerical_match is not None:
        numerical_prediction = numerical_match.groups()[0]

    return leaf_node_selected.attributes.get(
        "prediction",
        numerical_prediction,
    ), leaf_node_selected.attributes.get("model_name")


def get_f1_safe(labels: List, predictions: List) -> Optional[float]:
    """
    Return f1 score after verification and filtering.

    Params:
        labels: list of labels
        prediction: list of prediction

    Returns:
        f1_score float if there are exactly two possible values in labels.
        None otherwise.
    """
    label_options = set(labels)
    num_classes = len(label_options)
    if num_classes != 2:
        print(
            "num_classes is {} (not 2.) "
            "F1 score isn't available.".format(num_classes)
        )
        return None

    # Exclude pairs where prediction is not in the set of labels.
    predictions_filtered = []
    labels_filtered = []
    predictions_ignored = Counter()
    for label, prediction in zip(labels, predictions):
        if prediction in label_options:
            predictions_filtered.append(prediction)
            labels_filtered.append(label)
        else:
            predictions_ignored[prediction] += 1

    print(
        "The following prediction values were not in label_options "
        "and are ignored: {}".format(predictions_ignored)
    )
    macro_f1 = f1_score(labels_filtered, predictions_filtered, average="macro")

    return float(macro_f1)


def get_confidence_interval(
    sample_means: np.ndarray, confidence_level: float = 0.05
) -> str:
    """
    Obtain string describing confidence interval.

    Params:
        examples: list of observations.

    Returns:
        human-readable string describing confidence interval
    """
    t_score = student_t.ppf(1 - confidence_level / 2, df=len(sample_means) - 1)

    average_sample_mean = np.mean(sample_means)

    # standard error of average sample mean
    sem = np.std(sample_means, ddof=1)

    if np.isnan(sem) or np.isnan(average_sample_mean):
        return ""

    confidence_interval = (
        average_sample_mean - t_score * sem,
        average_sample_mean + t_score * sem,
    )

    return "{:.1f}% ± {:.1f}% ({:.5f}, {:.5f}, p={:.2f})".format(
        average_sample_mean.item() * 100,
        t_score * sem * 100,
        confidence_interval[0].item(),
        confidence_interval[1].item(),
        confidence_level,
    )


def get_majority_vote_accuracy_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    _none_prediction_placeholder: Optional[Any] = None,
) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
    """
    Return majority vote.
    Each row of labels and predictions is one example.

    Params:
        labels: (num_examples, num_folds) or (num_examples,),
            each row must contain exactly one unique value.
        predictions: (num_examples, num_folds)
        _none_prediction_placeholder: replace None in majority votes with this value.

    Return:
        accuracy and f1, both are float between 0.0 and 1.0, inclusive.
    """
    if len(labels) == 0:
        return None, None, None

    for label in labels:
        if label[0] is None:
            return -1, -1, None

    # insert num_folds axis if given (num_examples,)
    if len(labels.shape) == 1:
        labels = labels.reshape((-1, 1))

    assert len(predictions.shape) == 2  # (num_examples, num_folds)
    assert labels.shape[0] == predictions.shape[0]

    majority_predictions = []

    for label_row, prediction_row in zip(labels, predictions):
        # ensure consistency across labels for each example
        assert len(label_row) > 0
        assert len(set(label_row.tolist())) == 1, Counter(label_row.tolist())
        majority_vote = Counter(prediction_row.flatten().tolist()).most_common(1)[0][0]
        majority_vote = (
            majority_vote if majority_vote is not None else _none_prediction_placeholder
        )
        majority_predictions.append(majority_vote)

    majority_predictions_array = np.asarray(majority_predictions)
    accuracy = 100 * np.mean(labels[:, 0] == majority_predictions_array).item()
    f1 = get_f1_safe(labels[:, 0].flatten().tolist(), majority_predictions)
    if f1 is not None:
        f1 *= 100

    if not isinstance(majority_predictions_array[0], (np.number, float, int)):
        return -1, -1, None

    sklearn_report = classification_report(
        labels[:, 0], majority_predictions_array, output_dict=True
    )
    assert isinstance(sklearn_report, dict)

    sklearn_report_flattened = {}
    for k, v in sklearn_report.items():
        if not isinstance(v, dict):
            sklearn_report_flattened[(k,)] = v
            continue

        for k_nested, v_nested in v.items():
            sklearn_report_flattened[(k, k_nested)] = v_nested

    return accuracy, f1, sklearn_report


def _get_node_model_name(node: ReasoningTreeNode) -> Optional[str]:
    """
    Return model name for a given node.

    Params:
        node: ReasoningTreeNode.
    """

    model_name = node.attributes.get("model_name", node.data.get("model_name"))
    return model_name


def get_token_usage(node: ReasoningTreeNode) -> Dict[str, int]:
    """
    Retrieve token usage for a given node.

    Params:
        node: ReasoningTreeNode.

    Returns:
        dict mapping (model_name, token_type) to tally.
    """
    output: Dict[str, int] = {}

    for key_path in [
        ["num_tokens"],
        ["token_count"],
        "action_response.response.meta.billed_units".split("."),
    ]:
        usage_dict = _recursive_get(node.attributes, key_path, {})
        if (usage_dict is None) or (not isinstance(usage_dict, dict)):
            continue

        for key, value in usage_dict.items():
            if isinstance(value, int):
                output["{}/{}".format(_get_node_model_name(node), key)] = value

    return output


def get_tree_token_usage(
    tree: ReasoningTree,
    get_token_usage_fn: Callable[[ReasoningTreeNode], Dict[str, int]] = get_token_usage,
    split_by_is_correct: bool = False,
) -> "Counter[str]":
    """
    Given a reasoning tree, return a dictionary
    mapping token type (e.g., context, output)
    to a tally.

    Params:
        tree: ReasoningTree.
        get_token_usage_fn: (optional) Function to return the usage
            info for a given node.

    Returns:
        counter mapping token type to tally.
    """
    token_tally = Counter()

    for leaf_node_id in tree.get_leaf_node_ids():
        leaf_node = tree.nodes[leaf_node_id]
        is_correct = bool(leaf_node.attributes.get("leaf_value_sum", None))

        for path_element in tree.get_walk(leaf_node_id):
            token_tally += Counter(
                {
                    "{}/{}".format(is_correct, k) if split_by_is_correct else k: v
                    for k, v in get_token_usage_fn(path_element.node).items()
                }
            )

    return token_tally


def get_tree_correctness_stats(tree: ReasoningTree) -> "Counter[str]":
    """
    Given a reasoning tree, return a counter of num correct/incorrect.

    Params:
        tree: ReasoningTree.

    Returns:
        counter mapping token type to tally.
    """
    output = Counter()

    for leaf_node_id in tree.get_leaf_node_ids():
        leaf_node = tree.nodes[leaf_node_id]
        is_correct = str(bool(leaf_node.attributes.get("leaf_value_sum", None)))
        output[is_correct] += 1

    return output


def _recursive_get(
    input_dict: Dict[str, Any], key_path: List[str], placeholder: V
) -> V:
    """
    Recursively follow the given path of keys in the dictionary.
    As soon as key is not found, return
    """
    is_matched = True
    current_value = input_dict

    for key in key_path:
        if key in current_value:
            current_value = current_value[key]
        else:
            is_matched = False
            break

    if is_matched:
        return current_value  # type: ignore

    return placeholder
