from functools import partial

import numpy as np
import pytest

from prm_pipeline.metrics.utils.metrics_utils import (
    _get_example_ids_by_category,
    get_aggregated_preferences,
    get_categories,
    get_category_metrics,
    get_majority_votes,
    get_prediction_folded,
    get_tree_accuracy,
    get_tree_beam_search_prediction,
    value_function_labelled,
    get_tree_token_usage,
    get_majority_vote_accuracy_f1,
)
from prm_pipeline.utils import AggregationMethods, ReasoningTree, ValueBias
from prm_pipeline.utils.tests.test_tree_utils import reasoning_tree


@pytest.fixture()
def rollouts():
    return [
        {
            "prediction": 0,
            "example": {"id": "0.json", "label": 1},
            "ground_truth": 1,
        },
        {
            "prediction": 0,
            "example": {"id": "0.json", "label": 1},
            "ground_truth": 1,
        },
        {
            "prediction": 1,
            "example": {"id": "0.json", "label": 1},
            "ground_truth": 1,
        },
        {
            "prediction": 0,
            "example": {"id": "8.json", "label": 0},
            "ground_truth": 0,
        },
        {
            "prediction": 1,
            "example": {"id": "9.json", "label": 0},
            "ground_truth": 0,
        },
    ]


@pytest.fixture()
def preferences():
    return [0.3, 0.3, 1.0, 0.7, 0.0]


def test_get_majority_votes_with_preference(rollouts, preferences):
    """
    Test get_majority_votes on example rollouts:
    - with example preference weights

    Output should be {"0.json": 1, "8.json": 0, "9.json": 1}
    """
    majority_votes_ref = {"0.json": 1, "8.json": 0, "9.json": 1}
    majority_votes = get_majority_votes(rollouts, preferences)

    assert majority_votes_ref == majority_votes


def test_get_example_ids_by_category(rollouts):
    """
    Test if _get_example_ids_by_category returns the correct
    example_ids when requested.

    Output should be ["8.json", "9.json"] for ground_truth 0.
    Ordering of items does not matter in this use case.
    """
    example_ids_ref = ["8.json", "9.json"]
    example_ids = _get_example_ids_by_category(rollouts, [0])
    assert set(example_ids) == set(example_ids_ref)

    example_ids_ref = ["8.json", "9.json"]
    example_ids = _get_example_ids_by_category(rollouts, [-1, 0])
    assert set(example_ids) == set(example_ids_ref)

    example_ids_ref = ["0.json", "8.json", "9.json"]
    example_ids = _get_example_ids_by_category(rollouts, [0, 1])
    assert set(example_ids) == set(example_ids_ref)


def test_get_category_metrics(rollouts, preferences):
    """
    Verify output from get_category_metrics.
    """
    metrics = get_category_metrics(
        rollouts,
        preferences,
        ground_truth_category=0,
        correct_predictions=[0],
        skipped_predictions=[1],
    )
    metrics_ref = {
        "accuracy": 1.0,
        # "accuracy_full": 1 / 2,
        "count": 1,
        # "count_full": 2,
        "num_correct": 1,
        "num_incorrect": 0,
        # "num_incorrect_full": 1,
        "preference_auc": None,
    }

    assert metrics == metrics_ref

    metrics = get_category_metrics(
        rollouts,
        preferences,
        ground_truth_category=0,
        correct_predictions=[1],
        skipped_predictions=[0],
    )
    metrics_ref = {
        "accuracy": 1,
        # "accuracy_full": 1 / 2,
        "count": 1,
        # "count_full": 2,
        "num_correct": 1,
        "num_incorrect": 0,
        # "num_incorrect_full": 1,
        "preference_auc": None,
    }

    assert metrics == metrics_ref


def test_get_categories(rollouts):
    """
    Verify that get_categories correctly retrieves
    a list of all `ground_truth` values mentioned
    in the rollouts. Ordering does not matter in
    this use case.
    """

    categories_ref = [0, 1]
    categories = get_categories(rollouts)

    assert set(categories) == set(categories_ref)


def test_get_aggregated_preference():
    """
    Test get_aggregated_preference on the preferences of a
    particular rollout.
    """
    example_step_preferences = [[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, None, 1.0]]
    assert get_aggregated_preferences(
        example_step_preferences, AggregationMethods.AVERAGE
    ) == [6 / 8]
    assert get_aggregated_preferences(
        example_step_preferences, AggregationMethods.MINIMUM
    ) == [0]


def test_beam_search_metric(reasoning_tree: ReasoningTree):
    _value_function = partial(
        value_function_labelled,
        aggregation_method=AggregationMethods.AVERAGE,
        value_bias=ValueBias.AVERAGE,
    )

    popular_prediction = get_tree_beam_search_prediction(
        reasoning_tree, _value_function, num_beams=2
    )

    assert popular_prediction.popular_prediction == 1
    assert popular_prediction.popularity_count == 2

    selected_top_node_ids = [
        top_node_id for value, top_node_id in popular_prediction.beams_selected
    ]
    assert set(selected_top_node_ids) == set(["5", "6"])


def test_get_tree_accuracy(reasoning_tree: ReasoningTree):
    """
    Test mean accuracy of leaf nodes of a given reasoning tree.
    """
    reasoning_tree.attributes["ground_truth"] = 1
    accuracy = get_tree_accuracy(reasoning_tree)
    assert accuracy == 0.75


def test_get_tree_prediction(reasoning_tree: ReasoningTree):
    """
    Test retrieving prediction of a specific fold of the tree.
    """
    predictions_collected = []
    for fold_index in range(len(reasoning_tree.nodes.keys())):
        try:
            prediction, model_name = get_prediction_folded(reasoning_tree, fold_index)
        except IndexError:
            break

        predictions_collected.append(prediction)

    assert len(predictions_collected) == len(reasoning_tree.get_leaf_node_ids())


def test_get_majority_vote_accuracy():
    """
    Test computing majority vote given label and prediction arrays.
    """

    # label: (num_examples, num_folds)
    labels = np.asarray([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    predictions = np.asarray([[0, 0, 1], [1, 0, 0], [0, 0, 1]])

    majority_vote_accuracy, _ = get_majority_vote_accuracy_f1(labels, predictions)
    assert majority_vote_accuracy == 100 * (2 / 3)

    # labels: (num_examples,)
    labels = labels[:, 0]
    assert len(labels.shape) == 1
    majority_vote_accuracy, _ = get_majority_vote_accuracy_f1(labels, predictions)
    assert majority_vote_accuracy == 100 * (2 / 3)


def test_get_token_usage(reasoning_tree: ReasoningTree):
    """
    Test aggregating usage info across tree.
    """
    reasoning_tree.nodes["1"].attributes["num_tokens"] = {
        "context": 7,
        "output": 9,
        "total": 16,
    }

    reasoning_tree.nodes["2"].attributes["num_tokens"] = {
        "context": 12,
        "output": 17,
        "total": 29,
    }

    tree_token_usage = get_tree_token_usage(reasoning_tree)
    assert dict(tree_token_usage) == {
        "context": 7 + 12,
        "output": 9 + 17,
        "total": 16 + 29,
    }
