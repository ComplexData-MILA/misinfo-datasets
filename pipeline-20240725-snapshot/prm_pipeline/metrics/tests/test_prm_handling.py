"""
Tests for handling of process reward elements.
"""
import pytest

from prm_pipeline.metrics.utils.metrics_utils import (
    get_aggregated_preferences, get_process_reward_auc, split_rollout_rewards)
from prm_pipeline.utils import AggregationMethods


@pytest.fixture()
def process_rewards():
    return [
        {
            "rollout_id": 10,
            "rollout": {
                "example": {"id": 1110},
                "id": 10,
                "prediction": 0,
                "ground_truth": 0,
            },
            "prm_output": [1, 1, 1, 1, 1, 1, None],
            "id": 480,
        },
        {
            "rollout_id": 9,
            "rollout": {
                "example": {"id": 1509},
                "id": 9,
                "prediction": 0,
                "ground_truth": 1,
            },
            "prm_output": [1, 1, 1, 1, 1, 1, None, 1, None, 1],
            "id": 1250,
        },
        {
            "rollout_id": 11,
            "rollout": {
                "example": {"id": 1509},
                "id": 11,
                "prediction": 1,
                "ground_truth": 1,
            },
            "prm_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "id": 1250,
        },
    ]


def test_split_rollout_rewards(process_rewards):
    """
    Verify that split_rollout_rewards correctly splits
    rollout and process rewards into two separate lists.

    Ordering of elements is important.
    """
    rollouts, prm_labels = split_rollout_rewards(process_rewards)
    assert rollouts == [
        {"example": {"id": 1110}, "id": 10, "prediction": 0, "ground_truth": 0},
        {"example": {"id": 1509}, "id": 9, "prediction": 0, "ground_truth": 1},
        {"example": {"id": 1509}, "id": 11, "prediction": 1, "ground_truth": 1},
    ]
    assert prm_labels == [
        [1, 1, 1, 1, 1, 1, None],
        [1, 1, 1, 1, 1, 1, None, 1, None, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]


def test_get_auc_score(process_rewards):
    """
    Verify that auc score calculations are implemented correctly.
    """
    rollouts, preferences = split_rollout_rewards(process_rewards)
    aggregated_preferences = get_aggregated_preferences(
        preferences, AggregationMethods.AVERAGE
    )
    auc_by_example = get_process_reward_auc(rollouts, aggregated_preferences)
    assert auc_by_example == {1110: None, 1509: 1.0}
