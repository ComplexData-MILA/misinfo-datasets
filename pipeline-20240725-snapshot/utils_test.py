from typing import Any, Dict, List, Tuple

import pytest

from utils import aggregate_by_key, render_aggregations, align_paths


@pytest.fixture()
def example_table():
    return [
        {
            "annotator": 0,
            "tree_id": 1000,
            "attribute_value": True,
            "comment": "Comment 0",
        },
        {"annotator": 0, "tree_id": 1001, "attribute_value": True},
        {
            "annotator": 1,
            "tree_id": 1000,
            "attribute_value": False,
            "comment": "Comment 1",
        },
    ]


@pytest.fixture()
def example_paths():
    output: List[Tuple[Dict[str, Any]]] = [
        ({"tree_id": 0, "forest_name": "1-A"},),
        ({"tree_id": 0, "forest_name": "1-B"},),
        ({"tree_id": 1, "forest_name": "1-A"},),
    ]

    return output


def test_aggregate_by_key(example_table):
    aggregation_output = aggregate_by_key(example_table)
    print(aggregation_output)

    assert aggregation_output[(1000, None)].representative_row["tree_id"] == 1000
    assert "annotator" not in aggregation_output[(1000, None)].representative_row.keys()
    assert "comment" not in aggregation_output[(1000, None)].representative_row.keys()
    assert aggregation_output[(1000, None)].values == {
        0: {"attribute_value": True, "comment": "Comment 0"},
        1: {"attribute_value": False, "comment": "Comment 1"},
    }
    assert aggregation_output[(1001, None)].values == {
        0: {"attribute_value": True, "comment": None}
    }


def test_produce_aggregation_table(example_table):
    aggregation_output = aggregate_by_key(example_table)
    output_table = render_aggregations(aggregation_output)
    print(output_table)


def test_align_paths(example_paths):
    aligned_paths = align_paths(
        example_paths, lambda x: x[0]["tree_id"], lambda x: x[0]["forest_name"]
    )
    assert len(aligned_paths.keys()) == 2
    assert aligned_paths == {
        0: {
            "1-A": ({"tree_id": 0, "forest_name": "1-A"},),
            "1-B": ({"tree_id": 0, "forest_name": "1-B"},),
        },
        1: {"1-A": ({"tree_id": 1, "forest_name": "1-A"},)},
    }
