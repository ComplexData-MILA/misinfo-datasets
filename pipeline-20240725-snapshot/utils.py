from collections import defaultdict
import os
import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    TypeVar,
    NamedTuple,
    Hashable,
)


import pandas as pd
import streamlit as st

import pymongo
from pymongo.database import Database

from prm_pipeline.utils.tree_utils import NodeId, ReasoningTree, PathElement
from prm_pipeline.utils.submodule_utils import get_commit_hash, has_uncommitted_changes


submodule_path = "prm_pipeline/_experiment_configs"


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
Index = TypeVar("Index", bound=Hashable)


def get_included_kv_pairs(dictionary: Dict[K, V], keys: List[K]) -> Dict[K, V]:
    """
    Return a subset of dictionary consisting only of keys in `keys`.

    Params:
        dictionary: source dictionary.
        keys: keys to select.

    Returns:
        Subset of the original dictionary.
    """
    output: Dict[K, V] = {}
    for key, value in dictionary.items():
        if key in keys:
            output[key] = value

    return output


@st.cache_data
def get_experiment_commit_hash(submodule_path: str = submodule_path) -> str:
    experiment_commit_hash: str = get_commit_hash(submodule_path)
    return experiment_commit_hash


@st.cache_data
def get_labeling_configs(
    config_file_name: str, submodule_path: str = submodule_path
) -> Dict[str, Any]:
    config_relative_path = "labeling_configs/{}.json".format(config_file_name)
    config_path = os.path.join(submodule_path, config_relative_path)
    if has_uncommitted_changes(submodule_path, config_relative_path):
        raise RuntimeError(
            "in the submodule {}, please git-commit "
            "pending changes in {} before continuing.".format(
                submodule_path, config_path
            )
        )

    with open(config_path, "r") as labeling_configs_file:
        labeling_configs = json.load(labeling_configs_file)

    return labeling_configs


@st.cache_resource
def get_mongo_db() -> Database:
    client = pymongo.MongoClient(os.environ["MONGO_DB_SERVER"])
    db = client[os.environ["MONGO_DB_NAME"]]
    return db


class Aggregation(NamedTuple):
    representative_row: Dict[str, Any]

    # (name_column_by_values, (aggregated_key, value))
    values: Dict[Any, Dict[str, Any]] = defaultdict(dict)

    def get_flattened_values(self):
        """
        Flatten values (name_column_by_values, (aggregated_key, value))
        into ({name_column_by_values}_{aggregated_key}, value))

        Returns:
            Dict[str, Any]
        """
        values_flattened: Dict[str, Any] = {}
        for name_column_value, aggregated_value_pairs in self.values.items():
            for aggregated_key, value in aggregated_value_pairs.items():
                new_column_name = "{}: {}".format(name_column_value, aggregated_key)
                values_flattened[new_column_name] = value

        return values_flattened


def aggregate_by_key(
    table: List[Dict[str, Any]],
    aggregate_by: Tuple[str, ...] = ("tree_id", "node_id"),
    name_column_by: str = "annotator",
    aggregated_keys: Tuple[str, ...] = ("attribute_value", "comment"),
) -> Dict[Any, Aggregation]:
    """
    Aggregate table by the given aggregate_by keys.

    Returns:
        {
            (aggregated_by_1, ..., aggregated_by_n): Aggregation(
                representative_row: {...}
                values: {
                    name_column_by_value: {
                        aggregated_key_1: value_1,
                        ...
                        aggregated_key_m: value_m,
                    },
                    ...
                }
            )
        }
    """
    # (aggregate_by, Aggregation)
    # e.g., (dataset_id, Aggregation)
    aggregations: Dict[Any, Aggregation] = {}

    for row in table:
        aggregate_by_value = tuple(row.get(key) for key in aggregate_by)
        name_column_value = row.get(name_column_by)
        row.pop(name_column_by, None)
        aggregation = aggregations.get(
            aggregate_by_value, Aggregation({}, defaultdict(dict))
        )

        for aggregated_key in aggregated_keys:
            value = row.get(aggregated_key)
            row.pop(aggregated_key, None)
            aggregation.values[name_column_value][aggregated_key] = value

        aggregation = aggregation._replace(
            representative_row=row, values=aggregation.values
        )
        aggregations[aggregate_by_value] = aggregation

    return aggregations


def render_aggregations(aggregations: Dict[Any, Aggregation]) -> List[Dict]:
    """
    Render aggregations to table format.
    Each column is for one unique name_column_value.
    """
    # collect unique column names for ordering
    name_column_values = set()
    for aggregation in aggregations.values():
        values_flattened = aggregation.get_flattened_values()
        name_column_values = name_column_values.union(values_flattened.keys())

    name_columns_ordering = list(name_column_values)
    output_table: List[Dict[Any, Any]] = []
    for aggregation in aggregations.values():
        output_row = {
            **aggregation.representative_row,
            **get_included_kv_pairs(
                aggregation.get_flattened_values(), name_columns_ordering
            ),
        }
        output_table.append(output_row)

    return output_table


def align_paths(
    paths: List[V], extractor_fn: Callable[[V], K], key_fn: Callable[[V], Index]
) -> Dict[K, Dict[Index, V]]:
    """
    Align paths by match of extractor_fn.

    Params:
        paths: list of path representations
        extractor_fn: map path representation to a hashable key to align by.
        key_fn: map path representation to a hashable key for output.

    Returns:
        Dictionary mapping key to dict mapping key to paths.
    """
    # (key, path)
    path_by_key: Dict[K, Dict[Index, V]] = defaultdict(dict)

    for path in paths:
        key = extractor_fn(path)
        index = key_fn(path)
        path_by_key[key][index] = path

    return path_by_key


Path = Tuple[ReasoningTree, NodeId, List[PathElement]]


def get_unique_paths(paths: List[Path]) -> List[Path]:
    """
    Given a list of paths, returned subset of paths
    where each reasoning_tree is included no more than one time.

    Params:
        paths: list of tuples, where the first element of each tuple
            is a ReasoningTree.

    Returns:
        paths, in same order.
    """

    output: List[Path] = []
    tree_ids_added = set()
    for path in paths:
        tree, _, _ = path
        tree_id = tree.attributes["tree_id"]
        if tree_id not in tree_ids_added:
            output.append(path)
            tree_ids_added.add(tree_id)

    return output


def get_viewer_url(forest_name: str, tree_id: int, node_id: str) -> str:
    """
    Return viewer URL for the given node.

    Params:
        forest_name: str
        tree_id: int
        node_id: str

    Returns:
        URL str.
    """
    api_base_url = os.environ["WEBSERVER_API_BASE"].rstrip("/")
    return "/".join(map(str, [api_base_url, forest_name, tree_id, node_id]))


def add_index_to_nested_dicts(
    source_data: Dict[str, Dict[str, Any]],
    index_column_name: str,
    should_transpose: bool = True,
) -> List[Dict[str, str]]:
    """
    Transpose nested dict table into a list of row dicts.

    Params:
        source_data: dict mapping attribute to dict mapping row index to value.
        index_column: name of row index to use in output table.

    Returns:
        list of dict, same set of keys with the addition of index_column_name.

    """
    output: List[Dict[str, str]] = []
    source_data_transposed: Dict[str, Dict[str, str]] = defaultdict(dict)
    if should_transpose:
        for attribute, values in source_data.items():
            for row_index, value in values.items():
                source_data_transposed[row_index][attribute] = value
    else:
        source_data_transposed = source_data

    for row_index, row in source_data_transposed.items():
        output.append({index_column_name: row_index, **row})

    return output


def sort_table(
    source_data: Dict[str, Dict[str, Any]],
    index_column_name: str,
    should_transpose: bool = True,
) -> pd.DataFrame:
    """
    Sort table. See add_index_to_nested_dicts for input patterns.
    """
    if len(source_data) == 0:
        return pd.DataFrame()

    return pd.DataFrame(
        add_index_to_nested_dicts(source_data, index_column_name, should_transpose)
    ).sort_values(index_column_name)


def filter_newlines(input_dict: Dict[str, V], placeholder: str = " ") -> Dict[str, V]:
    """
    Replace newlines from str values in input_dict with `placeholder`.
    """

    output: Dict[str, V] = {}
    for key, value in input_dict.items():
        if isinstance(value, str):
            value = value.replace("\n", placeholder)

        output[key] = value

    return output