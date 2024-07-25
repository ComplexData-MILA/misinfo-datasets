import datetime
import os

import pymongo
import pytest

from prm_pipeline.metrics.utils.db_utils import (get_query_filename,
                                                 load_cache, write_cache)

from .test_rollout_metrics import rollouts


@pytest.fixture(scope="session")
def db():
    client = pymongo.MongoClient(os.environ.get("MONGO_DB_SERVER"))
    db = client[os.environ["MONGO_DB_NAME"]]

    return db


def test_get_cache_filename():
    """
    Verify that the filename created is path-save (no slashes)
    and deterministic.
    """
    example_query = {
        "model_identifier": "lmsys/vicuna-13b-v1.5",
        "dataset_name": "MATH",
    }

    # Verify path safety
    example_cache_path = get_query_filename(example_query)
    assert "/" not in example_cache_path

    # Verify determinism
    example_query_alt = dict(**example_query)
    example_cache_path_alt = get_query_filename(example_query_alt)
    assert example_cache_path == example_cache_path_alt


def test_write_load_cache(rollouts):
    """
    Test writing and loading the "rollout" list from
    the other test file and verify equivalence.
    """
    cache_folder = "/tmp"

    # Ensure uniqueness of indexing query.
    indexing_query = {"timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}

    # Verify that cache is None when the cache isn't found.
    cached_value = load_cache("testing", indexing_query, cache_folder)
    assert cached_value is None

    # Cache example element to disk.
    write_cache(rollouts, "testing", indexing_query, cache_folder)

    # Load from cache again and compare with previous value in memory.
    cached_value = load_cache("testing", indexing_query, cache_folder)
    assert cached_value == rollouts
