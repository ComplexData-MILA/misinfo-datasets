"""
Utils for handling database access and caching.

- Retrieve from database given filter.

- Save results to cache.

- Load from cache if matched.

- Generate path-safe filename for caches.
"""
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

from tqdm.auto import tqdm

EXCLUDED_ATTRIBUTES = ["_id"]


logger = logging.getLogger(__name__)


def get_filename(key: str) -> str:
    """
    Generate path-safe filenames.

    Params:
        key: unsafe unique identifier to encode.
    """
    filename = hashlib.sha256(key.encode()).hexdigest()[:8]
    return filename


def get_query_filename(query: Dict) -> str:
    """
    Generate path-safe filename from JSON-serializable query.

    Params:
        query: should be JSON-serializable.
    """
    # Serialize query as JSON string.
    query_serialized = json.dumps(query)

    # Get path-safe representation for query as filename.
    filename = get_filename(query_serialized)

    return filename


def load_cache(collection_name: str, query: Dict, cache_folder: str) -> Optional[List]:
    """
    Try loading JSON-serialized cache from the given path.
    Returns None if not found.

    Params:
        collection_name: name for the cache sub-folder, should be path-safe.
        query: JSON-serializable unique identifier for the cache.
        cache_folder: folder for cache sub-folders.
    """
    # Get filename for this query.
    filename = get_query_filename(query)

    # Join to obtain path to the query JSON file.
    cache_path = os.path.join(cache_folder, collection_name, filename)

    # Check if the cache JSON file exists.
    # Return None if the path does not exist.
    if not os.path.exists(cache_path):
        return None

    # Load cache JSON file.
    # The JSON library might throw an exception if cache file is
    # invalid. Catch such exceptions and provide path info
    # for debugging.
    with open(cache_path, "r") as cache_file:
        try:
            cache = json.load(cache_file)
        except json.decoder.JSONDecodeError as e:
            error_message = "Error loading JSON cache file {}".format(cache_path)
            raise json.decoder.JSONDecodeError(error_message, e.doc, e.pos)

    return cache


def write_cache(elements, collection_name: str, query: Dict, cache_folder: str):
    """
    Write JSON-serialized object to cache.
    Create cache folder and any sub-folder if needed.

    Params:
        elements: JSON-serializable elements to write to cache.
        collection_name: name for the cache sub-folder, should be path-safe.
        query: JSON-serializable unique identifier for the cache.
        cache_folder: folder for cache sub-folders.
    """
    # Get filename for this "query".
    filename = get_query_filename(query)

    # Get path to the cache sub-folder for this JSON file.
    cache_subfolder_path = os.path.join(cache_folder, collection_name)
    cache_file_path = os.path.join(cache_subfolder_path, filename)

    # Create folders if needed.
    os.makedirs(cache_subfolder_path, exist_ok=True)

    # Save JSON-serialized object to the path.
    with open(cache_file_path, "w") as cache_file:
        json.dump(elements, cache_file, indent=2)


def retrieve(
    db,
    collection_name: str,
    query: Dict,
    cache_folder: str,
    reset_cache: bool = False,
    max_count: Optional[int] = None,
) -> List[Dict]:
    """
    Retrieve data from DB.
    Attributes that are not JSON-serializable
    (see EXCLUDED_ATTRIBUTES) will not be included.

    Use local cache if possible unless reset_cache is set to True.
    Note that even if reset_cache is set to True, the cache won't
    be updated until the retrieval is complete.

    Params:
        collection_name: str, collection of database to retrieve from.
        query: Dict, filter for the query.
        cache_folder: str, folder holding the cache JSON files.
        reset_cache: bool, whether to discard cache.
        max_count: Optional[int], maximum number of items to retrieve from DB.

    Returns:
        List of retrieved item dictionaries.
    """
    # Create uniquely-identifying query with "limit" included.
    indexing_query = {**query, "limit": max_count, "db": db.name}
    if not reset_cache:
        cached_value = load_cache(collection_name, indexing_query, cache_folder)

        if cached_value is not None:
            logger.warning(
                "Reusing {} {} element(s) from cache.".format(
                    len(cached_value), collection_name
                )
            )
            return cached_value

    db_results = db[collection_name].find(query)
    db_count = db.rollout.estimated_document_count(query)
    if (max_count is not None) and (max_count > 0):
        db_results = db_results.limit(max_count)
        db_count = min(db_count, max_count)

    # Save rollouts in memory.
    elements_retrieved = []
    for element in tqdm(db_results, ncols=75, total=db_count):
        # Delete attributes that are not serializable, e.g., `_id`.
        for attribute in EXCLUDED_ATTRIBUTES:
            element.pop(attribute)

        elements_retrieved.append(element)

    # Update cache.
    write_cache(elements_retrieved, collection_name, indexing_query, cache_folder)

    return elements_retrieved
