"""
MongoDB-specific Utils
- methods for pagination through DB, keeping workers stateless 
- methods for retrieving reasoning trees from DB
- methods for uploading reasoning trees to DB
"""

from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import pymongo
import pymongo.collection
from flask_pymongo import PyMongo
from pymongo.database import Database

from tqdm.auto import tqdm

from ..utils.threading_utils import Cache
from ..utils.tree_utils import ForestName, ReasoningTree, ReasoningTreeIdTuple
from ..utils.preference_utils import Preference

mongo = PyMongo()

index_lock = Lock()


def get_next_index(db: Database, index_name: str) -> int:
    """
    Obtain a integer index for `index_name`.
    Initialize the index if not initialized.

    Params:
        db: Database
        index_name: one for each resource to index.

    Returns:
        int representing index
    """
    with index_lock:
        # Retrieve index and update counter in collection.
        result = db["index_collection"].find_one_and_update(
            {"_id": index_name},
            {"$inc": {"value": 1}},
        )

        if result is not None:
            previous_value = result["value"]
        else:
            # Create if not found.
            db["index_collection"].insert_one({"_id": index_name, "value": 1})
            previous_value = 0

    # Returns the original counter value before the update.
    return previous_value


class OffsetPointer:
    """
    Util for persisting to DB the state of pagination without
    having to specify the page in the request. Useful for keeping
    ai_preference and prm_inference state-less.

    Obtain the current position of the cursor for the
    specific index_name and job key. Create one if not
    found using an atomic upsert transaction.

    This cursor is not thread-safe and should be used
    with a threading Lock.

    ```python
    from threading import Lock

    rollout_cursor_lock = Lock()
    with rollout_cursor_lock:
        # Initialize cursor
        cursor = Cursor(db, "rollout", labelling_scheme_version)

        # Get results
        rollout_query = {"id": {"$gt": cursor.value}}
        rollout_matches = db.profiles.find(rollout_query).hint(
            [("id", pymongo.ASCENDING)]
        ).limit(batch_size)

        # Close and save updated cursor value.
        cursor.close(update=batch_size)
    ```

    For example, multiple workers of the same ai_preference job
    might be accessing the webserver simultaneously. The workers
    should share the same key unique to the job. Using this key,
    the webserver can obtain a shared cursor integer, so that it
    will update the cursor each time it receives a request from
    the worker. The workers will then receive different rollouts.

    """

    def __init__(self, db: Database, index_name: str, key: str, default: int = 0):
        """
        Params:
            db: Database
            index_name: type of the resource e.g., rollout
            key: job-specific key. Requests with the same job key
                will be shared the same index.
            default: default value of cursor in case none is found.
        """
        self.db = db
        self.id = "{}_{}".format(index_name, key)
        self.cursor = self.db["cursor_collection"].find_one({"_id": self.id})
        # If not found, use default value and create when closing.
        if self.cursor is not None:
            self.cursor_value = self.cursor.get("value", default)
        else:
            self.cursor_value = default

    @property
    def value(self):
        return self.cursor_value

    def close(self, new_value: int = 0):
        return self.db["cursor_collection"].find_one_and_update(
            {"_id": self.id}, {"$set": {"value": new_value}}, upsert=True
        )


# Threading lock for sequential_retrieve if the caller didn't
# specify one.
shared_lock = Lock()


def sequential_retrieve(
    db: Database,
    collection: str,
    key: str,
    limit: int,
    lock: Lock = shared_lock,
    additional_queries: Dict = {},
) -> List[Dict]:
    """
    Retrieve items sequentially, ensuring that concurrent requests
    with the same `key` will receive different values.

    Params:
        index_name: name of the resource, e.g., rollout.
        key: job-specific key.

    Returns:
        List[Dict]
    """
    # Atomically get cursor, retrieve entries, and update cursor.
    with lock:
        cursor = OffsetPointer(db, collection, key)
        item_query = {"id": {"$gte": cursor.value}, **additional_queries}
        item_matches = list(
            db[collection]
            .find(item_query)
            .hint([("id", pymongo.ASCENDING)])
            .limit(limit)
        )

        # New cursor value should be one plus the largest id
        # among all item values retrieved.
        item_ids = [item["id"] for item in item_matches]

        if len(item_ids) > 0:
            new_cursor_value = max(item_ids) + 1

        else:
            new_cursor_value = cursor.value

        cursor.close(new_value=new_cursor_value)

    return list(item_matches)


def filter_query_dict(
    source_dict: Dict[str, Any], key_whitelist: Set[str], int_keys: Set[str] = set()
) -> Dict[str, Any]:
    """
    Given a dictionary, copy over only the keys that are
    in key_whitelist. Values that are empty strings will
    not be included in the output. Values that are also
    in int_keys will be converted into int.

    Params:
        source_dict: Dictionary to copy from.
        key_whitelist: Set of keys to copy.
        int_keys: Optional, keys to convert from string to
           int.

    Returns:
        dictionary.
    """
    output = {}
    for key in key_whitelist:
        if key in source_dict.keys():
            value = source_dict[key]
            if value != "":
                output[key] = value

    for key in output.keys():
        if key in int_keys:
            output[key] = int(output[key])

    return output


class MongoReasoningTreeDB:
    """
    Provides the following through Mongo:
    - reasoning_tree_cache
    - get_forest_method
    - upload_tree_method
    """

    _TREES_MONGO_COLLECTION_NAME = "reasoning_trees"
    _PREFERENCES_MONGO_COLLECTION_NAME = "preferences"

    def __init__(self, database: Database, cache_capacity: int):
        self._trees_mongo_collection = database[self._TREES_MONGO_COLLECTION_NAME]
        self._preferences_mongo_collection = database[
            self._PREFERENCES_MONGO_COLLECTION_NAME
        ]

        # Threading locks to limit number of concurrent uploads for
        # each tree_id tuple.
        self.trees_in_flight: Set[ReasoningTreeIdTuple] = set()
        self._trees_in_flight_lock = Lock()

        # Consumers of this class should retrieve trees through the cache,
        # not directly through _get_tree.
        self.cache = Cache(cache_capacity, self._get_tree)

    @staticmethod
    def _get_tree_query(tree_id_tuple: ReasoningTreeIdTuple) -> Dict:
        """
        Return tree query dictionary given tree_id_tuple.

        Params:
            tree_id_tuple: (forest_name, tree_id)

        Returns:
            filter dict for Mongo queries
        """
        # Deconstruct tree_id_tuple into forest_name and tree_id.
        forest_name, tree_id = tree_id_tuple
        query = {"forest_name": forest_name, "tree_id": tree_id}
        return query

    @staticmethod
    def _get_forest_query(
        forest_name: ForestName,
        data_source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return query dictionary for all trees in the given forest.

        Params:
            forest_name: ForestName
            data_source_name: Optionally, specify name of dataset source/subset.

        Returns:
            filter dict for Mongo queries
        """
        query = {"forest_name": forest_name}
        if data_source_name is not None:
            query["data.attributes.source.source"] = data_source_name

        return query

    @staticmethod
    def _extract_tree(database_entry) -> ReasoningTree:
        """
        Given database_response to find_one, return parsed reasoning tree.
        """
        # Extract JSON-friendly tree data
        # and reconstruct tree from the dictionary.
        tree_data = database_entry["data"]
        tree = ReasoningTree.from_dict(tree_data)
        return tree

    @staticmethod
    def _serialize_tree(
        tree_id_tuple: ReasoningTreeIdTuple, tree: ReasoningTree
    ) -> Dict:
        """
        Utils for serializing tree into an entry for insertion
        into Mongo.

        Params:
            tree_id_tuple: (forest_name, tree_id)
            tree: ReasoningTree

        Returns:
            dictionary.
        """
        forest_name, tree_id = tree_id_tuple

        # Obtain JSON-friendly representation of the tree.
        # Info about the tree is up to date up until this point.
        tree_serializable = tree.to_dict()
        database_entry = {
            "forest_name": forest_name,
            "tree_id": tree_id,
            "data": tree_serializable,
        }
        return database_entry

    def _get_tree(self, tree_id_tuple: ReasoningTreeIdTuple) -> Optional[ReasoningTree]:
        """
        Utils for retrieving tree from Mongo database.
        For performance reasons, use the cache instead
        of invoking this method directly.
        ```python
        self.reasoning_tree_cache.get(tree_id_tuple)
        ```

        Params:
            tree_id_tuple: (forest_name, tree_id)

        Returns:
            ReasoningTree if found.
            None if tree is not found in the given collection.
        """

        # Construct and execute Mongo query.
        # Since (forest_name, tree_id) pairs are unique,
        # returning only one element should be sufficient.
        query = self._get_tree_query(tree_id_tuple)
        response = self._trees_mongo_collection.find_one(query)

        # Return None if no tree matches requirements.
        if response is None:
            return

        tree = self._extract_tree(response)
        return tree

    def upload_tree(self, tree_id_tuple: ReasoningTreeIdTuple, tree: ReasoningTree):
        """
        Upload tree to Mongo database.

        Handles concurrency- at any given time,
        only one thread should be uploading a given tree
        to the database.

        If a tree is already being uploaded (in-flight,)
        all other requests to upload the same tree id tuple
        would return directly. Refer to comments in
        ForestTraversalManager for reasoning.

        Params:
            tree_id_tuple: (forest_name, tree_id)
            tree: ReasoningTree instance.
        """

        with self._trees_in_flight_lock:
            # If tree_id_tuple is already in-flight, return directly.
            if tree_id_tuple in self.trees_in_flight:
                return
            else:
                # Otherwise, add tree_id_tuple to the in-flight set.
                self.trees_in_flight.add(tree_id_tuple)

        # Construct Mongo payload.
        query = self._get_tree_query(tree_id_tuple)
        database_entry = self._serialize_tree(tree_id_tuple, tree)

        self._trees_mongo_collection.find_one_and_update(
            query, {"$set": database_entry}, upsert=True
        )

        with self._trees_in_flight_lock:
            self.trees_in_flight.remove(tree_id_tuple)

    def get_forest(
        self,
        forest_name: ForestName,
        data_source_name: Optional[str] = None,
    ) -> List[ReasoningTree]:
        """
        Retrieve a list of all the trees in the given forest.

        TODO: replace with an iterator for memory efficiency.
        (MVP: ai-preference)

        IMPORTANT: if no tree is found with this forest_name attribute,
        it is assumed that the forest doesn't exist, and this method
        would return an empty list instead of None.

        Params:
            forest_name: ForestName
            data_source_name: Optionally, specify name of dataset source/subset.

        Returns:
            List[ReasoningTree], might be an empty list.
        """
        # Build and execute query for all trees with this forest name.
        query = self._get_forest_query(forest_name, data_source_name)
        matches = self._trees_mongo_collection.find(query)
        num_matches = self._trees_mongo_collection.count_documents(query)

        # Initialize buffer for storing output trees.
        trees: List[ReasoningTree] = []

        # Iterate through matches, creating each match into a tree.
        for database_entry in tqdm(matches, total=num_matches, ncols=75, desc="trees"):
            tree = self._extract_tree(database_entry)
            trees.append(tree)

        # Return list of trees, even if the list is empty.
        return trees

    def get_preferences(self, query: Dict[str, Any] = {}) -> List[Preference]:
        """
        Retrieve a list of preferences matching the given criteria.

        Params:
            forest_name: ForestName
            query: Dict[str, Any] to be included in the query filter.
        """
        matches = self._preferences_mongo_collection.find(query)
        num_documents = self._preferences_mongo_collection.count_documents(query)

        preferences: List[Preference] = []
        for database_entry in tqdm(
            matches, total=num_documents, ncols=75, desc="preferences"
        ):
            database_entry["_id"] = str(database_entry["_id"])
            preference = Preference.from_dict(database_entry)
            preferences.append(preference)

        return preferences

    def upload_preference(self, preference: Preference) -> str:
        """
        Add a new preference instance to DB.

        Params:
            preference: Preference instance

        Returns:
            object_id.
        """
        data = preference.to_dict()
        response = self._preferences_mongo_collection.insert_one(data)
        return str(response.inserted_id)
