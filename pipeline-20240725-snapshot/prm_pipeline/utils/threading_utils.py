import collections
import typing
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable, Dict, Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class RateLimitedThreadPoolExecutor(ThreadPoolExecutor):
    """
    Similar to thread pool executor, with the addition of a semaphore as
    a rate limit on the comined number of workers across all instances.
    """

    def __init__(self, max_workers, semaphore, *args, **kwargs):
        super().__init__(max_workers, *args, **kwargs)
        self.global_semaphore = semaphore

    def submit(self, fn, *args, **kwargs):
        self.global_semaphore.acquire()
        future = super().submit(fn, *args, **kwargs)
        future.add_done_callback(lambda x: self.global_semaphore.release())
        return future


class Cache(Generic[K, V]):
    """
    Cache for database items.

    Capacity is in terms of number of items that could be stored.
    Set to None for no limit.

    Retrieve forest traversal coordinator for the given
    (forest, job) pair. Create a new one if not found.

    Cache handling: after obtaining a forest traversal coordinator,
    add the traversal coordinator to the local dictionary.

    Concurrency handling: for each key that isn't in cache
    at most one thread may invoke get_item_uncached. All other
    threads requesting the same key should wait. Implementation:
    dictionary of locks, one for each key. See:
    chat.openai.com/share/6895aa47-a5dc-4a62-98a2-61a02a51aa7b

    Params:
        capacity: int, max number of keys to store.
            Set to None tonot enforce this limit.

            TODO: Limit cache capacity by memory usage,
            not by number of items.

        get_item_uncached: method for retrieving an item from
            upstream in case of a cache miss.

    IMPORTANT: there is no "update" method in this class since the
    assumption is that data within cache objects are writeable in a
    thread-safe way.
    """

    def __init__(
        self,
        capacity: Optional[int],
        get_item_uncached: Callable[[K], Optional[V]],
    ):
        self.cache: typing.OrderedDict[K, V] = collections.OrderedDict()
        self.cache_dict_lock = Lock()

        self.cache_locks: Dict[K, Lock] = {}
        self.cache_locks_lock = Lock()

        self.capacity = capacity
        self._get_item_uncached = get_item_uncached

    def _get_item_uncached_concurrent(self, key: K) -> Optional[V]:
        """
        Thread-safe wrapper around _get_item_uncached.
        Adds retrieved item to cache before returning the retrieved
        item.

        For each key, only one thread would invoke
        _get_item_uncached(key). All other threads requesting the
        same key should wait until that thread exit. This limit
        is specific to each key.

        Note that if a key is not found, other threads would invoke
        _get_item_uncached again instead of directly returning None.
        """
        # Threads compete in placing a lock in self.cache_locks.
        with self.cache_locks_lock:
            k_lock = self.cache_locks.get(key, Lock())

        with k_lock:
            # After getting lock on this key,
            # check again if the item has been added to
            # local cache before retrieving from upstream.
            if key in self.cache.keys():
                value = self.cache[key]
            else:
                value = self._get_item_uncached(key)

                # Store item in cache only when an item is
                # retrieved, not when no item was found.
                if value is not None:
                    with self.cache_dict_lock:
                        self.cache[key] = value

        # Remove lock for this key.
        #
        # For example, Assume that thread A is the first to place
        # the lock in cache_locks for this key, and thread B entered
        # while thread A is fetching the value from upstream.
        #
        # If thread C entered after thread A deleted
        # the lock from cache_locks while thread B is still holding
        # the lock, B and C would hold a different lock. The
        # result is still correct since thread A has already stored
        # the value in the cache.
        with self.cache_locks_lock:
            # Another thread might have poped this lock already,
            # so the default value is necessary.
            self.cache_locks.pop(key, None)

        return value

    def get(self, key: K) -> Optional[V]:
        """
        Retrieve an item from local cache.
        If no such item exists, fetch the item from upstream
        using _get_item_uncached method provided in __init__.

        Params:
            key: (forest_id: str, tree_id: str)

        Returns:
            Value if retrieved,
            either from local cache or from upstream.
            Return None if upstream returned None.
        """
        with self.cache_dict_lock:
            # Try retrieving from local cache.
            value = self.cache.get(key)

            # If tree is found, be sure to mark the key
            # as recently used.
            if value is not None:
                self.cache.move_to_end(key)
                return value

        # If value isn't available in local cache,
        # retrieve from upstream using _get_item_uncached.
        value = self._get_item_uncached_concurrent(key)

        # If no tree was retrieved from get_tree_data,
        # return None directly.
        if value is None:
            return

        if self.capacity is not None:
            with self.cache_dict_lock:
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)

        return value
