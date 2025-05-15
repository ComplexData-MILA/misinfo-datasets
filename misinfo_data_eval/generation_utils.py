"""
Utils for efficient async OpenAI text generation.
"""

import asyncio
import gzip
import hashlib
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional, TypeVar

import backoff
import openai
import pydantic
from elasticsearch import AsyncElasticsearch
from tqdm.asyncio import tqdm

client = openai.AsyncOpenAI()


class CacheEntry(pydantic.BaseModel):
    """One row in the cache file."""

    prompt: str
    response: str


class Cache:
    def __init__(self, cache_path: str | Path):
        self.cache: dict[str, str] = {}
        self.new_entries: dict[str, str] = {}
        self.cache_path = cache_path
        if not os.path.exists(cache_path):
            gzip.open(cache_path, "wt")

        with gzip.open(cache_path, "rt") as cache_file:
            for row in cache_file.readlines():
                if len(row.strip()) > 0:
                    entry = CacheEntry.model_validate_json(row.strip())
                    self.cache[entry.prompt] = entry.response

    def write(self):
        with gzip.open(self.cache_path, "at") as cache_file:
            cache_file.write("\n")
            for key, value in self.cache.items():
                line = CacheEntry(prompt=key, response=value).model_dump_json() + "\n"
                cache_file.write(line)

    @contextmanager
    def cache_response(self, prompt: str):
        """Cache response."""

        def _add_cache_callback(response: str):
            self.new_entries[prompt] = response
            self.cache[prompt] = response

        if prompt in self.cache:
            yield self.cache[prompt], _add_cache_callback
            return

        yield None, _add_cache_callback


class AsyncElasticsearchCache:
    def __init__(self, es: AsyncElasticsearch, index_name: str) -> None:
        self.es = es
        self.index_name = index_name
        logging.info(f"Elastic Index Name: {self.index_name}")

        self.is_refresh_required = bool(os.environ.get("FORCE_CACHE_REFRESH"))
        if self.is_refresh_required:
            logging.warning("FORCE_CACHE_REFRESH is enabled. All queries will miss.")

    @staticmethod
    async def maybe_from_env_var(index_name: str) -> "AsyncElasticsearchCache | None":
        """Initialize from env var. Returns None if any of the required env vars are missing."""
        required_env_keys = ["ELASTIC_SEARCH_HOST", "ELASTIC_SEARCH_API_KEY"]
        if not all((_key in os.environ) for _key in required_env_keys):
            logging.warning(
                "All of these are required to enable ElasticsearchCache: "
                f"{required_env_keys}."
                " Not enabling ElasticsearchCache since some keys are not set."
            )
            return None

        es = AsyncElasticsearch(
            os.environ["ELASTIC_SEARCH_HOST"],
            api_key=os.environ["ELASTIC_SEARCH_API_KEY"],
            request_timeout=None,
        )

        # Ensure the index exists at startup (parse the name if '/' is present)
        index_name = index_name.lower().replace("/", "_")
        if not await es.indices.exists(index=index_name):
            await es.indices.create(index=index_name)
        return AsyncElasticsearchCache(es=es, index_name=index_name)

    async def get(self, query: str, nonce: Optional[str] = None) -> str | None:
        """Try reading response from cache.

        Args:
            query (str): The query to fetch from cache.
            nonce (str, optional): An optional nonce to differentiate cache entries.

        Returns:
            str | None: Cached result if available.
        """
        if self.is_refresh_required:
            return None

        # Cache lookup
        query_hash = self._get_query_hash(query=query, nonce=nonce)
        try:
            response = await self.es.get(index=self.index_name, id=query_hash)
            if response.get("found"):
                logging.info(f"Cache hit: {query_hash}")
                return response["_source"]["result"]
        except Exception:
            logging.debug(f"Cache miss: {query_hash}")

        return None  # Cache miss or index doesn't exist

    async def set(self, query: str, value: str, nonce: Optional[str] = None) -> None:
        """Set/Update cache.

        Args:
            query (str): The query whose result is to be cached.
            value (str): The value to store in cache.
            nonce (str, optional): An optional nonce to differentiate cache entries.
        """
        query_hash = self._get_query_hash(query=query, nonce=nonce)
        doc = {"query": query, "result": value, "nonce": nonce}
        await self.es.index(index=self.index_name, id=query_hash, document=doc)

    async def close(self) -> None:
        """Close Elasticsearch connection."""
        await self.es.close()

    @staticmethod
    def _get_query_hash(query: str, nonce: Optional[str] = None) -> str:
        query_key = query
        if nonce is not None:
            query_key += f"\n{nonce}"
        return hashlib.sha256(query_key.encode()).hexdigest()


Data = TypeVar("Data")


V = TypeVar("V")
Serializer = TypeVar("Serializer", bound=pydantic.BaseModel)


async def cached(
    _fn: Callable[[], Coroutine[None, None, Serializer]],
    _key: str,
    output_serializer_class: type[Serializer],
    cache: AsyncElasticsearchCache,
) -> Serializer:
    """Run _fn only if cache is missed."""
    cached_data = await cache.get(_key)
    if (cached_data is not None) and not bool(os.getenv("IGNORE_CACHE")):
        # Cache hit
        return output_serializer_class.model_validate_json(cached_data)

    # Cache miss
    output = await _fn()
    cached_data = output.model_dump_json()
    await cache.set(_key, value=cached_data)

    return output


async def rate_limited(
    _fn: Callable[[], Coroutine[None, None, V]], semaphore: asyncio.Semaphore
) -> V:
    """Run _fn with semaphore rate limit."""
    async with semaphore:
        return await _fn()


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def generate(
    prompt: str,
    data: Data,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
    async_client: openai.AsyncOpenAI,
    cache: Cache,
    assert_cached: bool = False,
    max_completion_tokens: int = 4096,
) -> tuple[str, Data]:
    """Generate using ChatCompletion generation.

    Params:
        prompt: str
        data: to be returned verbatim
        model_name: str
        async_semaphore: to limit number of concurrent requests.
        async_client: async OpenAI client.
        cache: Cache
    """
    async with async_semaphore:
        with cache.cache_response(prompt) as (cached_output, _callback):
            if cached_output is not None:
                return cached_output, data

            assert not assert_cached, "Cache miss. Maybe run without --assert_cached?"

            response = await async_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
            )

            output = response.choices[0].message.content
            assert output is not None
            _callback(output)

    return output, data


class AsyncLLMEvaluator:
    def __init__(
        self,
        model_name: str,
        cache: Cache,
        async_semaphore: asyncio.Semaphore,
        assert_cached: bool = False,
        max_completion_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.cache = cache
        self.async_semaphore = async_semaphore
        self.assert_cached = assert_cached
        self.max_completion_tokens = max_completion_tokens
        self.async_client = openai.AsyncOpenAI()

    async def evalute_on_template(
        self,
        rows: list[dict[str, Data]],
        apply_template_fn: Callable[[dict[str, Data]], str],
        extract_answer_fn: Callable[[str], str | None],
    ) -> list[str | None]:
        """Apply template to data, generate, and return predictions.

        Params:
            rows: list of dataset rows, e.g., from HF dataset.
            apply_template_fn: given row of dataset, return query to the LLM.
            extract_answer_fn: given LLM response, extract LLM choice,
                or None if not matched.

        Returns:
            list of extracted answers, same length as data.
        """
        coros = [
            generate(
                prompt=apply_template_fn(row),
                data={**row, "_index": index},
                model_name=self.model_name,
                async_semaphore=self.async_semaphore,
                async_client=self.async_client,
                cache=self.cache,
                assert_cached=self.assert_cached,
                max_completion_tokens=self.max_completion_tokens,
            )
            for index, row in enumerate(rows)
        ]

        output: list[str | None] = [None for _ in range(len(coros))]

        for task in tqdm(asyncio.as_completed(coros), ncols=75, total=len(coros)):
            _text_output, _data = await task
            _index = _data["_index"]
            output[_index] = extract_answer_fn(_text_output)  # type: ignore

        return output
