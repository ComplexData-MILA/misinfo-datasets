"""
Utils for efficient async OpenAI text generation.
"""

from contextlib import contextmanager
from typing import Any, Callable, TypeVar
import asyncio
import os
from pathlib import Path
import gzip

import backoff
import openai
import pydantic
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


Data = TypeVar("Data")


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def generate(
    prompt: str,
    data: Data,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
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
        cache: Cache
    """
    async with async_semaphore:
        with cache.cache_response(prompt) as (cached_output, _callback):
            if cached_output is not None:
                return cached_output, data

            assert not assert_cached, "Cache miss. Maybe run without --assert_cached?"

            response = await client.chat.completions.create(
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
            output[_index] = extract_answer_fn(_text_output)

        return output
