import asyncio
import re
from os import getenv
from typing import TYPE_CHECKING, Any, Coroutine, Iterable, Literal

from agents import (
    Agent,
    ItemHelpers,
    OpenAIChatCompletionsModel,
    Runner,
    RunResult,
    function_tool,
)
from agents.tool import WebSearchTool
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from ..generation_utils import AsyncElasticsearchCache, cached, indexed, rate_limited
from ..tracing_utils.langfuse_otlp import get_langfuse_trace_id, langfuse, tracer

SearchVariant = Literal["es", "oai"]


class DatasetRow(BaseModel):
    claim: str
    veracity: str
    dataset: str


class Output(BaseModel):
    """Represents feasibility evaluation output."""

    statement: str
    factuality_prediction: int | None
    output_text: str
    is_output_invalid: bool
    trace: list[str]
    langfuse_trace_id: str
    data_source: DatasetRow | None = None
    is_correct: bool | None = None

    cache_hit: bool = False


def serialize_run_result(run_result: RunResult) -> list[str]:
    """Serialize Agent trace."""
    return [str(_result) for _result in run_result.new_items]


INSTRUCTIONS = """\
Your task is to analyze the factuality of the given statement.

You may invoke the "search" tool as many times as needed to retrieve \
up-to-date information. However, before invoking the tool, you must \
explain your rationale for doing so.

After providing all your analysis steps, summarize your analysis \
and state "True statement; Factuality: 1" if you think the statement \
is factual, or "False statement; Factuality: 0" otherwise.
"""

RETRIEVAL_INSTRUCTIONS = """\
Answer the question by summarizing the search results. 
"""

openai_client = AsyncOpenAI()
es_client = AsyncElasticsearch(getenv("KB_ES_HOST"), api_key=getenv("KB_ES_API_KEY"))


async def news_search(keyword: str) -> list[dict]:
    """Search News Database.

    Returns:
        a list of results. Empty list of no match is found.
    """

    title_match = {
        "match": {
            "title.fuzzy": {
                "query": keyword,
                "fuzziness": "AUTO",
                "operator": "and",
            }
        }
    }
    text_match = {
        "match": {
            "text": {
                "query": keyword,
                "fuzziness": "AUTO",
                "operator": "and",
            }
        }
    }

    # Return first match.
    for requirements_option in [[text_match]]:
        response = await es_client.search(
            index="ccnews",
            body={
                "query": {"bool": {"must": requirements_option}},
                "highlight": {
                    "fields": {
                        "text": {
                            # snippet length in characters
                            "fragment_size": 1000,
                            "number_of_fragments": 5,
                        }
                    },
                },
                "size": 5,
                # Do not return full document source
                "_source": False,
            },
        )
        hits = response["hits"]["hits"]
        if len(hits) > 0:
            return hits

    return []


search_agent_oai = Agent(
    name="SearchAgent",
    instructions=(
        "You are a search agent. You receive a single search query as input. "
        "Use the WebSearchTool to perform a web search, then produce a concise "
        "'search summary' of the key findings. Do NOT return raw search results."
    ),
    tools=[
        WebSearchTool(search_context_size="low"),
    ],
    # a faster, smaller model for quick searches
    model="gpt-4o-mini",
)

search_agent_es = Agent(
    name="SearchAgent",
    instructions=(
        "You are a search agent. You receive a single search query as input. "
        "Use the WebSearchTool to perform a web search, then produce a concise "
        "'search summary' of the key findings. Do NOT return raw search results."
    ),
    tools=[
        function_tool(news_search),
    ],
    # a faster, smaller model for quick searches
    model="gpt-4o-mini",
)

main_agent_oai = Agent(
    name="MainAgent",
    instructions=INSTRUCTIONS,
    tools=[
        search_agent_oai.as_tool(
            tool_name="search",
            tool_description="Perform a web search for a query and return a concise summary.",
        )
    ],
    # a larger, more capable model for reasoning over summaries
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=openai_client),
)

main_agent_es = Agent(
    name="MainAgent",
    instructions=INSTRUCTIONS,
    tools=[
        search_agent_es.as_tool(
            tool_name="search",
            tool_description="Perform a web search for a query and return a concise summary.",
        )
    ],
    # a larger, more capable model for reasoning over summaries
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=openai_client),
)


async def evaluate(
    statement: str, data_source: DatasetRow | None, variant: SearchVariant
) -> Output:
    """Evaluate on a single statement."""
    if variant == "es":
        main_agent = main_agent_es
    else:
        main_agent = main_agent_oai

    with tracer.start_as_current_span(f"Agent-SDK-{variant}-search"):
        result = await Runner.run(main_agent, statement)
        langfuse_trace_id = get_langfuse_trace_id()
        langfuse.trace(
            id=langfuse_trace_id,
            input=statement,
            output=result.final_output,
            metadata={
                "data_source": (
                    data_source.model_dump() if data_source is not None else None
                )
            },
        )

    prediction_match = re.search(r"Factuality:\s*(\d)", result.final_output)

    prediction = None
    try:
        if prediction_match is not None:
            prediction_str = prediction_match.group(1)
            prediction = int(prediction_str)
    except:
        pass

    is_correct: bool | None = None
    if data_source is not None:
        label = data_source.veracity
        if label == "true":
            is_correct = prediction == 1
        if label == "false":
            is_correct = prediction == 0

    return Output(
        statement=statement,
        factuality_prediction=prediction,
        output_text=result.final_output,
        is_output_invalid=prediction is None,
        trace=serialize_run_result(result),
        langfuse_trace_id=langfuse_trace_id,
        data_source=data_source,
        is_correct=is_correct,
    )


async def batch_evaluate(
    dataset_rows: Iterable[dict[str, Any]],
    cache: AsyncElasticsearchCache,
    async_semaphore: asyncio.Semaphore,
    variant: SearchVariant,
    total: int | None = None,
) -> list[Output]:
    """Evaluate veracity, reusing cache whenever possible."""
    rows = [DatasetRow(**_row) for _row in dataset_rows]
    coros: list[Coroutine[None, None, tuple[int, Output]]] = [
        indexed(
            index=_index,
            coro=rate_limited(
                _fn=lambda _row=_row: cached(
                    _fn=lambda _row=_row: evaluate(
                        _row.claim, data_source=_row, variant=variant
                    ),
                    _key=_row.claim,
                    output_serializer_class=Output,
                    cache=cache,
                ),
                semaphore=async_semaphore,
            ),
        )
        for _index, _row in enumerate(rows)
    ]

    # Map original position to output
    outputs: dict[int, Output] = {}
    for coro in tqdm(asyncio.as_completed(coros), ncols=75, total=total):
        _index, output = await coro
        outputs[_index] = output

    return [outputs[_index] for _index in range(len(coros))]


async def main():
    result = Runner.run_streamed(
        main_agent_es,
        " Comparing the price of oil and gas in June 2008 to March 2022 shows that oil companies are price gouging.",
    )

    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        if event.type == "raw_response_event":
            continue
        # When the agent updates, print that
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        # When items are generated, print them
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}\n")
            elif event.item.type == "message_output_item":
                print(
                    f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}\n"
                )
            else:
                pass  # Ignore other event types


if __name__ == "__main__":
    asyncio.run(main())
