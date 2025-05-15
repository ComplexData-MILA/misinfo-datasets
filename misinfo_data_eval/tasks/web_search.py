import asyncio
import re
from typing import Any, Coroutine, Iterable

from agents import Agent, ItemHelpers, Runner, RunResult, set_trace_processors
from agents.tool import WebSearchTool
from opik.integrations.openai.agents import OpikTracingProcessor
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from ..generation_utils import AsyncElasticsearchCache, cached, rate_limited

set_trace_processors(processors=[OpikTracingProcessor("20250514-CDL-misinfo-data-1a")])


class DatasetRow(BaseModel):
    claim: str
    veracity: str


class Output(BaseModel):
    """Represents feasibility evaluation output."""

    statement: str
    factuality_prediction: int | None
    output_text: str
    is_output_invalid: bool
    trace: list[str]
    data_source: dict[str, Any] | None = None
    is_correct: bool | None = None


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


search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "You are a search agent. You receive a single search query as input. "
        "Use the WebSearchTool to perform a web search, then produce a concise "
        "'search summary' of the key findings. Do NOT return raw search results."
    ),
    tools=[WebSearchTool(search_context_size="low")],
    # a faster, smaller model for quick searches
    model="gpt-4o-mini",
)

main_agent = Agent(
    name="MainAgent",
    instructions=INSTRUCTIONS,
    tools=[
        search_agent.as_tool(
            tool_name="search",
            tool_description="Perform a web search for a query and return a concise summary.",
        )
    ],
    # a larger, more capable model for reasoning over summaries
    model="gpt-4o",
)


async def evaluate(statement: str, data_source: dict[str, Any] | None) -> Output:
    """Evaluate on a single statement."""
    result = await Runner.run(main_agent, statement)
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
        label = data_source.get("veracity")
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
        data_source=data_source,
        is_correct=is_correct,
    )


async def batch_evaluate(
    dataset_rows: Iterable[dict[str, Any]],
    cache: AsyncElasticsearchCache,
    async_semaphore: asyncio.Semaphore,
    total: int | None = None,
) -> list[Output]:
    """Evaluate veracity, reusing cache whenever possible."""
    coros: list[Coroutine[None, None, Output]] = [
        rate_limited(
            lambda: cached(
                _fn=lambda: evaluate(DatasetRow(**_row).claim, data_source=_row),
                _key=DatasetRow(**_row).claim,
                output_serializer_class=Output,
                cache=cache,
            ),
            semaphore=async_semaphore,
        )
        for _row in dataset_rows
    ]

    outputs: list[Output] = []
    for coro in tqdm(asyncio.as_completed(coros), ncols=75, total=total):
        outputs.append(await coro)

    return outputs


async def main():
    result = Runner.run_streamed(
        main_agent,
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
