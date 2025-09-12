"""
Evaluate web retrieval pipeline using LangFuse

- Run Agent SDK to produce traces
- Fetch and score traces
- Update traces to include eval scores.
"""

import argparse
import asyncio
import hashlib

from langfuse import Langfuse
from tqdm.auto import tqdm

from .data_loading_utils import DATA_INSTRUCTIONS, load_data
from .generation_utils import AsyncElasticsearchCache, AsyncLLMEvaluator
from .tasks.feasibility_eval import FEASIBILITY_LEVEL_MAPS, evaluate_feasibility
from .tasks.web_search import DatasetRow
from .tasks.web_search import batch_evaluate as evaluate_veracity

parser = argparse.ArgumentParser()
parser.add_argument("--evaluator_model_name")
parser.add_argument("--max_generation_tokens", type=int, default=4096)
parser.add_argument("--web_search_variant", choices=("es", "oai"), required=True)
parser.add_argument("--langfuse_dataset_name", default="cdl-misinfo")
parser.add_argument("--langfuse_populate_dataset", action="store_true", default=False)
parser.add_argument("--langfuse_always_link", action="store_true", default=False)
parser.add_argument(
    "--source_dataset_paths", required=True, nargs="+", help=DATA_INSTRUCTIONS
)
parser.add_argument("--max_concurrency", type=int, default=1)
parser.add_argument("--evaluate_feasibility", action="store_true", default=False)
parser.add_argument("--evaluate_factuality", action="store_true", default=False)
parser.add_argument(
    "--evaluate_temporal_correlation", action="store_true", default=False
)
parser.add_argument("--keyword_analysis", action="store_true", default=False)
parser.add_argument("--assert_cached", action="store_true", default=False)
parser.add_argument("--limit", type=int, default=-1, help="Per-source limit")


async def main():

    args = parser.parse_args()
    run_name = f"openai-agent-{args.web_search_variant}-search"

    async_semaphore = asyncio.Semaphore(args.max_concurrency)
    langfuse = Langfuse()
    es_cache = await AsyncElasticsearchCache.maybe_from_env_var(
        f"cache_misinfo_eval_{run_name}"
    )
    llm_evaluator = AsyncLLMEvaluator(
        model_name=args.evaluator_model_name,
        cache=es_cache,
        async_semaphore=async_semaphore,
        assert_cached=args.assert_cached,
        max_completion_tokens=args.max_generation_tokens,
    )

    langfuse_dataset_name = args.langfuse_dataset_name
    print(f"langfuse_dataset_name: {langfuse_dataset_name}")
    dataset = langfuse.create_dataset(
        langfuse_dataset_name, metadata={"type": "benchmark"}
    )

    assert es_cache is not None

    if args.langfuse_populate_dataset:
        for source_dataset_path in tqdm(args.source_dataset_paths, ncols=75):
            dataset = load_data(source_dataset_path)
            source_path_hash = hashlib.sha256(source_dataset_path.encode()).hexdigest()
            dataset_name_hash = hashlib.sha256(
                langfuse_dataset_name.encode()
            ).hexdigest()

            for index, row in tqdm(
                enumerate(dataset),
                ncols=75,
                total=args.limit,
                desc="Populating LangFuse dataset",
            ):
                # Skip "unknown" rows.
                if (row["veracity"] not in ["true", "false"]) or (
                    len(row["claim"]) < 10
                ):
                    continue

                if (args.limit > 0) and (index >= args.limit):
                    break

                langfuse.create_dataset_item(
                    id=f"{source_path_hash[:6]}-{dataset_name_hash}-{index:05}",
                    dataset_name=langfuse_dataset_name,
                    input=row["claim"],
                    expected_output=row["veracity"],
                    metadata={"source": source_dataset_path, **row},
                )

    # Evaluate in batch, and then pair up dataset (input) and output traces.
    dataset_langfuse = langfuse.get_dataset(langfuse_dataset_name)
    input_items = [_item.metadata for _item in dataset_langfuse.items]
    input_items = [_row for _row in input_items if _row is not None]
    if len(input_items) == 0:
        raise FileNotFoundError(
            f"Langfuse Dataset {args.langfuse_dataset_name} is empty. "
            "Maybe re-run with the following? --langfuse_populate_dataset"
        )

    _, feasibility_outputs = await evaluate_feasibility(
        [DatasetRow(**_item).claim for _item in input_items], llm_evaluator
    )
    eval_outputs = await evaluate_veracity(
        input_items,  # type: ignore[dict]
        cache=es_cache,
        async_semaphore=async_semaphore,
        variant=args.web_search_variant,
        total=len(input_items),
    )
    for index, (item, eval_output, feasibility_output) in enumerate(
        zip(
            dataset_langfuse.items,
            tqdm(eval_outputs, desc="Scoring", ncols=75),
            feasibility_outputs,
        )
    ):
        # Link only on the first run.
        if (not eval_output.cache_hit) or (args.langfuse_always_link):
            item.link(
                None,
                run_name=run_name,
                trace_id=eval_output.langfuse_trace_id,
            )

        langfuse_trace = langfuse.trace(id=eval_output.langfuse_trace_id)

        is_correct = eval_output.is_correct
        if is_correct is not None:
            langfuse_trace.score(
                id=f"correct_{eval_output.langfuse_trace_id}",
                name="Correct",
                value=is_correct,
                data_type="BOOLEAN",
            )

        langfuse_trace.score(
            id=f"valid_{eval_output.langfuse_trace_id}",
            name="Valid",
            value=not eval_output.is_output_invalid,
            data_type="BOOLEAN",
        )
        langfuse_trace.update(tags=[langfuse_dataset_name])

        if feasibility_output is not None:
            feasibility_level = FEASIBILITY_LEVEL_MAPS[feasibility_output]
            langfuse_trace.score(
                id=f"feasibility_search_{eval_output.langfuse_trace_id}",
                name="Feasibility with Search",
                value=feasibility_level >= 1,
                data_type="BOOLEAN",
            )
            langfuse_trace.score(
                id=f"feasibility_no_search_{eval_output.langfuse_trace_id}",
                name="Feasibility, no Search",
                value=feasibility_level == 2,
                data_type="BOOLEAN",
            )

        if (index - 1) % 10 == 0:
            langfuse.flush()

    print("Waiting for langfuse.flush()")
    langfuse.flush()
    await es_cache.close()


if __name__ == "__main__":
    asyncio.run(main())
