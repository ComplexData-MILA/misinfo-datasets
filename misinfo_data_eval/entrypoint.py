import argparse
import asyncio
import json
from os import makedirs

from .data_loading_utils import DATA_INSTRUCTIONS, load_data
from .generation_utils import AsyncLLMEvaluator, Cache
from .tasks.feasibility_eval import evaluate_feasibility
from .tasks.temporal_correlation import evaluate_temporal_correlations

parser = argparse.ArgumentParser()
parser.add_argument("--evaluator_model_name")
parser.add_argument("--source_dataset_path", required=True, help=DATA_INSTRUCTIONS)
parser.add_argument("--max_concurrency", type=int, default=1)
parser.add_argument("--evaluate_feasibility", action="store_true", default=False)
parser.add_argument(
    "--evaluate_temporal_correlation", action="store_true", default=False
)
parser.add_argument("--assert_cached", action="store_true", default=False)
parser.add_argument("--max_generation_tokens", type=int, default=4096)
parser.add_argument("--limit", type=int, default=-1)


async def main():
    args = parser.parse_args()

    makedirs("data/cache", exist_ok=True)
    async_semaphore = asyncio.Semaphore(args.max_concurrency)
    cache = Cache(f"data/cache/{args.evaluator_model_name}.jsonl.gz")
    llm_evaluator = AsyncLLMEvaluator(
        model_name=args.evaluator_model_name,
        cache=cache,
        async_semaphore=async_semaphore,
        assert_cached=args.assert_cached,
        max_completion_tokens=args.max_generation_tokens,
    )

    dataset = load_data(args.source_dataset_path)
    print("len(dataset):", len(dataset))

    # Feasibility Evaluation
    if args.evaluate_feasibility:
        if args.evaluator_model_name is None:
            msg = "Must specify an LLM evaluator for evaluate_feasibility."
            raise ValueError(msg)

        try:
            feasibility_metrics = await evaluate_feasibility(
                statements=[_row["claim"] for _row in dataset][: args.limit],
                llm_evaluator=llm_evaluator,
            )
            print(json.dumps(feasibility_metrics, indent=2))

        finally:
            # Cache previous generations if interrupted.
            cache.write()

    # Temporal Correlation Evaluation, if "tweet_id" data is available
    if args.evaluate_temporal_correlation:
        if not (len(dataset) > 0) and ("tweet_id" in dataset[0].keys()):
            msg = (
                "tweet_id is not present in dataset. "
                "Cannot run temporal correlation analysis."
            )
            raise ValueError(msg)

        temporal_correlation_metrics = evaluate_temporal_correlations(dataset)
        print(json.dumps(temporal_correlation_metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
