import argparse
import asyncio
from os import makedirs

import datasets
from pathlib import Path
import pandas as pd

from .generation_utils import Cache, AsyncLLMEvaluator
from .tasks.feasibility_eval import evaluate_feasibility
from .data_loading_utils import load_data, DATA_INSTRUCTIONS

parser = argparse.ArgumentParser()
parser.add_argument("--evaluator_model_name", required=True)
parser.add_argument("--source_dataset_path", required=True, help=DATA_INSTRUCTIONS)
parser.add_argument("--max_concurrency", type=int, default=1)
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

    try:
        feasibility_metrics = await evaluate_feasibility(
            statements=[_row["claim"] for _row in dataset][: args.limit],
            llm_evaluator=llm_evaluator,
        )
    
    # Cache previous generations if interrupted.
    finally:
        cache.write()

    print(feasibility_metrics)


if __name__ == "__main__":
    asyncio.run(main())
