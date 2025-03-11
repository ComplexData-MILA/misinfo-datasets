"""Evaluate feasibility using LLM."""

from collections import Counter
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from ..generation_utils import AsyncLLMEvaluator

from ..prompt_templates import (
    TEMPLATE_BINARY_NO_SEARCH_NO_DATE,
    TEMPLATE_BINARY_WITH_SEARCH_NO_DATE,
)

TEMPLATES = [TEMPLATE_BINARY_NO_SEARCH_NO_DATE, TEMPLATE_BINARY_WITH_SEARCH_NO_DATE]

# Map (NO_SEARCH, WITH_SEARCH) predictions to one of the three aggregated predictions
PROJECTIONS = {
    ("0", "0"): "not feasible even with search",
    ("0", "1"): "feasible, requires search",
    ("1", "0"): "feasible, no search required",
    ("1", "1"): "feasible, no search required",
}


async def evaluate_feasibility(
    statements: list[str], llm_evaluator: "AsyncLLMEvaluator"
) -> dict[str | None, float]:
    """Evaluate feasibility of the given data.

    Params:
        statements: list of statements to evaluate.
        llm_evaluator: AsyncLLMEvaluator. See documentation for more detail.

    Returns:
        metrics:
        - "feasible, no search required": float between 0.0 and 1.0
        - "feasible, requires search": float between 0.0 and 1.0
        - "not feasible even with search": float between 0.0 and 1.0
        - None: float between 0.0 and 1.0
        (these categories are mutually-exclusive, and add up to 1.0)
    """

    per_template_predictions: list[list[str | None]] = []

    for _template in tqdm(TEMPLATES, ncols=75, desc="Evaluating Feasibility"):
        _predictions = await llm_evaluator.evalute_on_template(
            rows=[{"statement": _statement} for _statement in statements],
            apply_template_fn=lambda row: _template.format(statement=row["statement"]),
            extract_answer_fn=lambda response: (
                response.split("|")[-1].strip() if "|" in response else None
            ),
        )
        per_template_predictions.append(_predictions)

    # Each "_paired_prediction" is a tuple of
    # (NO_SEARCH feasibility, WITH_SEARCH feasibility)
    projected_predictions: list[str | None] = []
    for _paired_prediction in zip(*per_template_predictions):
        projected_predictions.append(PROJECTIONS.get(_paired_prediction))

    return Counter(projected_predictions)
