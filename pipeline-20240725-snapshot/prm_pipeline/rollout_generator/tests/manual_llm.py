import argparse
import json
import os

from tqdm.auto import tqdm

from ...metrics.utils.metrics_utils import get_categories, get_category_metrics
from ...utils import chat_competion, split_reasoning, substitute
from ..data_utils import get_prediction, load_dataset_dict

parser = argparse.ArgumentParser()
parser.add_argument("template_json")
parser.add_argument("--dataset_name", default="liar")
parser.add_argument("--dataset_split", default="train")
parser.add_argument("--num_examples", type=int, default=12)
args = parser.parse_args()

with open(args.template_json, "r") as template_file:
    template_data = json.load(template_file)

dataset_dict = load_dataset_dict(args.dataset_name)


dataset = dataset_dict[args.dataset_split]

# For compatibility with metric utils.
rollouts = []

for index, dataset_entry in tqdm(enumerate(dataset), total=args.num_examples):
    if index > args.num_examples:
        break

    for _ in range(1):
        statement = dataset_entry["statement"]  # type: ignore
        prompt = substitute(template_data["template"], dict(dataset_entry))
        response = chat_competion(os.environ.get("MODEL_NAME"), prompt)
        print(prompt)

        prediction = get_prediction(response, template_data["prediction_pattern"])

        print(
            "DATASET_ENTRY: {}\nPROMPT: \n```\n{}\n```\n\nRESPONSE:\n```\n{}\n"
            "\nPARSED\n{}\n```\nPREDICTION: {}".format(
                dataset_entry,
                prompt,
                response,
                json.dumps(
                    split_reasoning(
                        response, template_data["reasoning_delimiter_pattern"]
                    ),
                    indent=2,
                ),
                prediction,
            )
        )

        rollouts.append(
            {
                "prediction": prediction,
                "example": dataset_entry,
                "ground_truth": dataset_entry["label"],  # type: ignore
            }
        )


categories = get_categories(rollouts)

# Maps AI "prediction" to dataset label.
LABELS_MAP = {
    0: "false",
    1: "half-true",
    2: "mostly-true",
    3: "true",
    4: "barely-true",
    5: "pants-fire",
}

# Map label to correct prediction label.
CORRECT_PREDICTION_MAP = {
    "false": 0,
    "half-true": 1,
    "mostly-true": 1,
    "true": 1,
    "barely-true": 0,
    "pants-fire": 0,
}

PREDICTION_MAP = {
    -1: "More-Info",
    0: "False",
    1: "True",
}


for category in categories:
    metrics = get_category_metrics(
        rollouts, None, category, [CORRECT_PREDICTION_MAP[LABELS_MAP[category]]], []
    )
    print(metrics)
