import collections
import json
import os
import random
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple


from ..interfaces import RolloutConfig, RolloutWorkerConfig
from ...utils import Action, ActionsConfig

TEMPLATE = """\
You have access to a search engine tool. To invoke search, \
begin your query with the phrase "SEARCH: ". You may invoke the search \
tool as many times as needed. 

Your task is to analyze the factuality of the given statement.

Statement: {statement}

After providing all your analysis steps, summarize your analysis \
and state "True statement; Factuality: 1" if you think the statement \
is factual, or "False statement; Factuality: 0" otherwise.
"""

LABEL_MAP = {
    2: 0,
    1: 1,
}


def _is_source_listed(
    source_list: List[Tuple[str, str]], source_dataset_name: str, source_split: str
) -> bool:
    """Return whether (source_dataset_name, source_split) matches source_list.

    Params:
        source_list: list of (source_dataset_name, source_split) patterns.
    """
    for source_dataset_pattern, source_split_pattern in source_list:
        if re.match(source_dataset_pattern, source_dataset_name) and re.match(
            source_split_pattern, source_split
        ):
            return True

    return False


class Config(RolloutConfig):
    root_num_children: int = 2
    forest_name_suffix: str = "basic_search"

    rollout_worker_config = RolloutWorkerConfig(
        rollout_delimiter_pattern=r"(SEARCH)[\s\n]*",
        prediction_pattern=r"Factuality: ([0-9])",
        max_branching_altitude=10,
        max_num_retries=10,
        max_num_children=2,
    )

    dataset_path = os.environ.get("DATASET_PATH")
    dataset_shuffling_seed = int(os.environ.get("DATASET_SHUFFLING_SEED", 0))
    per_source_limit = int(os.environ.get("PER_SOURCE_LIMIT", -1))

    # MongoDB limit per file is 16MB.
    max_line_length = int(os.environ.get("MAX_LINE_LENGTH", 65536))

    actions_config = ActionsConfig(
        actions=[
            Action(
                name="search",
                pattern=r"SEARCH:?[\s\n]*(?P<query>.+)",
                max_altitude=30 * 2,
                max_altitude_placeholder="Summary: ",
                query_templates=("Look up info about the following: {query}",),
                response_role="user",
                response_template="Search result: {}",
            )
        ]
    )

    # Information about this forest configuration.
    # For the convenience of human labellers.
    forest_info: Dict[str, Any] = {
        "prompt_template": TEMPLATE,
        "dataset_path": dataset_path,
        "dataset_shuffling_seed": dataset_shuffling_seed,
        "per_source_limit": per_source_limit,
        "max_claim_length": max_line_length,
    }
    dataset_feature_names: List[str] = ["dataset_split", "claim", "label", "source"]

    @staticmethod
    def get_dataset_iterator(split: str) -> Iterator[Mapping]:
        """
        Handles data loading. For example: download data using the
        HuggingFace `datasets` library and specify dataset-specific
        configs.

        Params:
            split: name of dataset split.

        Returns:
            an iterator of mappings from column name to value.
            e.g., a list of dictionaries, one for each data item.

        - Load dataset_dict from path specified in env var.
            - from Hub if not a local path
        - Select split
        - Return dataset iterator
        """
        data_sources_list: List[Tuple[str, str]] = [
            tuple(split_name.split("/")) for split_name in split.split(",")
        ]

        # number of examples per dataset source.
        per_source_count = collections.Counter()

        if (Config.dataset_path is None) or (not os.path.isfile(Config.dataset_path)):
            raise EnvironmentError(
                "Please point DATASET_PATH to a jsonlines file. "
                "example: DATASET_PATH=data/liar_new.jsonl python3 ..."
            )

        dataset: List[Dict[str, Any]] = []
        with open(Config.dataset_path, "r") as data_file:
            input_lines = data_file.read().splitlines()
            random.seed(Config.dataset_shuffling_seed)
            random.shuffle(input_lines)

            for line in input_lines:
                line_parsed = json.loads(line)
                source_dataset = line_parsed.get("dataset", "")
                source_split = line_parsed.get("split", "")

                if (
                    _is_source_listed(data_sources_list, source_dataset, source_split)
                    and (Config.get_label(line_parsed) is not None)
                    and (
                        (
                            per_source_count[(source_dataset, source_split)]
                            < Config.per_source_limit
                        )
                        or (Config.per_source_limit == -1)
                    )
                    and ("claim" in line_parsed)
                ):
                    line_parsed["source"] = "{}/{}".format(source_dataset, source_split)
                    dataset.append(line_parsed)
                    per_source_count[(source_dataset, source_split)] += 1

        return dataset  # type: ignore

    @staticmethod
    def serialize_dataset_item(dataset_entry: Mapping) -> Dict[str, Any]:
        """
        Create a JSON-serializable representation of one item
        from the dataset (e.g., for reference in the tree attributes.)

        Params:
            dataset_entry: Mapping, see get_dataset_iterator.

        Returns:
            JSON-serializable representation of the given
            dataset item.
        """
        output = dict(dataset_entry)
        if len(json.dumps(output)) > Config.max_line_length:
            return {"source": output.get("source")}

        return output

    @staticmethod
    def get_root_prompt(dataset_entry: Mapping) -> str:
        """
        Return the question to the LLM about the given item
        from the dataset.

        Params:
            dataset_entry: Mapping, see get_dataset_iterator.

        Returns:
            a string denoting the initial instruction for the language model
        """
        context = {
            "statement": dataset_entry["claim"],
        }
        root_prompt = TEMPLATE.format(**context)
        return root_prompt

    @staticmethod
    def get_label(dataset_entry: Mapping) -> Optional[Any]:
        """
        If applicable, return the ground truth label for this
        particular item in the dataset. Note that some datasets
        are unlabelled- return None in these cases.

        Params:
            dataset_entry: Mapping, see get_dataset_iterator.

        Returns:
            A comparable label for the dataset item, or
            None if not applicable for this task.
        """
        label_value = dataset_entry.get("veracity")

        # Return None if the label is unavailable.
        label: Optional[int] = LABEL_MAP.get(label_value)

        return label
