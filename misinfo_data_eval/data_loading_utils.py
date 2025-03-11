"""Utils for loading source data."""

from typing import Any
import os

import datasets

DATA_INSTRUCTIONS = (
    "Specify data source in one of the following format: "
    "\nhf://dataset_path_or_repo_name[@revision]"
    "\nhf://dataset_path_or_repo_name[@revision]:split_name"
    "\nhf://dataset_path_or_repo_name[@revision]:subset_name:split_name"
    "\ntsv://path_to_local_tsv_file"
    "\ncsv://path_to_local_csv_file"
)


def load_data(data_source: str) -> list[dict[str, Any]]:
    """Load dataset rows from various sources."""
    if data_source.count("://") != 1:
        raise ValueError(DATA_INSTRUCTIONS)

    provider, path = data_source.split("://", maxsplit=1)
    if provider == "hf":
        _hf_args = path.split(":", maxsplit=2)
        if len(_hf_args) == 3:
            _hf_data_path, _hf_subset_name, _hf_split_name = _hf_args
        elif len(_hf_args) == 2:
            _hf_subset_name = None
            _hf_data_path, _hf_split_name = _hf_args
        else:
            _hf_subset_name = None
            _hf_split_name = None
            _hf_data_path = _hf_args[0]

        if os.path.exists(_hf_data_path) and (_hf_subset_name is None):
            print(
                f"Loading HF from disk: {_hf_data_path}; "
                f"Name of data split: {_hf_split_name}"
            )
            _hf_dataset = datasets.load_from_disk(_hf_data_path)
        else:
            # Load from Hub
            if "@" in _hf_data_path:
                _hf_repo_name, _hf_git_revision = _hf_data_path.split("@", maxsplit=1)
            else:
                _hf_repo_name = _hf_data_path
                _hf_git_revision = None

            print(
                f"Loading from HF hub: {_hf_repo_name}\n"
                f"Revision: {_hf_git_revision}\n"
                f"Name of data subset: {_hf_subset_name}\n"
                f"Name of data split: {_hf_split_name}"
            )
            _hf_dataset = datasets.load_dataset(
                _hf_repo_name,
                name=_hf_subset_name,
                revision=_hf_git_revision,
            )

        if _hf_split_name:
            return _hf_dataset[_hf_split_name]
        else:
            return _hf_dataset

    elif provider == "tsv":
        import pandas as pd

        return datasets.Dataset.from_pandas(pd.read_csv(path, delimiter="\t"))

    elif provider == "csv":
        import pandas as pd

        return datasets.Dataset.from_pandas(pd.read_csv(path))

    raise ValueError(DATA_INSTRUCTIONS)
