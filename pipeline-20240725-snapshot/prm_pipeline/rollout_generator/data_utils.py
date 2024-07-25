import os
import re
from typing import Optional, Union

import datasets

PredictionType = Union[str, float]


def load_dataset_dict(hf_dataset_path: str) -> datasets.DatasetDict:
    """
    Load HuggingFace dataset_dict from the given local path,
    and if that path does not exist, from the hub.

    Params:
        hf_dataset_path: string, local path or Hub path.

    Returns:
        datasets.DatasetDict
    """
    if os.path.exists(hf_dataset_path):
        dataset_dict = datasets.load_from_disk(hf_dataset_path)
    else:
        dataset_dict = datasets.load_dataset(hf_dataset_path)

    assert isinstance(dataset_dict, datasets.DatasetDict), type(dataset_dict)
    return dataset_dict


def get_prediction(
    prediction: str, prediction_pattern: str = r"<em>(-?[0-9]+)</em>"
) -> Optional[PredictionType]:
    """
    Extract prediction int from text.
    Return None if there are none or more than one matches.

    Params:
        prediction: str
        prediction_pattern: Optional[str] regexp.

    Returns:
        int or None.
    """
    matches = re.findall(prediction_pattern, prediction)
    if len(matches) != 1:
        return None

    prediction_string = matches[0]

    try:
        if "." in prediction_string:
            return float(prediction_string)
        else:
            return int(prediction_string)

    except ValueError:
        return prediction_string
