from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

LABEL_MAP = {"true": 1, "false": 0, "unknown": None}


def evaluate_temporal_correlations(dataset: list[dict[str, Any]]) -> dict[str, float]:
    """Evaluate temporal correlation on the given data.

    Adapted from implementation by Kellin Pelrine.
    """
    temp_df = pd.DataFrame(dataset)

    temp_df = temp_df[temp_df.veracity != 3]
    temp_df["veracity"] = temp_df["veracity"].apply(LABEL_MAP.get)
    temp_df = temp_df[temp_df.veracity.notna()]
    temp_df["veracity"] = temp_df["veracity"].astype(int)
    print("len(dataset) filtered by veracity is not unknown:", len(temp_df))
    temp_df = temp_df[temp_df.tweet_id.notna()]
    print("len(dataset) filtered by tweet_id is not unknown:", len(temp_df))

    def convert_to_int(x):
        x = str(x)[:4]
        if "." in x:
            x = x.replace(".", "")
        else:
            x = x[:3]

        try:
            return int(x)
        except ValueError:
            return "ERROR"

    temp_df["dates"] = temp_df["tweet_id"].apply(convert_to_int)
    print(len(temp_df[temp_df.dates == "ERROR"]))
    temp_df = temp_df[temp_df.dates != "ERROR"]
    temp_df["dates"] = temp_df["dates"].astype(int)

    clf = RandomForestClassifier(
        max_depth=20, random_state=0
    )  # , class_weight='balanced')

    train, test = train_test_split(temp_df, test_size=0.25, random_state=42)

    clf.fit(temp_df.dates.values.reshape(-1, 1), temp_df.veracity.values)
    preds = clf.predict(test.dates.values.reshape(-1, 1))
    true = test.veracity.values
    print(classification_report(preds, true, digits=3))
    return classification_report(preds, true, digits=3, output_dict=True)
