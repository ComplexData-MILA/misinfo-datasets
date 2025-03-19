import pandas as pd
import numpy as np
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK stopwords are downloaded
import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
from typing import Any

LABEL_MAP = {"true": 1, "false": 0, "unknown": None}

def keyword_analysis(dataset: list[dict[str, Any]]) -> dict[str, float]:
    """
    Processes a dataset to analyze veracity distribution, top keywords, and train a classifier.
    
    Parameters:
        data (pd.DataFrame or np.ndarray): A 2D dataset with "dataset", "claim", and "veracity" columns.
        dataset_name (str): The name of the dataset to process.
    
    Returns:
        dict: A dictionary containing veracity counts, top keywords, confusion matrix, and F1 scores.
    """

    # Convert to DataFrame if input is NumPy array
    temp_df = pd.DataFrame(dataset)


    # exclude all but T/F (TODO: Add if statement to include 'mixed' values) 
    temp_df = temp_df[temp_df.veracity != 3]
    temp_df["veracity"] = temp_df["veracity"].apply(LABEL_MAP.get)
    temp_df = temp_df[temp_df.veracity.notna()]
    temp_df["veracity"] = temp_df["veracity"].astype(int)
    print("len(dataset) filtered by veracity is not unknown:", len(temp_df))

    # Eventual overwrite
    df = temp_df

    # Veracity Label Distribution
    label_counts = df["veracity"].value_counts().to_dict()
    label_props = (df["veracity"].value_counts(normalize=True) * 100).to_dict()

    # Text Preprocessing Function
    stop_words = set(stopwords.words("english"))
    def preprocess_text(text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        return words

    df["processed_claims"] = df["claim"].apply(preprocess_text)

    # Top 40 Keywords
    all_words = [word for claim in df["processed_claims"] for word in claim]
    word_counts = Counter(all_words).most_common(40)
    top_keywords = {word: count for word, count in word_counts}

    # Convert Claims to Features (Bag-of-Words)
    word_features = list(top_keywords.keys())
    def claim_to_vector(words):
        return [1 if word in words else 0 for word in word_features]

    X = np.array(df["processed_claims"].apply(claim_to_vector).tolist())
    y = df["veracity"].values

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = rf_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON compatibility
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Baseline Random Predictions
    random_preds = np.random.choice(np.unique(y_train), size=len(y_test), p=np.bincount(y_train) / len(y_train))
    baseline_f1 = f1_score(y_test, random_preds, average="macro")

    # Return results as a dictionary
    return {
        "veracity_counts": label_counts,
        "veracity_proportions": label_props,
        "top_keywords": top_keywords,
        "confusion_matrix": conf_matrix,
        "macro_f1_random_forest": macro_f1,
        "macro_f1_random_baseline": baseline_f1
    }

# Example Usage:
# data = pd.read_csv("dat_claims.csv")  # Load dataset
# results = analyze_dataset(data, "checkcovid")
# print(results)
