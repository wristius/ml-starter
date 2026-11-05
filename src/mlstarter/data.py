"""Load and prepare the dataset (the 'data wrangling' step)."""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_dataframe() -> pd.DataFrame:
    """Return the UCI Wine dataset as a DataFrame with a 'target' column."""
    bunch = load_wine(as_frame=True)
    df = bunch.frame.copy()
    df["target"] = df["target"].astype(int)
    return df


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split so class balance is preserved."""
    X = df.drop(columns="target")
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
