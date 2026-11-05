"""Baseline models to compare. Each is wrapped in a pipeline so scaling is
fit only on training folds (avoids data leakage)."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(seed: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "knn": Pipeline(
            [("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]
        ),
        "random_forest": Pipeline(
            [("clf", RandomForestClassifier(n_estimators=200, random_state=seed))]
        ),
    }
