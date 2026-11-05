"""Cross-validate each model, then score the best on a held-out test set."""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .data import load_dataframe, split
from .models import build_models


def run(seed: int = 42):
    df = load_dataframe()
    X_train, X_test, y_train, y_test = split(df, seed=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    rows = []
    models = build_models(seed)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()})
    results = pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)

    best_name = results.loc[0, "model"]
    best = models[best_name].fit(X_train, y_train)
    report = classification_report(y_test, best.predict(X_test))
    return results, best_name, report
