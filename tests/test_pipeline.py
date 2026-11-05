from mlstarter.data import load_dataframe, split
from mlstarter.evaluate import run
from mlstarter.models import build_models


def test_dataframe_shape_and_target():
    df = load_dataframe()
    assert df.shape == (178, 14)
    assert set(df["target"].unique()) == {0, 1, 2}


def test_split_is_stratified_and_disjoint():
    df = load_dataframe()
    X_train, X_test, y_train, y_test = split(df)
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(y_test.unique()) == {0, 1, 2}


def test_all_models_are_built():
    assert set(build_models()) == {"logistic_regression", "knn", "random_forest"}


def test_run_beats_chance_comfortably():
    results, best_name, _ = run()
    assert best_name in results["model"].values
    assert results["cv_mean"].max() > 0.9  # chance is ~0.40
