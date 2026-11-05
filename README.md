# ml-starter

A small baseline-modeling project: load a dataset, compare three classic
classifiers with cross-validation, and report the best one on a held-out test
set. It's meant to be read, run and extended.

## Run it

```bash
./run.sh
```

That creates a virtualenv, installs dependencies, runs the tests, then runs the
comparison. Results land in `outputs/` (a CSV and a chart).

## Layout

```
src/mlstarter/
  data.py       load + stratified split
  models.py     logistic regression, k-NN, random forest (scaling inside pipelines)
  evaluate.py   cross-validation + held-out test report
  __main__.py   writes outputs/ and prints the summary
tests/          pytest checks for data, split, models and the pipeline
run.sh          setup + test + run in one command
```

## Ideas to extend

- Swap in a dataset from your own coursework
- Add hyperparameter tuning (`GridSearchCV`)
- Add a small neural network and compare it to the baselines
- Add a confusion-matrix plot or feature-importance chart
