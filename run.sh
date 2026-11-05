#!/usr/bin/env bash
# One-command workflow: set up a virtualenv, install deps, test, then run.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet -r requirements.txt

export PYTHONPATH="$PWD/src"
python -m pytest -q
python -m mlstarter
