#!/usr/bin/env bash

set -eux

pip install .
pre-commit install
python tools/generate_opset.py
cat $(git diff)
git diff --exit-code
