#!/usr/bin/env bash

set -eux

pip install .
pre-commit install
python tools/generate_opset.py
git diff --exit-code
