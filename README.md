# Spox

[![CI](https://github.com/Quantco/spox/actions/workflows/ci.yml/badge.svg)](https://github.com/Quantco/spox/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-success?style=plastic)](https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/spox/latest/index.html)

Spox is a Python framework for constructing [ONNX](https://github.com/onnx/onnx/) computational graphs.

Spox:

- Closely follows the ONNX standard while also allowing Pythonic code.
  - Spox follows ONNX conventions first, then numpy and other Python libraries.
- Enforces the strong type system of ONNX, by raising errors with Python tracebacks to the offending operator.
  - Checks are performed as eagerly as possible!
- Supports the entirety of modern opsets, including features like subgraphs (control flow) and types other than tensors (like sequences and optionals).
  - Standard operators all have typed Python signatures and docstrings!
- Is designed for predictability. No mutable types are passed around, so it's difficult to invalidate the graph.
  - If it's legal Spox, it should be legal ONNX!

The main goal of Spox is to provide a robust and Pythonic framework for developing libraries building ONNX graphs, such as converters or other custom applications.

## Installation

Spox is published on conda-forge and can be installed as expected:

```bash
conda install spox
```
