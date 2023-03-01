# Spox

[![CI](https://github.com/Quantco/spox/actions/workflows/ci.yml/badge.svg)](https://github.com/Quantco/spox/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/spox/badge/?version=latest)](https://spox.readthedocs.io/en/latest/?badge=latest)

Spox makes it easy to construct [ONNX](https://github.com/onnx/onnx/) models through clean and idiomatic Python code.

## Why use Spox?

A common application of ONNX is converting models from various frameworks. This requires replicating their runtime behaviour with ONNX operators.
In the past this has been a major challenge.
Based on our experience, we designed Spox from the ground up to make the process of writing converters (and ONNX models in general) as easy as possible.

Spox's features include:

- Eager operator validation and type inference
- Errors with Python tracebacks to offending operators
- First-class support for subgraphs (control flow)
- A lean and predictable API

## Installation

Spox releases are available on PyPI:

```bash
pip install spox
```

There is also a package available on conda-forge:

```bash
conda install spox
```

## Quick start

In Spox, you primarily interact with `Var` objects - **variables** - which are placeholders for runtime values.
The initial `Var` objects, which represent the _arguments_ of a model (the model inputs in ONNX nomenclature), are created with an explicit type using the `argument(Type) -> Var` function. The possible types include `Tensor`, `Sequence`, and `Optional`.
All further `Var` objects are created by calling functions which take existing `Var` objects as inputs and produce new `Var` objects as outputs. Spox determines the `Var.type` for these eagerly to allow validation.
Spox provides such functions for all operators in the standard. They are grouped by domain and version in the `spox.opset` submodule.

The final `onnx.ModelProto` object is built by passing input and output `Var`s for the model to the `spox.build` function.

Below is an example for defining an ONNX graph which computes the [geometric mean](https://en.wikipedia.org/wiki/Geometric_mean) of two inputs.
Make sure to consult the Spox [documentation](https://spox.readthedocs.io/en/latest) to find more details and tutorials.

```python
import onnx

from spox import argument, build, Tensor, Var
# Import operators from the ai.onnx domain at version 17
from spox.opset.ai.onnx import v17 as op

def geometric_mean(x: Var, y: Var) -> Var:
    # use the standard Sqrt and Mul
    return op.sqrt(op.mul(x, y))

# Create typed model inputs. Each tensor is of rank 1
# and has the runtime-determined length 'N'.
a = argument(Tensor(float, ('N',)))
b = argument(Tensor(float, ('N',)))

# Perform operations on `Var`s
c = geometric_mean(a, b)

# Build an `onnx.ModelProto` for the given inputs and outputs.
model: onnx.ModelProto = build(inputs={'a': a, 'b': b}, outputs={'c': c})
```

## Credits

Original designed and developed by [@jbachurski](https://github.com/jbachurski) with the supervision of [@cbourjau](https://github.com/cbourjau).
