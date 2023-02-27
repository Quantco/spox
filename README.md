# Spox

[![CI](https://github.com/Quantco/spox/actions/workflows/ci.yml/badge.svg)](https://github.com/Quantco/spox/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/spox/badge/?version=latest)](https://spox.readthedocs.io/en/latest/?badge=latest)

Spox makes it easy to construct [ONNX](https://github.com/onnx/onnx/) models through clean and idiomatic Python code.

## Why use Spox?

Converting a trained model into ONNX entails replicating its inference logic with ONNX operators.
In the past that constituted a major undertaking.
Based on those experiences we designed Spox from the ground up to make the process of writing converters (and ONNX models in general) as easy as possible.

## Installation

With pip:

```bash
pip install spox
```

Spox is also available on conda-forge and can be installed as expected:

```bash
conda install spox
```

## Quick start

Spox users interact most commonly with `Var` objects - **variables** which will have values assigned to them at runtime.
The initial `Var` objects which represent the *argument*s of a model (the model inputs in ONNX nomenclature) are created with an explicit type and shape using the `spox.argument` function.
All further `Var` objects are created by calling functions which take existing `Var` objects as inputs and produce new `Var` objects as outputs.
Spox provides such functions for all operators in the standard grouped by domain and version inside the `spox.opset` module.

The final `onnx.ModelProto` object is build by specifying the explicit input and output `Var`s of the model to in the `spox.build` function.

Below is an example for defining an ONNX graph which computes the [geometric mean](https://en.wikipedia.org/wiki/Geometric_mean) of two inputs.
Please consult the [documentation](https://spox.readthedocs.io/en/latest) for more details and further tutorials on how to use Spox.

```python
import onnx

from spox import argument, build, Tensor, Var
# import operators from the ai.onnx domain at version 17
from spox.opset.ai.onnx import v17 as op

def geometric_mean(x: Var, y: Var) -> Var:
    # use the standard Sqrt and Mul
    return op.sqrt(op.mul(x, y))

# Create typed model inputs. Each tensor is of rank 1
# and has the runtime-defined length "N".
a = argument(Tensor(float, ('N',)))
b = argument(Tensor(float, ('N',)))

# Perform operations on `Var`s
c = geometric_mean(a, b)

# Build an `onnx.ModelProto` for the given inputs and outputs.
model: onnx.ModelProto = build(inputs={'a': a, 'b': b}, outputs={'c': c})
```

## Credits

Original authors: @jbachurski with supervision from @cbourjau
