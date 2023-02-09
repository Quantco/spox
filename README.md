# Spox

[![CI](https://github.com/Quantco/spox/actions/workflows/ci.yml/badge.svg)](https://github.com/Quantco/spox/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-success?style=plastic)](https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/spox/latest/index.html)

Spox is a Python framework for constructing [ONNX](https://github.com/onnx/onnx/) computational graphs.

Spox:

- Implements the ONNX standard while also allowing Pythonic code.
  - Spox follows conventions set out by numpy and other Python libraries.
- Enforces the strong type system of ONNX, by raising errors with Python tracebacks to the offending operator.
  - Checks are performed as eagerly as possible!
- Supports the entirety of modern opsets, including features like subgraphs (control flow) and types other than tensors (like sequences and optionals).
  - Standard operators all have typed Python signatures and docstrings!
- Is designed for predictability. No mutable types are passed around, so it's difficult to invalidate the graph accidentally.
  - If it's legal Spox, it (should be) legal ONNX!

The main goal of Spox is to provide a robust and Pythonic framework for developing libraries building ONNX graphs, such as converters or other custom applications.

## Installation

Spox is published on conda-forge and can be installed as expected:

```bash
conda install spox
```

## Getting started

### Constructing

In Spox, most of the time you'll be working with `Var` objects - **variables**. You either create them yourself as arguments for a model, or receive them from another source, for example when you're writing a converter in a library.

You may print out the `Var` or check its `Var.type` to learn a bit more about it.

To perform operations on a `Var`, use an _opset_ (operator set) module like `spox.opset.ai.onnx.v17` or `spox.opset.ai.onnx.ml.v3` - which correspond to `ai.onnx@17` and `ai.onnx.ml@3`, the standard opsets. These are pre-generated for you and you may import them as required.

For instance, using the default opset you could write a function that given two variables returns their [geometric mean](https://en.wikipedia.org/wiki/Geometric_mean):

```python
from spox import Var
from spox.opset.ai.onnx import v17 as op

def geometric_mean(x: Var, y: Var) -> Var:
    return op.sqrt(
      op.mul(x, y)
    )
```

Since ONNX is tensor-oriented, this code assumes that `x` and `y` are floating point tensors with matching (broadcastable) shapes, and the geometric mean is computed elementwise. You may learn more about what types and shapes are acceptable for given operators by checking their docstring.

Performing operations on Variables creates further Variables, all of which keep track of what computation has been performed so far. If an operation is determined to be illegal (for example due to mismatching types/shapes), an exception will be immediately raised by Spox.

### Building

To introduce _argument_ variables (which are placeholders for runtime values - graph/model inputs in ONNX nomenclature), you may use the `argument(typ: Type) -> Var` function with the required type, like `Tensor`.

When you're done you may build an `onnx.ModelProto` - the model protobuf, which is the ONNX program representation. You may do this with the public `build(inputs, outputs) -> onnx.ModelProto` function:

```python
from spox import argument, build, Tensor, Var

def geometric_mean(x: Var, y: Var) -> Var: ...

a = argument(Tensor(float, (1, 'N')))
b = argument(Tensor(float, ('M', 1)))

c = geometric_mean(a, b)

model = build({'a': a, 'b': b}, {'c': c})
```

The keys of passed dictionaries are used to _name_ the arguments and results you would like the model to have. Keep in mind that `inputs` may only contain `Var` returned by `argument`.

Building the model may be done for you if you're writing a component of a library using Spox.

### Running

Afterwards, you may use an ONNX runtime/backend to execute the model with some inputs, for example the [ONNX Runtime](https://github.com/microsoft/onnxruntime).

You can learn more about Spox in the [documentation](https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/spox/latest/index.html).
