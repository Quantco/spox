=================
Unstable features
=================

.. warning:: This part of Spox is unstable and the mentioned features are subject to change. However, they may prove useful when performing more advanced programming with ONNX and their capability `should` not change significantly in the future.

Custom operators
================

To define a custom operator, the following are required:

- Operator class

  - This is a type that inherits from ``Node`` – or ``StandardNode`` (if it has an operator schema in the standard).
  - It has to define the following three member classes as class attributes:

    - ``Attributes`` – ``@dataclass`` subclass of ``BaseAttributes`` with type-hinted members named like the respective attributes and typed with subtypes of ``Attr``, like ``AttrStrings`` or ``AttrInt``.
    - ``Inputs`` and ``Outputs`` – ``@dataclass`` subclasses of ``BaseInputs`` & ``BaseOutputs`` with members type-hinted as ``Var``, ``Optional[Var]`` (optional), or ``Sequence[Var]`` (variadic). The ordering of inputs/outputs is significant. ONNX deems that only a suffix may be optional, and only the last field may be variadic.

  - It also defines an ``op_type`` class attribute, of the form ``OpType(operator_name, operator_domain, domain_version)``.
  - Optionally, ``attrs``, ``inputs``, and ``outputs`` members should be type-hinted as the defined classes.
  - May override ``Node.infer_output_types`` and ``Node.propagate_values`` that may use the ``attrs`` and ``inputs`` members and return a dictionary from output names into types/values.

    - Lack of a type inference routine may prove detrimental, as no graph without a declared output type may be built
    - Value propagation is a routine helper to type inference which essentially performs constant folding – if all input values are known, then the output values should be computed.

- Operator constructor

  - Though an operator constructor is essentially a convenience function, it is recommended that it creates *(and does not return)* an instance of the operator class - only unwrapping the required ``outputs`` member.
  - The previously defined ``Attributes``, ``Inputs`` and ``Outputs`` should be instantiated with arguments passed to the constructor.

    - For your specific case try to find an existing generated constructor, as there may be extra actions required.

Definitions of operators may always be found in the generated opset modules, which update with Spox. A simple example may be found in ``tests/test_custom_operator.py``.

Functions
=========

ONNX functions are primarily used to avoid repeating the function body in the output graph and to define equivalent replacements for operators not defined by a runtime. Their inputs and outputs do not have type constraints – but once a function is used as an operator, it should type properly.

When functions are constructed in Spox, they will run type inference - and as such run through the entire body. Defining a function is done when a special ``Node`` subclass is constructed. The function body, however constructed, must be constructed deterministically. This is checked at build time by comparing the built protobuf.

Basic functions are supported with the ``@to_function(name, domain, version=0)`` decorator. The decorated function must take only ``Var`` arguments and return an ``Iterable[Var]``.

..  code:: python

    from typing import List
    from spox import argument, Tensor, Var
    from spox._function import to_function
    import spox.opset.ai.onnx.v17 as op

    @to_function("IsNaN", "spox.test")
    def isnan(v: Var) -> Iterable[Var]:
        return [op.not_(op.equal(v, v))]

    a = argument(Tensor(float, ()))
    (b,) = isnan(a)
    assert b.type == Tensor(bool, ())
    # In a built graph with b, IsNaN becomes a function

More advanced functions (for example including attributes) are implemented like custom operators, but by inheriting from ``Function`` and overriding the ``constructor`` method – which takes a description of attributes (which are passed as attribute references to the function's attributes) and the inputs, and returns the outputs.

Functions are tested in ``tests/test_function.py``.

.. warning::
   Functions do not have full support in the mainline ONNX Runtime. You may encounter obscure bugs when using them, especially when using attributes or nesting them. As this limits the possibility of testing them, they may be buggy in Spox as well. Support for attribute references and nested functions in particular is very limited.

Value propagation backend
=========================

By default, the internal value propagation mechanism is implemented by calling the reference implementation (``onnx.reference``). This implementation is currently incomplete and sometimes yields unexpected results. It is possible to switch the backend to ``onnxruntime`` instead by calling ``_future.set_value_prop_backend(_future.ValuePropBackend.ONNXRUNTIME)``. This is a global switch - it is expected that the same choice will be used throughout the program.

Type inference warning levels
=============================

As the use case may choose to have different behaviour on type errors or otherwise missing types, this is controllable via ``_future.set_type_warning_level`` called with members of the ``_future.TypeWarningLevel`` enum:

1. `None` - no type check warnings at all
2. `Critical` - warn on missing output types (type is None)
3. `Initial` (default) - warn on and incomplete output types, but only when all the input types were known; also warn on all missing output types
4. `Outputs` - warn on all output types that are missing or incomplete

It may be useful to note that if absolutely necessary the ``_internal_op.unsafe_cast`` and ``_internal_op.unsafe_reshape`` functions may be used to forcibly convert types. As a last resort, the ``Var.type`` field may be mutated - but this is not recommended and is less safe than ``unsafe_cast``.

The most common cause of missing or incomplete types is a missing type inference implementation in ONNX. In some cases Spox adds a `type inference patch` that attempts to fix this. In rare cases it is impossible to determine the type (like a dynamic reshape) - it is recommended to use the above cast functions as early as possible to get type checks back.

Initializers
============

Initializers are an ONNX mechanism for expressing constant tensor values in the model. Spox provides a function to create them:

..  code:: python

    def initializer(value: ArrayLike, dtype: DTypeLike = None) -> Var:
        ...

This is a mechanism alternative to the ``ai.onnx::Constant`` operator. In our experience initializers sometimes have worse support, which is why they remain unstable. Additionally, both methods essentially achieve the same result.


Operator overloading
====================

Currently, Spox does not support operator overloading on ``Var`` out of the box. This is because it's hard to set conventions that work for every use case.

An experimental implementation may be invoked via the ``spox._future.operator_overloading`` context manager - which may be used both in a ``with`` block and as a decorator. When initialised with an ``ai.onnx`` opset module, it enables ``Var`` to use operator overloads. When constant promotion is enabled, Python constants (like ``1`` or ``2.7``) are automatically wrapped in ``constant``. Additionally, type promotion may be enabled (but is disabled by default). In both cases numpy typing rules are used.

..  code:: python

    import numpy as np
    from spox.opset.ai.onnx import v17 as op
    x, y = op.const(2), op.const(3)
    with operator_overloading(op):
        z = x + y
    assert z._get_value() == np.array(5)
    @operator_overloading(op)
    def foo():
       return x * y
    assert foo()._get_value() == np.array(6)

This implementation is tested in ``tests/future/test_var_operators.py``.
