=====================
Experimental features
=====================

.. warning:: This part of Spox is experimental and the mentioned features are subject to change. However, they may prove useful when performing more advanced programming with ONNX and their capability should not change significantly in the future.

Custom operators
================

As Spox makes rich usage of operator schemas, not every protobuf node is allowed. To define a custom operator, the following are required:

- Operator class
    - This is a type that inherits from ``Node`` – or ``StandardNode`` if it has an operator schema in the standard.
    - It has to define the following three member classes as class attributes:
        - ``Attributes`` – a ``@dataclass`` with type-hinted members named like the respective attributes and typed with subtypes of ``Attr``, like ``AttrStrings`` or ``AttrInt``.
        - ``Inputs`` and ``Outputs`` – subclasses of ``VarFields`` *(likely subject to change)* with members type-hinted as ``Var``, ``Optional[Var]`` (optional), ``Sequence[Var]`` (variadic). Note that primarily the ordering of inputs/outputs is significant.
    - An ``op_type`` class attribute should be defined as ``OpType(operator_name, operator_domain, domain_version)``.
    - Optionally, ``attrs``, ``inputs``, and ``outputs`` members should be typed as the defined classes.
    - May override ``Node.infer_output_types`` and ``Node.propagate_values`` that may use ``attrs`` and ``inputs`` and return a dictionary from output names into types/values.
        - Lack of a type inference routine may prove detrimental, as no graph without a declared output type may be constructed.
        - Value propagation is a routine helper to type inference which essentially performs constant folding – if all input values are known, then the output values should be computed.
- Operator constructor
    - Though an operator constructor is essentially a convenience function, it is recommended that it creates *(and does not return)* an instance of the operator class, unwrapping the required ``outputs`` member.
    - The previously defined ``Attributes``, ``Inputs`` and ``Outputs`` should be instantiated with arguments passed to the constructor.
        - For possible instances of housekeeping for this purpose refer to opset modules. *(possibly subject to change)*

Model definitions of operators may always be found in the generated opset modules, as those operators are defined in the same way. A simple example should be found in ``tests/test_custom_operator.py``.

Functions
=========

ONNX functions are implemented mainly to avoid repeating the function body. Their inputs and outputs do not have type constraints – but once a function is used as an operator, it should type properly. When functions are defined in Spox, they will run type inference. The function body, however constructed, must be constructed deterministically. This is checked at build time by comparing the built protobuf.

Basic functions are supported with the ``@to_function(name, domain, version=0)`` decorator. The decorated function must take only ``Var`` arguments and return and ``Iterable[Var]``.

More advanced functions (for example including attributes) are implemented like custom operators, but by inheriting from ``Function`` and overriding the ``constructor`` method – which takes a description of attributes, the inputs and returns the outputs.

.. warning::
   Functions do not have full support in the mainline ONNX Runtime. You may encounter obscure bugs when using them, especially when using attributes or nesting them. As this limits the possibility of testing them, they may be buggy in Spox as well.

Eager execution
===============

The ``spox._standard._USE_ONNXRUNTIME_VALUE_PROP`` flag when set to ``True`` enables using the ONNX runtime to run value propagation.

This essentially means that all expressions which are constant (there are no arguments/runtime-dependent values) may be computed. For example, ``op.add(op.const(2), op.const(2))._value == numpy.array(2)``.

The feature is useful when experimenting with operators.

.. note::
   For some operators an exception may be raised when the value propagation produces a value incompatible with the inferred type. This usually indicates a bug within ONNX type inference, ONNX Runtime execution (interpretation of the standard), or a Spox type inference patch.

.. note::
   In the future an equivalent feature may become enabled by default if there is no dependency on ONNX Runtime. The only patched-in value propagation routine is for ``Constant`` (so that ``Reshape`` into a constant shape succeeds in type inference).

   The relevant concept within the ONNX standard is partial data propagation and as it is not exposed directly within the Python ``onnx`` module it is not used in Spox.