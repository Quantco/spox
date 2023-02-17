=========
Operators
=========

Introduction
============

*Operators* are the operational primitives in the ONNX standard.

Though in ONNX they are identified with nodes in a computational graph, in Spox this is abstracted away via the use of `operator constructors`. A programmer only passes around variables (``Var``), which operator constructors take and return.

Representation
==============

On the fundamental level, operators have attributes, inputs and outputs. For an ONNX runtime, operators are the elementary operations (instructions) that are performed within the model.

Example operators include:

- Arithmetic - ``Add``, ``Mul``, ``MatMul``, ...
- Tensor/ndarray operations - ``Reshape``, ``Concat``, ``Gather``, ``Scatter``, ...
- ML/DL-specific operations like ``AveragePool``, ``OneHot``, ...

Attributes
----------

Attributes are strongly typed constants known at build time – similarly to *template parameters* for the operator. They often affect the operation performed by the operator, including what type or shape the output is, and may not be runtime-dependent (contrary to operator `inputs`).

They may be either required or optional (in which case they may have a default). Another common construct is `one-of attributes`, where only one of a given group of arguments may be set. Such more complex logic is usually documented in the docstring of the operator as it is not directly expressible in either a Python signature or ONNX itself.

In Spox, attributes are keyword arguments to operator constructors. They expect Python and numpy datatypes to avoid the inconvenience of constructing ONNX protobuf objects. Whenever a constructor takes an attribute the type-hint explicitly specifies which type is required. Passing the wrong attribute type is an error, as using it would cause the model to fail validation.

For example, an ``ints``-type attribute becomes ``Iterable[int]``, while ``dtype`` in an operator like ``Cast`` becomes ``numpy.typing.DTypeLike``.

.. note::
   The value of the default attribute will be included with the output (unless it is optional without a default, i.e. ``None``). A model optimiser is expected to strip this information.

.. warning::
   Sparse tensors and tensors referred to by filename are currently not supported in Spox. Workarounds on the ONNX output are required until they are added.

Inputs & outputs
----------------

Inputs are like *function arguments*. Inputs are a list of variables which indicate what values computed by the runtime should be passed in to the operator.

Outputs are like *function results*. Outputs are also a list of variables, to which we assign the newly computed values, after the operator is executed by the runtime.

In the representation inputs and outputs are specified positionally. Because of this, inputs are positional arguments and outputs are either a single variable or a tuple.

Runtime-dependent values (variables) – such as operator inputs and outputs – are represented with immutable ``Var`` objects in Spox. Internally, ``Var`` store the information (their ancestry) necessary to recover the computational graph to compute them.

Inputs and outputs may also be variadic or optional.

- Variadic indicates that we may have a (usually arbitrary) number of variables passed in for this input. Since in ONNX protobuf inputs/outputs are just a homogenous positional list, only the last argument may be variadic. However, in Spox a ``Sequence[Var]`` is used to represent this case. As a variadic output's length may be ambiguous, this is resolved at generation-time either with a `output variadic solution patch`, or by generating an extra argument to specify this.
- Optional implies the input may be omitted. This is slightly inconsistent within the ONNX representation, as there are two ways to omit an argument: either it can be not provided in the argument list or instead an "empty variable" (empty string for the name) may be passed in. The generated Spox constructors make a best attempt to abstract this out, and optionals are represented with ``Optional[Var]``.

Multiple outputs of an operator are returned as a tuple, with its elements typed as above.

.. note::
   Though optional outputs are allowed by the standard, Spox does not support them. All the possible outputs will be named in the model. It is expected that model optimisers will remove references to unused outputs.

A simple example would be ``Reshape`` with two inputs: ``data: Var`` and ``shape: Var``. On the other hand, ``Concat`` has a variadic input ``inputs: Sequence[Var]``, and ``Split``'s variadic output is returned as ``Sequence[Var]``.

Schema
------

Information about an operator is summarised in the `operator schema`, which is programmatically accessible from the ``onnx`` Python module. Spox does not directly expose schemas, but operator constructors (including their docstrings) are generated based on them.

.. warning::
   Spox makes a best effort to figure out how to interpret schemas. However, not all operators are tested by hand and some may not be instantiable in some forms which are actually allowed by the standard. In that case, instantiating them in another way and using an inlined model may be required.

Usage
=====

The best way to use an opset is to import it like ``import spox.opset.ai.onnx.v17 as op``. That way, all the public functions from the module are the operator constructors for the relevant domain – in this example, ``ai.onnx`` (the default domain, sometimes written as the empty string ``""``) at version 17.

An example signature is:

..  code:: python

    def reshape(
        data: Var,
        shape: Var,
        *,
        allowzero: int = 0,
    ) -> Var:
        ...

Operator constructors may be used in any way that is legal in Python, As such, constructs such as ``functools.reduce(op.mul, vars)`` are legal to express a product of all the variables ``vars``.

No state of ``Var`` or other internal Spox objects is modified. Breaking this rule should be considered undefined behaviour.

In operator constructors Spox expects exactly the types that are specified in the type hints. No other types should be passed in.
In particular, ``Var`` subclasses are currently not supported.

Docstrings
----------

The docstrings are automatically generated based on the operator schema. These usually describe the runtime behaviour and in part the typing rules. All of the attributes, inputs and outputs may also have docstrings which are included accordingly.

There is also rudimentary type inference information in the form of basic type variables and constraints – usually including just the element type of a tensor. The type information is listed along with the input/output docstrings and in the Notes section.

Docstrings are not handwritten by the Spox developers and are based on what the ONNX schema contains. As such, it may e.g. contain inconsistent formatting, especially for code blocks. These shortcomings will be ironed out over time.

Operator renames
-----------------

To follow Python conventions, operator constructors are renamed to follow PEP8. This is done by renaming to snake-case, by prepending underscores before capitals at the start of words, followed lower-casing all characters. If the result is a Python keyword (like in the case of ``if``, ``and``, ``or``, ``not``), an underscore ``_`` is appended.

.. note::
   This naming scheme causes some operators (like ``min``, ``max``, ``abs``, ``range``, ...) to shadow builtin Python functions. A programmer may choose to alias them to another name when they are imported directly to avoid this issue. Additionally, ``IsInf``, ``IsNaN``, and ``MatMul`` are hard to predictably get right. They are called ``isinf``, ``isnan``, and ``matmul`` - like in numpy.


Data type attributes
--------------------

In standard ONNX an operator like ``Cast`` takes an ``int`` to express the datatype of the resulting tensor. Spox overrides this behaviour to instead take a value of ``numpy.typing.DTypeLike``.  This includes values like ``float``, ``np.int16``, ``np.dtype('bool')``. The type hint is changed accordingly, but the docstring may suggest otherwise.

Also note that Spox represents types using ``numpy`` datatypes. As such, it also follows its typing conventions - a Python ``float`` becomes ``float64``, and an ``int`` becomes ``int64``. This may be unexpected, as ONNX may prefer ``float32`` in some contexts.

Spox strays from ONNX in the representation of the string datatype. ``np.str_``/``np.dtype(str)`` is used instead of ``np.object_``/``np.dtype(object)``. For example, a ``Cast`` to a string type is expressible as ``op.cast(x, to=str)``.

Subgraphs
---------

To implement control flow in a computational graph the ONNX standard introduces `graph attributes`, which work like subprograms.

Spox abstracts this away by instead expecting a callable of the right signature which the subgraph will be constructed from. The callable takes some number of arguments (the number of subgraph inputs), which are ``Var``, and returns an ``Iterable[Var]``.

For example, since ``If``'s subgraphs (``then_branch`` and ``else_branch``) take no arguments but both return some *n* variables, a valid construct would be ``op.if_(cond, lambda: (x, y), lambda: (y, x))``, where ``cond``, ``x``, ``y`` are ``Var``.

The passed callbacks will be called once to determine what the subgraph body is. The arguments for the subgraph are constructed implicitly.

.. note::
   ONNX subgraphs have name scoping rules – as such, a subgraph may access the variables from outer graphs, but a graph may not access its subgraph's variables.

   Currently Spox uses a resolution method to find scoping which matches all of the operator and subgraph constraints. This ONNX scoping may not directly follow from what the respective Python scopes were, as it is difficult to reliably detect where a Python variables is instantiated.

   This behaviour should be treated as unstable. It is recommended to avoid side effects within subgraph callbacks.

Type inference
--------------

One of the main features of Spox is eager type inference and checks. This is primarily implemented for standard operators (as provided), with a potential for overriding it in custom operators. This uses the ONNX utility ``onnx.shape_inference.infer_shapes``, which on the C-level calls ``TypeAndShapeInferenceFunction`` as defined in C++. Because of this, some errors may not be properly formatted. Spox makes attempts to extend the error messages with notes on passed types to improve the experience.

It is a common occurrence that type inference is missing or in some way partial. Spox will in that case make attempts to warn the user in conditions it deems unsafe or unexpected. Lack of type inference is not a bug within Spox itself, but capability for `patching type inference` is exposed, and this is done for several operators.

.. note::
   In the future access to warning levels will be exposed to modify this behaviour.
