"""
Module for Spox operators that implement special internal behaviour that does not fit into the ONNX IR.
They behave like a normal Node, but their inference, building and translation behaviour may be overriden.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import onnx

from ._attributes import AttrString, AttrTensor, AttrType
from ._fields import BaseAttributes, BaseInputs, BaseOutputs
from ._node import Node, OpType
from ._scope import Scope
from ._shape import SimpleShape
from ._type_system import Tensor, Type
from ._value_prop import PropValueType
from ._var import Var

# This is a default used for internal operators that
# require the default domain. The most common of these
# is Introduce, which is effectively used in every graph.
# It implements an 'opset-independent' `Identity`.
#
# At the time of writing, there are three versions for
# `Identity` in the standard: 1, 13 and 14. 1 and 13 only
# cover `tensor`s. 14 includes `seq` as well. No version
# currently supports `map`. Another constraint at play is that
# ORT will throw warnings when encountering any version less
# than 7. If we want to avoid spamming the user with warnings
# in simple test cases, we ought to use 7 or higher.
#
# Version 7 was released some time in May 2018, version 13 in
# Nov 2020, and version 14 in April 2021. The only version
# that users can currently choose via `spox.opset.ai.onnx` is
# version 17 (released June 2022) and I doubt we will ever
# decrease that.
#
# Putting it all together, it seems reasonable to set this to
# 14. That version is almost 2 years old at this point and
# spares us any special casing if we want to use `seq`.
# This effectively sets the lower bound on every model.
INTERNAL_MIN_OPSET = 14


class _InternalNode(Node, ABC):
    @property
    def opset_req(self) -> Set[Tuple[str, int]]:
        return set()


class Argument(_InternalNode):
    """
    Internal operator representing the source of an argument.

    - The ``type`` has to be set, as otherwise the graph would be malformed.
    - If ``name`` is undeclared, it may be set to anything by the build (useful for subgraphs where order is used).
    - Additionally, an argument may have a ``default`` (an initializer) -
      but note that ONNX Runtime only supports non-overridable initializers (implemented by Initializer).
    """

    op_type = OpType("Argument", "spox.internal", 0)

    @dataclass
    class Attributes(BaseAttributes):
        type: AttrType
        name: Optional[AttrString] = None
        default: Optional[AttrTensor] = None

    @dataclass
    class Inputs(BaseInputs):
        pass

    @dataclass
    class Outputs(BaseOutputs):
        arg: Var

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

    def post_init(self, **kwargs):
        if self.attrs.name is not None:
            self.outputs.arg._rename(self.attrs.name.value)

    def infer_output_types(self) -> Dict[str, Type]:
        # Output type is based on the value of the type attribute
        return {"arg": self.attrs.type.value}

    def update_metadata(self, opset_req, initializers, functions):
        super().update_metadata(opset_req, initializers, functions)
        var = self.outputs.arg
        if self.attrs.default is not None:
            initializers[var] = self.attrs.default.value

    def to_onnx(
        self, scope: "Scope", doc_string: Optional[str] = None, build_subgraph=None
    ) -> List[onnx.NodeProto]:
        return []


class _Initializer(_InternalNode):
    """Internal operator representing a non-overridable initializer."""

    op_type = OpType("Initializer", "spox.internal", 0)

    @dataclass
    class Attributes(BaseAttributes):
        value: AttrTensor

    @dataclass
    class Outputs(BaseOutputs):
        arg: Var

    attrs: Attributes
    inputs: BaseInputs
    outputs: Outputs

    def infer_output_types(self) -> Dict[str, Type]:
        # Output type is based on the value of the type attribute
        arr = self.attrs.value.value
        return {"arg": Tensor(arr.dtype, arr.shape)}

    def propagate_values(self) -> Dict[str, PropValueType]:
        return {"arg": self.attrs.value.value}

    def update_metadata(self, opset_req, initializers, functions):
        super().update_metadata(opset_req, initializers, functions)
        initializers[self.outputs.arg] = self.attrs.value.value

    def to_onnx(
        self, scope: "Scope", doc_string: Optional[str] = None, build_subgraph=None
    ) -> List[onnx.NodeProto]:
        # Initializers are added via update_metadata and don't affect the nodes proto list
        return []


class _Introduce(_InternalNode):
    """Internal operator used for introducing values, to manually evaluate them in the current scope."""

    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        inputs: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        outputs: Sequence[Var]

    op_type = OpType("Introduce", "spox.internal", 0)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

    def infer_output_types(self) -> Dict[str, Type]:
        return {
            f"outputs_{i}": arr.type
            for i, arr in enumerate(self.inputs.inputs)
            if arr.type is not None
        }

    @property
    def opset_req(self) -> Set[Tuple[str, int]]:
        return {("", INTERNAL_MIN_OPSET)}

    def to_onnx(
        self, scope: Scope, doc_string: Optional[str] = None, build_subgraph=None
    ) -> List[onnx.NodeProto]:
        assert len(self.inputs.inputs) == len(self.outputs.outputs)
        # Just create a renaming identity from what we forwarded into our actual output
        protos = []
        name = scope.node[self]
        for i in range(len(self.inputs.inputs)):
            protos.append(
                onnx.helper.make_node(
                    "Identity",
                    [scope.var[self.inputs.inputs[i]]],
                    [scope.var[self.outputs.outputs[i]]],
                    name + f"_id{i}",
                    doc_string,
                )
            )
        return protos


def intros(*args: Var) -> Sequence[Var]:
    """
    Internal identity operator with variadic arguments.

    As the underlying node is dependent on all passed arguments, this can be used to enforce specific evaluation order
    for values used in a subgraph - but otherwise ignored.

    For example, in a Loop whose body uses some ``x``, ``x`` may only be built within the subgraph and hence
    reevaluated on every iteration. If the Loop is wrapped with ``intro(x, loop(...))`` it is guaranteed that ``x``
    will be built outside of Loop's subgraph. It can be said that ``x`` was `introduced` in the outer scope.

    Parameters
    ----------
    args
        Vars to introduce in current scope.

    Returns
    -------
    Sequence[Var]
        Vars of the same value as ``args``, but with a shared dependency.
    """
    return _Introduce(
        None, _Introduce.Inputs(args), out_variadic=len(args)
    ).outputs.outputs


def intro(*args: Var) -> Var:
    """Introduces arguments like ``intros``, but only returns the last."""
    return intros(*args)[-1]


def unsafe_cast(x: Var, typ: Type) -> Var:
    """
    Creates a new var with the type forcefully set to ``typ``.

    Assumes that the real type of the Var is indeed compatible with ``shape`` (for example it was unknown).

    The function is meant for use when type inference failed, and it has to be overriden to avoid further failures.

    If you want to properly change a ``Var``'s type, use an operator like Cast, CastLike, Optional, etc.

    Parameters
    ----------
    x
        Var to retype.
    typ
        Target type - must be a constant.

    Returns
    -------
    Var
        Var with the type reset to whatever was given.
    """
    y = intro(x)
    y.type = typ
    y._value = x._value
    return y


def unsafe_reshape(x: Var, shape: SimpleShape) -> Var:
    """
    Creates a new var with the shape forcefully set to ``shape`` (like an unsafe cast).

    Assumes that the real shape of the Var is indeed compatible with ``shape`` (for example it was unknown).

    The function is meant for use when shape inference failed, and it has to be overriden to avoid failures.

    If you want to reshape to the shape of another var, use a Reshape operator.

    Parameters
    ----------
    x
        Var to reshape.
    shape
        Target shape - must be a constant.
    Returns
    -------
    Var
        Var with the same Tensor element type, but different shape.
    """
    return unsafe_cast(x, Tensor(x.unwrap_tensor().dtype, shape))
