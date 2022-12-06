"""
Module for Spox operators that implement special internal behaviour that does not fit into the ONNX IR.
They behave like a normal Node, but their inference, building and translation behaviour may be overriden.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import onnx

from ._attributes import AttrString, AttrTensor, AttrType
from ._node import Node, OpType
from ._scope import Scope
from ._shape import SimpleShape
from ._type_system import Tensor, Type
from ._var import Var
from ._varfields import NoVars, VarFields


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
    class Attributes:
        type: AttrType
        name: Optional[AttrString] = None
        default: Optional[AttrTensor] = None

    class Outputs(VarFields):
        arg: Var

    attrs: Attributes
    inputs: NoVars
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
    class Attributes:
        type: AttrType
        default: AttrTensor

    class Outputs(VarFields):
        arg: Var

    attrs: Attributes
    inputs: NoVars
    outputs: Outputs

    def infer_output_types(self) -> Dict[str, Type]:
        # Output type is based on the value of the type attribute
        return {"arg": self.attrs.type.value}

    def update_metadata(self, opset_req, initializers, functions):
        super().update_metadata(opset_req, initializers, functions)
        initializers[self.outputs.arg] = self.attrs.default.value

    def to_onnx(
        self, scope: "Scope", doc_string: Optional[str] = None, build_subgraph=None
    ) -> List[onnx.NodeProto]:
        return []


class _Embedded(_InternalNode):
    """Internal operator used for embedding an existing ONNX ModelProto inside a Spox graph."""

    model: onnx.ModelProto

    @dataclass
    class Attributes:
        pass

    class Inputs(VarFields):
        inputs: Sequence[Var]

    class Outputs(VarFields):
        outputs: Sequence[Var]

    op_type = OpType("Embedded", "spox.internal", 0)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

    def pre_init(self, **kwargs):
        self.model = kwargs["model"]

    @property
    def graph(self):
        return self.model.graph

    @property
    def opset_req(self) -> Set[Tuple[str, int]]:
        return {(imp.domain, imp.version) for imp in self.model.opset_import}

    def infer_output_types(self) -> Dict[str, Type]:
        # First, type check that we match the ModelProto type requirements
        for i, var in zip(self.graph.input, self.inputs.inputs):
            if var.type is not None and not (var.type <= Type._from_onnx(i.type)):
                raise TypeError(
                    f"Embedded model input {i.name} type {var.type} "
                    f"does not match expected {Type._from_onnx(i.type)}."
                )
        # If we do, take the types as declared in the model
        return {
            f"outputs_{k}": Type._from_onnx(o.type)
            for k, o in enumerate(self.graph.output)
        }

    def to_onnx(
        self, scope: Scope, doc_string: Optional[str] = None, build_subgraph=None
    ) -> List[onnx.NodeProto]:
        # Prefix all names in the graph to try and avoid name clashes
        name = scope.node[self] if self in scope.node else None
        graph = (
            onnx.compose.add_prefix_graph(self.graph, f"{name}__")
            if name is not None
            else self.graph
        )
        # Apply a trivial renaming of inputs
        nodes = []
        for i, var in zip(graph.input, self.inputs.inputs):
            nodes.append(
                onnx.helper.make_node(
                    "Identity",
                    [scope.var[var]],
                    [i.name],
                    f"{i.name}__Identity_rename",
                )
            )
        # Then graph body
        nodes.extend(graph.node)
        # Finish with output renaming
        for o, var in zip(graph.output, self.outputs.outputs):
            nodes.append(
                onnx.helper.make_node(
                    "Identity",
                    [o.name],
                    [scope.var[var]],
                    f"{o.name}__Identity_rename",
                )
            )
        return nodes


def embedded(model: onnx.ModelProto):
    """
    Create an operator constructor embedding a given ``ModelProto``.

    Parameters
    ----------
    model
        Model to embed.
    Returns
    -------
    Callable
        When called with keyword arguments (model input names into vars), applies the
        embedded model and returns a dictionary of the outputs (model output names into vars).
    """

    def embed(**inputs: Var) -> Dict[str, Var]:
        """Local function created by ``embedded``. Call with expected embedded model inputs to get model outputs."""
        assert set(inputs) == {i.name for i in model.graph.input}
        node = _Embedded(
            attrs=None,
            inputs=_Embedded.Inputs([inputs[i.name] for i in model.graph.input]),
            out_variadic=len(model.graph.output),
            model=model,
        )
        return {o.name: var for o, var in zip(model.graph.output, node.outputs.outputs)}

    return embed


class _Introduce(_InternalNode):
    """Internal operator used for introducing values, to manually evaluate them in the current scope."""

    @dataclass
    class Attributes:
        pass

    class Inputs(VarFields):
        inputs: Sequence[Var]

    class Outputs(VarFields):
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
        from ._config import get_default_opset

        return {("", get_default_opset()._OPERATORS["Identity"].op_type.version)}

    def to_onnx(
        self, scope: Scope, doc_string: Optional[str] = None, build_subgraph=None
    ) -> List[onnx.NodeProto]:
        assert len(self.inputs.inputs) == len(self.outputs.outputs)
        # Just create a renaming identity from what we forwarded into our actual output
        protos = []
        name = scope.node[self] if self in scope.node else None
        for i in range(len(self.inputs.inputs)):
            protos.append(
                onnx.helper.make_node(
                    "Identity",
                    [scope.var[self.inputs.inputs[i]]],
                    [scope.var[self.outputs.outputs[i]]],
                    name + f"_id{i}" if name is not None else None,
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
