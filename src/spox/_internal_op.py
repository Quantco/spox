"""
Module for Spox operators that implement special internal behaviour that does not fit into the ONNX IR.
They behave like a normal Node, but their inference, building and translation behaviour may be overriden.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy
import onnx

from . import _value_prop
from ._attributes import AttrString, AttrTensor, AttrType
from ._fields import BaseAttributes, BaseInputs, BaseOutputs
from ._node import Node, OpType
from ._scope import Scope
from ._shape import SimpleShape
from ._type_system import Tensor, Type
from ._utils import from_array
from ._value_prop import PropValueType
from ._var import Var


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
        type: AttrType
        default: AttrTensor

    @dataclass
    class Outputs(BaseOutputs):
        arg: Var

    attrs: Attributes
    inputs: BaseInputs
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


class _Constant(_InternalNode):
    """Internal operator allowing usage of a universal-versioned Constant operator."""

    op_type = OpType("Constant", "spox.internal", 0)
    version: Optional[int]

    @dataclass
    class Attributes(BaseAttributes):
        value: AttrTensor

    @dataclass
    class Inputs(BaseInputs):
        pass

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

    def post_init(self, **kwargs):
        self.version = kwargs.get("version")

    def infer_output_types(self) -> Dict[str, Type]:
        # Output type is based on the value of the type attribute
        value = self.attrs.value.value
        return {"output": Tensor(value.dtype, value.shape)}

    def propagate_values(self) -> Dict[str, PropValueType]:
        return {"output": self.attrs.value.value}

    @property
    def opset_req(self) -> Set[Tuple[str, int]]:
        return {("", self.version)} if self.version is not None else set()

    def to_onnx(
        self,
        scope: "Scope",
        doc_string=None,
        build_subgraph=None,
    ) -> List[onnx.NodeProto]:
        return [
            onnx.helper.make_node(
                "Constant",
                [],
                [scope.var[self.outputs.output]],
                scope.node[self],
                value=from_array(self.attrs.value.value),
            )
        ]


def constant(value: numpy.ndarray, version: Optional[int]) -> Var:
    return _Constant(
        _Constant.Attributes(AttrTensor(value)), version=version
    ).outputs.output


class _Embedded(_InternalNode):
    """Internal operator used for embedding an existing ONNX ModelProto inside a Spox graph."""

    model: onnx.ModelProto

    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        inputs: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
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
        constant_req = (
            {("", 11)}
            if self.graph.sparse_initializer
            else {("", 9)}
            if self.graph.initializer
            else set()
        )
        return {
            (imp.domain, imp.version) for imp in self.model.opset_import
        } | constant_req

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

    def propagate_values(self) -> Dict[str, _value_prop.PropValueType]:
        if any(
            var.type is None or var._value is None
            for var in self.inputs.get_vars().values()
        ):
            return {}
        wrap_feed, run, unwrap_feed = _value_prop.get_backend_calls()
        input_feed = {
            i.name: wrap_feed(var._value)
            for i, var in zip(self.model.graph.input, self.inputs.inputs)
        }
        output_feed = run(self.model, input_feed)
        return {
            f"outputs_{k}": unwrap_feed(var.unwrap_type(), output_feed[o.name]).value
            for k, (o, var) in enumerate(zip(self.graph.output, self.outputs.outputs))
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
        nodes: List[onnx.NodeProto] = []
        # Move initializers to Constant nodes
        input_names = {i.name for i in graph.input}
        nodes.extend(
            onnx.helper.make_node("Constant", [], [i.name], value=i)
            for i in graph.initializer
            if i.name not in input_names
        )
        nodes.extend(
            onnx.helper.make_node("Constant", [], [i.values.name], sparse_value=i)
            for i in graph.sparse_initializer
            if i.values.name not in input_names
        )
        # Apply a trivial renaming of inputs
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
        # This is a questionable default (this operator is used in every graph),
        # but there's not much else to do that doesn't lower-bound the version in an implicit way.
        # The assumption here is that no-one will need graphs which only have Introduce nodes.
        return {("", 1)}

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
