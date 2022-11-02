"""
Module for Steelix operators that implement special internal behaviour that does not fit into the ONNX IR.
They behave like a normal Node, but their inference, building and translation behaviour may be overriden.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import onnx

from ._attributes import AttrString, AttrTensor, AttrType
from ._scope import Scope
from .arrow import Arrow
from .arrowfields import ArrowFields, NoArrows
from .node import Node, OpType
from .shape import Shape, SimpleShape
from .type_system import Tensor, Type


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

    op_type = OpType("Argument", "steelix.internal", 0)

    @dataclass
    class Attributes:
        type: AttrType
        name: Optional[AttrString] = None
        default: Optional[AttrTensor] = None

    class Outputs(ArrowFields):
        arg: Arrow

    attrs: Attributes
    inputs: NoArrows
    outputs: Outputs

    def post_init(self, **kwargs):
        if self.attrs.name is not None:
            self.outputs.arg._rename(self.attrs.name.value)

    def infer_output_types(self) -> Dict[str, Type]:
        # Output type is based on the value of the type attribute
        return {"arg": self.attrs.type.value}

    def update_metadata(self, opset_req, initializers, functions):
        super().update_metadata(opset_req, initializers, functions)
        arrow = self.outputs.arg
        if self.attrs.default is not None:
            initializers[arrow] = self.attrs.default.value

    def to_onnx(
        self, scope: "Scope", doc_string: Optional[str] = None, build_subgraph=None
    ) -> List[onnx.NodeProto]:
        return []


class _Initializer(_InternalNode):
    """Internal operator representing a non-overridable initializer."""

    op_type = OpType("Initializer", "steelix.internal", 0)

    @dataclass
    class Attributes:
        type: AttrType
        default: AttrTensor

    class Outputs(ArrowFields):
        arg: Arrow

    attrs: Attributes
    inputs: NoArrows
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
    """Internal operator used for embedding an existing ONNX ModelProto inside a Steelix graph."""

    model: onnx.ModelProto

    @dataclass
    class Attributes:
        pass

    class Inputs(ArrowFields):
        inputs: Sequence[Arrow]

    class Outputs(ArrowFields):
        outputs: Sequence[Arrow]

    op_type = OpType("Embedded", "steelix.internal", 0)

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
        for i, arrow in zip(self.graph.input, self.inputs.inputs):
            if arrow.type is not None and not (arrow.type <= Type.from_onnx(i.type)):
                raise TypeError(
                    f"Embedded model input {i.name} type {arrow.type} "
                    f"does not match expected {Type.from_onnx(i.type)}."
                )
        # If we do, take the types as declared in the model
        return {
            f"outputs_{k}": Type.from_onnx(o.type)
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
        for i, arrow in zip(graph.input, self.inputs.inputs):
            nodes.append(
                onnx.helper.make_node(
                    "Identity",
                    [scope.arrow[arrow]],
                    [i.name],
                    f"{i.name}__Identity_rename",
                )
            )
        # Then graph body
        nodes.extend(graph.node)
        # Finish with output renaming
        for o, arrow in zip(graph.output, self.outputs.outputs):
            nodes.append(
                onnx.helper.make_node(
                    "Identity",
                    [o.name],
                    [scope.arrow[arrow]],
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
        When called with keyword arguments (model input names into arrows), applies the
        embedded model and returns a dictionary of the outputs (model output names into arrows).
    """

    def embed(**inputs: Arrow) -> Dict[str, Arrow]:
        """Local function created by ``embedded``. Call with expected embedded model inputs to get model outputs."""
        assert set(inputs) == {i.name for i in model.graph.input}
        node = _Embedded(
            attrs=None,
            inputs=_Embedded.Inputs([inputs[i.name] for i in model.graph.input]),
            out_variadic=len(model.graph.output),
            model=model,
        )
        return {
            o.name: arrow for o, arrow in zip(model.graph.output, node.outputs.outputs)
        }

    return embed


class _Introduce(_InternalNode):
    """Internal operator used for introducing values, to manually evaluate them in the current scope."""

    @dataclass
    class Attributes:
        pass

    class Inputs(ArrowFields):
        inputs: Sequence[Arrow]

    class Outputs(ArrowFields):
        outputs: Sequence[Arrow]

    op_type = OpType("Introduce", "steelix.internal", 0)

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
        from . import config

        return {("", config.get_default_opset()._OPERATORS["Identity"].op_type.version)}

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
                    [scope.arrow[self.inputs.inputs[i]]],
                    [scope.arrow[self.outputs.outputs[i]]],
                    name + f"_id{i}" if name is not None else None,
                    doc_string,
                )
            )
        return protos


def intro(*args: Arrow) -> Arrow:
    """
    Evaluates all the argument Arrows in the current scope, and returns the last.

    Useful when a temporary value is reused in multiple subgraphs, but it should be defined in the outer scope.
    Otherwise, every subgraph will expand its definition inside it.

    The basis of this introduction is that we introduce a dependency of the returned output on all inputs.
    Hence, if we use this in a context like ``intro(x, ..., operator(..., subgraph(...x...)))``
    then ``operator`` explicitly depends on ``x``, bringing it into ``operator``'s scope. Then ``subgraph``
    is built afterwards and may access ``x``. Without this, it is not guaranteed that ``x`` does not only
    end up in the inner scope of the ``subgraph``. This may be desirable to prevent a behaviour where ``x``
    is a complicated done that is reused (and hence inlined) in many subgraphs, but never in the main scope.

    This is often also used as a variadic Identity, which is sometimes needed for ONNX IR or to simplify behaviour.
    As such, it brings in ``config.default_opset``.
    """
    return intros(*args)[-1]


def intros(*args: Arrow) -> Sequence[Arrow]:
    """Same as intro, but all the arguments are returned & made dependent on each other, and not only the last."""
    return _Introduce(
        None, _Introduce.Inputs(args), out_variadic=len(args)
    ).outputs.outputs


def unsafe_cast(x: Arrow, typ: Type) -> Arrow:
    """
    Creates a new arrow with the type forcefully set to ``typ``.

    Assumes that the real type of the Arrow is indeed compatible with ``shape`` (for example it was unknown).

    The function is meant for use when type inference failed, and it has to be overriden to avoid further failures.

    If you want to properly change an Arrow's type, use an operator like Cast, CastLike, Optional, etc.

    Parameters
    ----------
    x
        Arrow to retype.
    typ
        Target type - must be a constant.

    Returns
    -------
    Arrow
        Arrow with the type reset to whatever was given.
    """
    x = intro(x)
    x.type = typ
    return x


def unsafe_reshape(x: Arrow, shape: Union[Shape, SimpleShape]) -> Arrow:
    """
    Creates a new arrow with the shape forcefully set to ``shape`` (like an unsafe cast).

    Assumes that the real shape of the Arrow is indeed compatible with ``shape`` (for example it was unknown).

    The function is meant for use when shape inference failed, and it has to be overriden to avoid failures.

    If you want to reshape to the shape of another arrow, use a Reshape operator.

    Parameters
    ----------
    x
        Arrow to reshape.
    shape
        Target shape - must be a constant.
    Returns
    -------
    Arrow
        Arrow with the same Tensor element type, but different shape.
    """
    return unsafe_cast(x, Tensor(x.unwrap_tensor().elem_type, shape))
