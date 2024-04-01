"""Module implementing a base for standard ONNX operators, which use the functionality of ONNX node-level inference."""

from typing import TYPE_CHECKING, Callable, Dict, Tuple

import numpy
import onnx
import onnx.reference
import onnx.shape_inference
from onnx.defs import OpSchema

from . import _value_prop
from ._exceptions import InferenceError
from ._node import Node
from ._schemas import SCHEMAS
from ._scope import Scope
from ._shape import SimpleShape
from ._type_system import Optional, Sequence, Tensor, Type
from ._utils import from_array
from ._value_prop import PropValueType

if TYPE_CHECKING:
    from ._graph import Graph


class StandardNode(Node):
    """
    Base type for a Node which has a known reference Schema (``self.schema``), extracted based on the ``op_type``.
    Implements ``infer_output_types_onnx``, which is used for type inference.
    This is based on the functionality of ``onnx.shape_inference.infer_node_outputs``.
    The default ``infer_output_types`` is overriden to use this function.
    """

    @classmethod
    def get_schema(cls) -> OpSchema:
        return SCHEMAS[cls.op_type.domain][cls.op_type.version][cls.op_type.identifier]

    @property
    def schema(self) -> OpSchema:
        return self.get_schema()

    @property
    def min_input(self) -> int:
        return self.schema.min_input

    @property
    def min_output(self) -> int:
        return self.schema.min_output

    def to_singleton_onnx_model(
        self, *, dummy_outputs: bool = True, with_dummy_subgraphs: bool = True
    ) -> Tuple[onnx.ModelProto, Scope]:
        """
        Build a singleton model consisting of just this StandardNode. Used for type inference.
        Dummy subgraphs are typed, but have no graph body, so that we can avoid the build cost.
        They refer to non-existent nodes, but ONNX does not raise an error (for now?).
        """
        # Prepare names for the values in scope of the node
        scope = Scope()
        scope.node[self] = "_this_"
        for key, var in self.inputs.get_vars().items():
            if var not in scope.var:
                scope.var[var] = key
        for key, var in self.outputs.get_vars().items():
            if var not in scope.var:
                scope.var[var] = key
        # We inject the evaluated attribute values here and then substitute back
        self_attrs = self.attrs
        try:
            current_fields = self_attrs.get_fields().items()
            self.attrs = self.Attributes(
                **{k: v.deref() if v is not None else None for k, v in current_fields}
            )
            node_proto: onnx.NodeProto
            # Subgraphs are not fully built for possibly significant performance gains.
            # However, this uses a trick so that they type correctly.
            # This may throw if we are building ``not with_subgraphs``.
            build_subgraph = _make_dummy_subgraph if with_dummy_subgraphs else None
            (node_proto,) = self.to_onnx(scope, build_subgraph=build_subgraph)
        finally:
            self.attrs = self_attrs
        # Create a singleton graph for type inference with our node
        # Input types
        input_info = [
            var.unwrap_type()._to_onnx_value_info(key)
            for key, var in self.inputs.get_vars().items()
        ]

        # Output types with placeholder empty TypeProto (or actual type if not using dummies)
        def out_value_info(curr_key, curr_var):
            if dummy_outputs or curr_var.type is None or not curr_var.type._is_concrete:
                return onnx.helper.make_value_info(curr_key, onnx.TypeProto())
            return curr_var.unwrap_type()._to_onnx_value_info(curr_key)

        output_info = [
            out_value_info(key, var) for key, var in self.outputs.get_vars().items()
        ]
        # Initializers, passed in to allow partial data propagation
        #  - used so that operators like Reshape are aware of constant shapes
        initializers = [
            from_array(var._value.value, key)
            for key, var in self.inputs.get_vars().items()
            if var._value and isinstance(var._value.value, numpy.ndarray)
        ]
        #  Graph and model
        graph = onnx.helper.make_graph(
            [node_proto],
            "StandardOpNode_infer_output_types_onnx",
            input_info,
            output_info,
            initializers,
        )
        # Subgraph internals are hidden by make_dummy_subgraph - so we don't care about subgraphs' opset requirements
        model = onnx.helper.make_model(
            graph,
            opset_imports=[
                onnx.helper.make_operatorsetid(
                    self.op_type.domain, self.op_type.version
                )
            ],
        )
        return model, scope

    def infer_output_types_onnx(self) -> Dict[str, Type]:
        """Execute type & shape inference with ``onnx.shape_inference.infer_node_outputs``."""
        # Check that all (specified) inputs have known types, as otherwise we fail
        if any(var.type is None for var in self.inputs.get_vars().values()):
            return {}

        model, _ = self.to_singleton_onnx_model()

        # Attempt to do shape inference - if an error is caught, we extend the traceback a bit
        try:
            typed_model = onnx.shape_inference.infer_shapes(
                model, check_type=True, strict_mode=True, data_prop=True
            )
        except InferenceError as e:
            raise type(e)(
                f"{str(e)} -- for {self.schema.name}: {self.signature}"
            ) from e

        # Recover the types from protobuf, ignoring empty protobuf objects for failing/unimplemented type inference.
        results = {
            info.name: Type._from_onnx(info.type)
            for info in typed_model.graph.output
            if info.type != onnx.TypeProto()
        }
        # Strips some unuseful type data (unknown dimensions become global-scoped dimension parameters).
        return {
            key: _strip_dim_symbol(type_, lambda x: x.startswith("unk__"))
            for key, type_ in results.items()
        }

    def propagate_values_onnx(self) -> Dict[str, PropValueType]:
        """Perform value propagation by evaluating singleton model.

        The backend used for the propagation can be configured with the `spox._standard.ValuePropBackend` variable.
        """
        # Cannot do propagation when some inputs were not propagated/inferred
        if any(
            var.type is None or var._value is None
            for var in self.inputs.get_vars().values()
        ):
            return {}
        if next(iter(self.subgraphs), None) is not None:
            # Cannot do propagation with subgraphs implicitly for performance - should be reimplemented
            return {}
        model, scope = self.to_singleton_onnx_model(with_dummy_subgraphs=False)
        wrap_feed, run, unwrap_feed = _value_prop.get_backend_calls()
        input_feed = {
            scope.var[var]: wrap_feed(var._value)
            for var in self.inputs.get_vars().values()
            if var._value
        }

        output_feed = run(model, input_feed)

        results = {
            scope.var[str(name)]._which_output: unwrap_feed(
                scope.var[str(name)].unwrap_type(), result
            ).value
            for name, result in output_feed.items()
        }
        return {k: v for k, v in results.items() if k is not None}

    def infer_output_types(self) -> Dict[str, Type]:
        return self.infer_output_types_onnx()

    def propagate_values(self) -> Dict[str, PropValueType]:
        if _value_prop._VALUE_PROP_BACKEND != _value_prop.ValuePropBackend.NONE:
            return self.propagate_values_onnx()
        return {}


def _strip_dim_symbol_shape(
    shape: SimpleShape, pred: Callable[[str], bool]
) -> SimpleShape:
    """
    Remove all instances all ``unk__`` dimension parameters from a shape -- created when running
    ONNX shape inference and no parameter to use existed. It is stripped to avoid conflicts (due to global scoping).
    """
    if shape is None:
        return shape
    xs = [None if isinstance(x, str) and pred(x) else x for x in shape]
    return tuple(xs)


def _strip_dim_symbol(typ: Type, pred: Callable[[str], bool]) -> Type:
    """Apply ``_strip_unk_param_shape`` to all shapes present in this ``Type``."""
    if isinstance(typ, Tensor):
        return Tensor(typ.dtype, _strip_dim_symbol_shape(typ.shape, pred))
    elif isinstance(typ, Sequence):
        return Sequence(_strip_dim_symbol(typ.elem_type, pred))
    elif isinstance(typ, Optional):
        return Optional(_strip_dim_symbol(typ.elem_type, pred))
    else:
        return typ


def _make_dummy_subgraph(_node: Node, key: str, graph: "Graph") -> onnx.GraphProto:
    """
    Make a dummy GraphProto that has inputs and outputs typed like Graph, without a graph body.

    Assumes ``graph`` requests arguments to avoid building it. The body is avoided by having the outputs be equal to
    unknown forwarded values from the outer scope.

    - inputs: ``-> __dummy_input{i}``
    - outputs: ``<- dummy_output{i} <- __dummy_outer_output{i} <- ?``

    Warning: this method may break if ONNX starts throwing errors against missing value names in shape inference.
    """
    if graph.requested_arguments is None:
        raise RuntimeError("Subgraph must have requested_arguments to construct dummy.")

    inputs = []
    for i, arr in enumerate(graph.requested_arguments):
        inputs.append(arr.unwrap_type()._to_onnx_value_info(f"__dummy_input{i}"))

    value_infos = []
    nodes = []
    outputs = []
    for i, arr in enumerate(graph.requested_results.values()):
        outer = f"__dummy_outer_output{i}"
        value_infos.append(arr.unwrap_type()._to_onnx_value_info(outer))
        out = f"__dummy_output{i}"
        outputs.append(arr.unwrap_type()._to_onnx_value_info(out))
        nodes.append(onnx.helper.make_node("Identity", [outer], [out]))
    return onnx.helper.make_graph(
        nodes, f"__dummy_{key}", inputs, outputs, value_info=value_infos
    )
