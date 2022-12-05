"""Module implementing a base for standard ONNX operators, which use the functionality of ONNX node-level inference."""

import typing
from typing import Any, Dict, Tuple, Union

import numpy
import onnx
import onnx.shape_inference
from onnx.defs import OpSchema

from ._node import Node
from ._schemas import SCHEMAS
from ._scope import Scope
from ._shape import SimpleShape
from ._type_inference import InferenceError
from ._type_system import Optional, Sequence, Tensor, Type
from ._utils import from_array
from ._var import Nothing, _nil

if typing.TYPE_CHECKING:
    from .graph import Graph

try:
    import onnxruntime
except ImportError:
    onnxruntime = None  # type: ignore

_USE_ONNXRUNTIME_VALUE_PROP = False


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
        self, *, dummy_outputs: bool = True, with_subgraphs: bool = True
    ) -> Tuple[onnx.ModelProto, Scope]:
        """Build a singleton model consisting of just this StandardNode. Used for type inference."""
        # Prepare names for the values in scope of the node
        scope = Scope()
        scope.node[self] = "_this_"
        scope.var[_nil] = ""
        for key, var in self.inputs.as_dict().items():
            if var not in scope.var:
                scope.var[var] = key
        for key, var in self.outputs.as_dict().items():
            if var not in scope.var:
                scope.var[var] = key
        # We inject the evaluated attribute values here and then substitute back
        self_attrs = self.attrs
        try:
            # Get exact attribute values to run inference (as
            # otherwise refs aren't handled properly).
            self.attrs = self.Attributes(
                **{
                    k: type(v)(v.value) if v is not None else v
                    for k, v in self.attrs.__dict__.items()
                }
            )
            node_proto: onnx.NodeProto
            # Subgraphs are not fully built for possibly significant performance gains.
            # However, this uses a trick so that they type correctly.
            # This may throw if we are building ``not with_subgraphs``.
            build_subgraph = _make_dummy_subgraph if with_subgraphs else None
            (node_proto,) = self.to_onnx(scope, build_subgraph=build_subgraph)
        finally:
            self.attrs = self_attrs
        # Create a singleton graph for type inference with our node
        # Input types
        input_info = [
            var.unwrap_type()._to_onnx_value_info(key)
            for key, var in self.inputs.as_dict().items()
            if var
        ]

        # Output types with placeholder empty TypeProto (or actual type if not using dummies)
        def out_value_info(curr_key, curr_var):
            if dummy_outputs or curr_var.type is None or not curr_var.type._is_concrete:
                return onnx.helper.make_value_info(curr_key, onnx.TypeProto())
            return curr_var.unwrap_type()._to_onnx_value_info(curr_key)

        output_info = [
            out_value_info(key, var)
            for key, var in self.outputs.as_dict().items()
            if var
        ]
        # Initializers, passed in to allow partial data propagation
        #  - used so that operators like Reshape are aware of constant shapes
        initializers = [
            from_array(var._value, key)
            for key, var in self.inputs.as_dict().items()
            if isinstance(var._value, numpy.ndarray)
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
        if any(var.type is None for var in self.inputs.as_dict().values() if var):
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
        return {key: _strip_unk_param(type_) for key, type_ in results.items()}

    def propagate_values_onnx(self) -> Dict[str, Any]:
        """
        Perform value propagation by evaluating singleton models with ONNX Runtime.

        Assumes onnxruntime was imported successfully. Does not support subgraphs.
        """
        if any(var and var._value is None for var in self.inputs.as_dict().values()):
            # Cannot do propagation when some inputs were not propagated
            return {}
        if next(iter(self.subgraphs), None) is not None:
            # Cannot do propagation with subgraphs implicitly for performance - should be reimplemented
            return {}
        # Silence possible warnings during execution (especially constant folding)
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        # Set everything up for evaluation
        model, scope = self.to_singleton_onnx_model(with_subgraphs=False)
        session = onnxruntime.InferenceSession(model.SerializeToString(), options)
        input_feed = {
            scope.var[var]: _value_prop_to_ort(var._value)
            for var in self.inputs.as_dict().values()
        }
        # Get outputs and give a map from output field names
        output_feed = dict(zip(session.get_outputs(), session.run(None, input_feed)))
        return {
            scope.var[output.name]._which_output: _value_prop_from_ort(result)
            for output, result in output_feed.items()
        }

    def infer_output_types(self) -> Dict[str, Type]:
        return self.infer_output_types_onnx()

    def propagate_values(self) -> Dict[str, Any]:
        if _USE_ONNXRUNTIME_VALUE_PROP:
            if onnxruntime is None:
                raise RuntimeError(
                    "Cannot use ONNX Runtime value prop when ONNX Runtime isn't available "
                    "(ImportError was raised)."
                )
            return self.propagate_values_onnx()
        return {}


def _strip_unk_param_shape(shape: SimpleShape) -> SimpleShape:
    """
    Remove all instances all ``unk__`` dimension parameters from a shape -- created when running
    ONNX shape inference and no parameter to use existed. It is stripped to avoid conflicts (due to global scoping).
    """
    if shape is None:
        return shape
    xs = [None if isinstance(x, str) and x.startswith("unk__") else x for x in shape]
    return tuple(xs)


def _strip_unk_param(typ: Type) -> Type:
    """Apply ``_strip_unk_param_shape`` to all shapes present in this ``Type``."""
    if isinstance(typ, Tensor):
        return Tensor(typ.dtype, _strip_unk_param_shape(typ.shape))
    elif isinstance(typ, Sequence):
        return Sequence(_strip_unk_param(typ.elem_type))
    elif isinstance(typ, Optional):
        return Optional(_strip_unk_param(typ.elem_type))
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
    nodes = []
    outputs = []
    for i, arr in enumerate(graph.requested_results.values()):
        outer = f"__dummy_outer_output{i}"
        out = f"__dummy_output{i}"
        outputs.append(arr.unwrap_type()._to_onnx_value_info(out))
        nodes.append(onnx.helper.make_node("Identity", [outer], [out]))

    return onnx.helper.make_graph(nodes, f"__dummy_{key}", inputs, outputs)


def _value_prop_to_ort(value) -> Union[numpy.ndarray, list, None]:
    if value is Nothing:
        return None
    return value


def _value_prop_from_ort(value: Union[numpy.ndarray, list, None]):
    if value is None:
        return Nothing
    elif isinstance(value, list):
        return [_value_prop_from_ort(elem) for elem in value]
    elif isinstance(value, numpy.ndarray):
        # This looks ridiculous, but is required to normalise numpy.longlong back into a fixed size type.
        # ORT sometimes returns non-sized types (like longlong) and Var's value typecheck will fail because of it.
        # - numpy.dtype(longlong).type is longlong, but
        # - numpy.dtype(longlong) == numpy.dtype(int64), while
        # - longlong != int64
        return value.astype(numpy.dtype(value.dtype.name))
    raise TypeError(f"Cannot handle ORT value: {value}")