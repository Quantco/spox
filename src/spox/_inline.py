from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import onnx

from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._internal_op import INTERNAL_MIN_OPSET, _InternalNode
from spox._node import OpType
from spox._scope import Scope
from spox._type_system import Type
from spox._var import Var

from . import _value_prop


class _Inline(_InternalNode):
    """Internal operator used for inlining (embedding) an existing ONNX ModelProto inside a Spox graph."""

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

    op_type = OpType("Inline", "spox.internal", 0)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

    def pre_init(self, **kwargs):
        self.model = kwargs["model"]

    @property
    def graph(self) -> onnx.GraphProto:
        return self.model.graph

    @property
    def opset_req(self) -> Set[Tuple[str, int]]:
        return {(imp.domain, imp.version) for imp in self.model.opset_import} | {
            ("", INTERNAL_MIN_OPSET)
        }

    def infer_output_types(self) -> Dict[str, Type]:
        # First, type check that we match the ModelProto type requirements
        for i, var in zip(self.graph.input, self.inputs.inputs):
            if var.type is not None and not (
                var.type._subtype(Type._from_onnx(i.type))
            ):
                raise TypeError(
                    f"Input '{i.name}' to inlined model got type {var.type}, "
                    f"expected {Type._from_onnx(i.type)}."
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
        name = scope.node[self]
        graph = onnx.GraphProto()
        graph.CopyFrom(self.graph)
        # FIXME: This is a bug upstream - when add_prefix_graph has rename_edges,
        #        unused inputs are not renamed. We apply identities to use the inputs.
        for i in graph.input:
            graph.node.append(
                onnx.helper.make_node(
                    "Identity", [i.name], [f"__{i.name}_Identity_dummy_use"]
                )
            )
        graph = onnx.compose.add_prefix_graph(graph, f"{name}__")
        for _ in graph.input:
            graph.node.pop()

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
