import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import onnx

from spox._exceptions import BuildError
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._internal_op import INTERNAL_MIN_OPSET, _InternalNode
from spox._node import OpType
from spox._scope import Scope
from spox._type_system import Type
from spox._var import Var

from . import _value_prop


def rename_in_graph(
    graph_: onnx.GraphProto,
    rename: Callable[[str], str],
    *,
    rename_node: Optional[Callable[[str], str]] = None,
    rename_op: Optional[Callable[[str, str], Tuple[str, str]]] = None,
) -> onnx.GraphProto:
    def rename_in_subgraph(subgraph):
        return rename_in_graph(
            subgraph,
            rename,
            rename_node=rename_node,
            rename_op=rename_op,
        )

    graph = onnx.GraphProto()
    graph.CopyFrom(graph_)

    for p in itertools.chain(graph.input, graph.initializer):
        p.name = rename(p.name)
    for si in graph.sparse_initializer:
        si.values.name = rename(si.values.name)
        si.indices.name = rename(si.indices.name)

    for nd in graph.node:
        if nd.name and rename_node is not None:
            nd.name = rename_node(nd.name)
        if rename_op is not None:
            # This is a bit elaborate, but we do it this way as
            # an unset domain field is different from an empty one.
            if nd.HasField("domain"):
                nd.domain, nd.op_type = rename_op(nd.domain, nd.op_type)
            else:
                # An empty domain is the default domain (ai.onnx)
                domain, nd.op_type = rename_op("", nd.op_type)
                if domain:  # Only set the domain explicitly if it's changing
                    nd.domain = domain
        for seq in (nd.input, nd.output):
            for i, name in enumerate(seq):
                seq[i] = rename(name)
        for attr_proto in nd.attribute:
            attr = onnx.helper.get_attribute_value(attr_proto)
            if isinstance(attr, onnx.GraphProto):
                attr_proto.g.CopyFrom(rename_in_subgraph(attr))
            elif isinstance(attr, list) and all(
                isinstance(g, onnx.GraphProto) for g in attr
            ):
                for i, sub in enumerate(attr):
                    attr_proto.graphs[i].CopyFrom(rename_in_subgraph(sub))

    for p in itertools.chain(graph.output, graph.value_info):
        p.name = rename(p.name)

    return graph


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
            if o.name in output_feed
        }

    def to_onnx(
        self, scope: Scope, doc_string: Optional[str] = None, build_subgraph=None
    ) -> List[onnx.NodeProto]:
        input_names: Dict[str, int] = {
            p.name: i for i, p in enumerate(self.graph.input)
        }
        output_names: Dict[str, int] = {
            p.name: i for i, p in enumerate(self.graph.output)
        }
        inner_renames: Dict[str, str] = {}
        inner_node_renames: Dict[str, str] = {}

        def reserve_prefixed(name: str) -> str:
            return scope.var.reserve(
                scope.var.maybe_enum(f"{scope.node[self]}__{name}")
            )

        def apply_rename(name: str) -> str:
            if name in input_names:
                return scope.var[self.inputs.inputs[input_names[name]]]
            if name in output_names:
                return scope.var[self.outputs.outputs[output_names[name]]]
            if name not in inner_renames:
                inner_renames[name] = reserve_prefixed(name)
            return inner_renames[name]

        def apply_node_rename(name: str) -> str:
            if name not in inner_node_renames:
                inner_node_renames[name] = reserve_prefixed(name)
            return inner_node_renames[name]

        graph = rename_in_graph(self.graph, apply_rename, rename_node=apply_node_rename)

        if graph.initializer:
            raise BuildError(
                "Inlined graph initializers should be handled beforehand and be removed from the graph."
            )
        nodes: List[onnx.NodeProto] = list(graph.node)
        return nodes
