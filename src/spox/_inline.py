# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import onnx

from spox._exceptions import BuildError
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._internal_op import INTERNAL_MIN_OPSET, _InternalNode
from spox._node import OpType
from spox._scope import Scope
from spox._type_system import Type
from spox._var import _VarInfo

from . import _value_prop


def rename_in_graph(
    graph_: onnx.GraphProto,
    rename: Callable[[str], str],
    *,
    rename_node: Callable[[str], str] | None = None,
    rename_op: Callable[[str, str], tuple[str, str]] | None = None,
) -> onnx.GraphProto:
    def rename_in_subgraph(subgraph: onnx.GraphProto) -> onnx.GraphProto:
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
        inputs: Sequence[_VarInfo]

    @dataclass
    class Outputs(BaseOutputs):
        outputs: Sequence[_VarInfo]

    op_type = OpType("Inline", "spox.internal", 0)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

    def __init__(
        self,
        attrs: BaseAttributes | None = None,
        inputs: BaseInputs | None = None,
        outputs: BaseOutputs | None = None,
        *,
        out_variadic: int | None = None,
        infer_types: bool = True,
        validate: bool = True,
        model: onnx.ModelProto,
    ) -> None:
        self.model = model
        super().__init__(
            attrs,
            inputs,
            outputs,
            out_variadic=out_variadic,
            infer_types=infer_types,
            validate=validate,
        )

    @property
    def graph(self) -> onnx.GraphProto:
        return self.model.graph

    @property
    def opset_req(self) -> set[tuple[str, int]]:
        return {(imp.domain, imp.version) for imp in self.model.opset_import} | {
            ("", INTERNAL_MIN_OPSET)
        }

    def infer_output_types(
        self, input_prop_values: _value_prop.PropDict
    ) -> dict[str, Type]:
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

    def propagate_values(
        self, input_prop_values: _value_prop.PropDict
    ) -> _value_prop.PropDict:
        if any(
            var_info.type is None or input_prop_values.get(var_info.name) is None
            for var_info in self.model.graph.input
        ):
            return {}

        res = _value_prop.infer(self.model, input_prop_values)
        return {
            f"outputs_{i}": res[info.name]
            for i, info in enumerate(self.model.graph.output)
            if info.name in res
        }

    def to_onnx(
        self,
        scope: Scope,
        doc_string: str | None = None,
        build_subgraph: Callable | None = None,
    ) -> list[onnx.NodeProto]:
        input_names: dict[str, int] = {
            p.name: i for i, p in enumerate(self.graph.input)
        }
        output_names: dict[str, int] = {
            p.name: i for i, p in enumerate(self.graph.output)
        }
        inner_renames: dict[str, str] = {}
        inner_node_renames: dict[str, str] = {}

        def reserve_prefixed(name: str) -> str:
            if not name:
                return name
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
        nodes: list[onnx.NodeProto] = list(graph.node)
        return nodes
