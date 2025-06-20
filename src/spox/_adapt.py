# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import onnx
import onnx.version_converter

from ._attributes import AttrGraph
from ._inline import _Inline
from ._internal_op import _InternalNode
from ._node import Node
from ._schemas import SCHEMAS
from ._scope import Scope
from ._utils import make_model
from ._var import _VarInfo


def adapt_node(
    node: Node,
    proto: onnx.NodeProto,
    source_version: int,
    target_version: int,
    var_names: dict[_VarInfo, str],
) -> list[onnx.NodeProto] | None:
    if source_version == target_version:
        return None

    try:
        # By using a dictionary we ensure that we only have a single
        # ValueInfo per (possibly repeated) input name.
        input_info = {
            var_names[var_info]: var_info.unwrap_type()._to_onnx_value_info(
                var_names[var_info], _traceback_name=f"adapt-input {key}"
            )
            for key, var_info in node.inputs.get_var_infos().items()
        }
        output_info = [
            var_info.unwrap_type()._to_onnx_value_info(
                var_names[var_info], _traceback_name=f"adapt-output {key}"
            )
            for key, var_info in node.outputs.get_var_infos().items()
        ]
    except ValueError:
        return None

    source_model = make_model(
        onnx.helper.make_graph(
            [proto],
            "spox__singleton_adapter_graph",
            list(input_info.values()),
            output_info,
        ),
        opset_imports=[onnx.helper.make_operatorsetid("", source_version)],
    )
    onnx.checker.check_model(source_model, full_check=True)
    target_model = onnx.version_converter.convert_version(source_model, target_version)

    return list(target_model.graph.node)


def adapt_inline(
    node: _Inline,
    protos: list[onnx.NodeProto],
    target_opsets: dict[str, int],
    var_names: dict[_VarInfo, str],
    node_name: str,
) -> list[onnx.NodeProto]:
    source_version = max({v for d, v in node.opset_req if d in ("", "ai.onnx")})
    target_version = target_opsets[""]

    # convert_version fails if the inlined model does not import the default domain
    seen_domains = {prot.domain for prot in protos}
    if not seen_domains & {"", "ai.onnx"}:
        return protos
    if source_version != target_version:
        target_model = onnx.version_converter.convert_version(
            node.model, target_version
        )
        base_model = node.model
        try:
            node.model = target_model
            target_nodes = node.to_onnx(Scope.of((node, node_name), *var_names.items()))
        finally:
            node.model = base_model
        return target_nodes
    return protos


def adapt_best_effort(
    node: Node,
    protos: list[onnx.NodeProto],
    opsets: dict[str, int],
    var_names: dict[_VarInfo, str],
    node_names: dict[Node, str],
) -> list[onnx.NodeProto] | None:
    if isinstance(node, _Inline):
        return adapt_inline(
            node,
            protos,
            opsets,
            var_names,
            node_names[node],
        )
    if isinstance(node, _InternalNode) or len(protos) != 1:
        return None
    if any(isinstance(attr, AttrGraph) for attr in node.attrs.get_fields().values()):
        return None
    proto: onnx.NodeProto
    (proto,) = protos
    domain = proto.domain if proto.domain != "ai.onnx" else ""

    reqs = {v for d, v in node.opset_req if d == domain}
    source_version = max(reqs)
    target_version = opsets[domain]

    version_mismatch = source_version != target_version
    if version_mismatch:
        source_schema = (
            SCHEMAS.get(domain, {}).get(source_version, {}).get(proto.op_type)
        )
        target_schema = (
            SCHEMAS.get(domain, {}).get(target_version, {}).get(proto.op_type)
        )
        if source_schema is not None and target_schema is not None:
            version_mismatch = source_schema != target_schema
    if not version_mismatch:
        return None

    if proto.domain not in ("", "ai.onnx"):
        warnings.warn(
            RuntimeWarning(
                "Node adapters are only supported for the default domain (ai.onnx), "
                f"but {proto.domain!r} is at {target_version} versus requested "
                f"{source_version} of {node_names[node]}.",
            )
        )
        return None

    adapted = adapt_node(
        node,
        proto,
        source_version,
        target_version,
        var_names,
    )
    return adapted
