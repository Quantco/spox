import warnings
from typing import Dict, List, Optional

import numpy
import onnx
import onnx.version_converter

from . import attr
from ._scope import Scope
from .arrow import Arrow
from .internal_op import _Embedded, _InternalNode
from .node import Node
from .schemas import SCHEMAS


def adapt_node(
    node: Node,
    proto: onnx.NodeProto,
    source_version: int,
    target_version: int,
    arrow_names: Dict[Arrow, str],
) -> Optional[List[onnx.NodeProto]]:
    if source_version == target_version:
        return None

    try:
        input_info = [
            arrow.unwrap_type().to_onnx_value_info(
                arrow_names[arrow], _traceback_name=f"adapt-input {key}"
            )
            for key, arrow in node.inputs.as_dict().items()
            if arrow
        ]
        output_info = [
            arrow.unwrap_type().to_onnx_value_info(
                arrow_names[arrow], _traceback_name=f"adapt-output {key}"
            )
            for key, arrow in node.outputs.as_dict().items()
            if arrow
        ]
        initializers = [
            attr.from_array(arrow.value, name)
            for name, arrow in node.inputs.as_dict().items()
            if isinstance(arrow.value, numpy.ndarray)
        ]
    except ValueError:
        return None

    source_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [proto],
            "steelix__singleton_adapter_graph",
            input_info,
            output_info,
            initializers,
        ),
        opset_imports=[onnx.helper.make_operatorsetid("", source_version)],
    )
    onnx.checker.check_model(source_model, full_check=True)
    target_model = onnx.version_converter.convert_version(source_model, target_version)

    return list(target_model.graph.node)


def adapt_embedded(
    node: _Embedded,
    protos: List[onnx.NodeProto],
    target_opsets: Dict[str, int],
    arrow_names: Dict[Arrow, str],
    node_name: str,
) -> List[onnx.NodeProto]:
    source_version = max({v for d, v in node.opset_req if d in ("", "ai.onnx")})
    target_version = target_opsets[""]

    if source_version != target_version:
        target_model = onnx.version_converter.convert_version(
            node.model, target_version
        )
        base_model = node.model
        try:
            node.model = target_model
            target_nodes = node.to_onnx(Scope.of(*arrow_names.items()), node_name)
        finally:
            node.model = base_model
        return target_nodes
    return protos


def adapt_best_effort(
    node: Node,
    protos: List[onnx.NodeProto],
    opsets: Dict[str, int],
    arrow_names: Dict[Arrow, str],
    node_names: Dict[Node, str],
) -> Optional[List[onnx.NodeProto]]:
    if isinstance(node, _Embedded):
        return adapt_embedded(
            node,
            protos,
            opsets,
            arrow_names,
            node_names[node],
        )
    if isinstance(node, _InternalNode) or len(protos) != 1:
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
            "Node adapters are only supported for the default domain (ai.onnx), "
            f"but {proto.domain!r} is at {target_version} versus requested "
            f"{source_version} of {node_names[node]}.",
            RuntimeWarning,
        )
        return None

    adapted = adapt_node(
        node,
        proto,
        source_version,
        target_version,
        arrow_names,
    )
    return adapted
