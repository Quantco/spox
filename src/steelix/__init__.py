import onnx.helper as _helper  # noqa
import pkg_resources

# Apply fix for https://github.com/onnx/onnxmltools/issues/575 if we have onnxmltools
try:
    _make_tensor = _helper.make_tensor
    import onnxmltools.proto as _omt_proto  # noqa

    _helper.make_tensor = _make_tensor
except ModuleNotFoundError:
    pass

from . import (
    arrow,
    arrowfields,
    attr,
    attrfields,
    config,
    converter,
    fields,
    function,
    graph,
    internal_op,
    node,
    schemas,
    shape,
    standard,
    type_system,
)
from ._build import BuildError as _BuildError
from ._scope import ScopeError as _ScopeError
from .arrow import Arrow, result_type
from .arrowfields import ArrowFields, NoArrows
from .attr import Attr, from_array, to_array
from .attrfields import AttrFields, NoAttrs
from .converter import convert, converters
from .graph import (
    arguments,
    arguments_dict,
    enum_arguments,
    enum_results,
    results,
    subgraph,
)
from .internal_op import embedded, intro, unsafe_cast, unsafe_reshape
from .node import Node, OpType
from .shape import ShapeError as _ShapeError
from .type_system import Optional, Sequence, Tensor, Type

__all__ = [
    "arrow",
    "arrowfields",
    "attr",
    "attrfields",
    "config",
    "converter",
    "fields",
    "function",
    "graph",
    "internal_op",
    "node",
    "schemas",
    "shape",
    "standard",
    "type_system",
    "convert",
    "converters",
    "_BuildError",
    "_ScopeError",
    "_ShapeError",
    "Arrow",
    "result_type",
    "ArrowFields",
    "NoArrows",
    "Attr",
    "AttrFields",
    "NoAttrs",
    "from_array",
    "to_array",
    "enum_arguments",
    "enum_results",
    "arguments",
    "arguments_dict",
    "results",
    "subgraph",
    "embedded",
    "intro",
    "unsafe_cast",
    "unsafe_reshape",
    "Node",
    "OpType",
    "Optional",
    "Sequence",
    "Tensor",
    "Type",
]

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"
