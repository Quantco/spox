import onnx.helper as _helper  # noqa
import pkg_resources

# Apply fix for https://github.com/onnx/onnxmltools/issues/575 if we have onnxmltools
try:
    _make_tensor = _helper.make_tensor
    import onnxmltools.proto as _omt_proto  # noqa

    _helper.make_tensor = _make_tensor
except ModuleNotFoundError:
    pass

from spox._type_system import Optional, Sequence, Tensor
from spox._var import Var

__all__ = [
    "Var",
    "Tensor",
    "Sequence",
    "Optional",
]

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"
