import pkg_resources

from spox._public import argument, build, inline
from spox._type_system import Optional, Sequence, Tensor, Type
from spox._var import Var

__all__ = [
    "Var",
    "Type",
    "Tensor",
    "Sequence",
    "Optional",
    "argument",
    "build",
    "inline",
]

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"
