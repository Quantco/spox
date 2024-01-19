import importlib.metadata

from spox._public import argument, build, initializer, inline
from spox._type_system import Optional, Sequence, Tensor, Type
from spox._var import Var

__all__ = [
    "Var",
    "Type",
    "Tensor",
    "Sequence",
    "Optional",
    "argument",
    "initializer",
    "build",
    "inline",
]

try:
    __version__ = importlib.metadata.distribution(__name__).version
except Exception:
    __version__ = "unknown"
